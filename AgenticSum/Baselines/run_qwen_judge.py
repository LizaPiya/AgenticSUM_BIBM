"""
LLM-as-a-judge evaluation for Qwen2.5 summaries (Table 2).
Reads pre-generated CSVs from run_qwen_mimic.py / run_qwen_soap.py.

Usage:
    python run_qwen_judge.py --model_size 1b --dataset mimic
    python run_qwen_judge.py --model_size 1b --dataset soap
    python run_qwen_judge.py --model_size 3b --dataset mimic
    python run_qwen_judge.py --model_size 3b --dataset soap
"""

import argparse
import os
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_size", choices=["1b", "3b"], required=True)
parser.add_argument("--dataset",    choices=["mimic", "soap"], required=True)
args = parser.parse_args()

MODEL_TAG  = f"qwen2.5_{args.model_size}"
DATASET    = args.dataset

INPUT_CSV  = f"/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/{MODEL_TAG}_{DATASET}_summaries.csv"
OUTPUT_DIR = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum"
OUTPUT_CSV = f"{OUTPUT_DIR}/{MODEL_TAG}_{DATASET}_judge_results.csv"

JUDGE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN    = os.environ.get("HUGGINGFACE_HUB_TOKEN")

assert torch.cuda.is_available(), "CUDA not available"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Evaluating: {MODEL_TAG} on {DATASET.upper()}")
print(f"Input CSV:  {INPUT_CSV}\n")


# ──────────────────────────────────────────────
# Load judge model
# ──────────────────────────────────────────────
print(f"Loading judge model: {JUDGE_MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HF_TOKEN,
).eval()
print("Judge model loaded.\n")


# ──────────────────────────────────────────────
# Load generated summaries
# ──────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} samples from {INPUT_CSV}")


# ──────────────────────────────────────────────
# Evaluation prompt (matches paper Appendix B)
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = """Evaluate the generated medical summary against the source clinical document. Assign an integer score from 1 to 5 for each criterion defined below.

Source Document:
{source}

Generated Summary:
{summary}

Evaluation Criteria (1-5 scale):
- Hallucination: Degree of unsupported or fabricated content (1 = no hallucination; 5 = major fabrications)
- Factual Consistency: Faithfulness of statements to the source document (1 = highly inaccurate; 5 = fully accurate)
- Completeness: Coverage of core clinical information (1 = key information missing; 5 = fully comprehensive)
- Coherence: Fluency and logical organization of the summary (1 = poorly written; 5 = highly coherent)

Output Format: Return the scores using the following strict key-value format, with no additional text:
Hallucination: X
Factual: X
Complete: X
Coherent: X"""


def parse_scores(response):
    def extract(pattern, text, default=3.0):
        m = re.search(pattern, text)
        if m:
            return max(1.0, min(5.0, float(m.group(1))))
        return default

    return {
        "hallucination_score": extract(r"Hallucination:\s*(\d+)", response),
        "factual_consistency":  extract(r"Factual:\s*(\d+)",       response),
        "completeness":         extract(r"Complete:\s*(\d+)",       response),
        "coherence":            extract(r"Coherent:\s*(\d+)",       response),
    }


# ──────────────────────────────────────────────
# Run judge
# ──────────────────────────────────────────────
results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Judging"):
    try:
        source  = str(row["input"])[:2000]
        summary = str(row["generated_summary"])

        prompt = PROMPT_TEMPLATE.format(source=source, summary=summary)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.1,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        scores = parse_scores(response)

    except Exception as e:
        print(f"Error at index {idx}: {e}")
        scores = {"hallucination_score": 3.0, "factual_consistency": 3.0,
                  "completeness": 3.0, "coherence": 3.0}

    results.append(scores)

# ──────────────────────────────────────────────
# Save & print Table 2 row
# ──────────────────────────────────────────────
results_df = pd.DataFrame(results)
for col in results_df.columns:
    df[col] = results_df[col].values
df.to_csv(OUTPUT_CSV, index=False)

halluc = df["hallucination_score"]
factual = df["factual_consistency"]
complete = df["completeness"]
coherence = df["coherence"]

dataset_label = "MIMIC-IV" if DATASET == "mimic" else "SOAP Summary"
model_label   = "Qwen/Qwen2.5-1.5B-Instruct" if args.model_size == "1b" else "Qwen/Qwen2.5-3B-Instruct"

print("\n" + "="*65)
print(f"TABLE 2 ROW  —  {model_label}  ({dataset_label})")
print("="*65)
print(f"Hallucination:       {halluc.mean():.2f} ± {halluc.std():.2f}  (↓)")
print(f"Factual Consistency: {factual.mean():.2f} ± {factual.std():.2f}  (↑)")
print(f"Completeness:        {complete.mean():.2f} ± {complete.std():.2f}  (↑)")
print(f"Coherence:           {coherence.mean():.2f} ± {coherence.std():.2f}  (↑)")
print("="*65)
print(f"\nFull results saved to {OUTPUT_CSV}")
