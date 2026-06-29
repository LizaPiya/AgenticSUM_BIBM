"""
Generate and evaluate Qwen2.5 summaries on SOAP for Table 1.
Usage:
    python run_qwen_soap.py --model_size 1b
    python run_qwen_soap.py --model_size 3b
"""

import argparse
import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
from rouge_metric import PyRouge


# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_size", choices=["1b", "3b"], required=True,
                    help="Qwen model size: 1b (Qwen2.5-1.5B-Instruct) or 3b (Qwen2.5-3B-Instruct)")
args = parser.parse_args()

MODEL_MAP = {
    "1b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
}
MODEL_NAME = MODEL_MAP[args.model_size]
MODEL_TAG  = f"qwen2.5_{args.model_size}"

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

DATA_PATH   = "../Dataset/df_soap_mimic.csv"
OUTPUT_DIR  = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum"
OUTPUT_PATH = f"{OUTPUT_DIR}/{MODEL_TAG}_soap_summaries.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

assert torch.cuda.is_available(), "CUDA not available"
print(f"GPU: {torch.cuda.get_device_name(0)}")


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
print(f"\nLoading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    token=HF_TOKEN,
    attn_implementation="eager",
)
model.eval()
print("Model loaded.\n")


# ──────────────────────────────────────────────
# Generation  (SOAP prompt, same as draft_agent_soap.py)
# ──────────────────────────────────────────────
PROMPT_TEMPLATE = (
    "You are a helpful medical assistant. Analyze the following patient-doctor dialogue and "
    "create a concise medical summary in SOAP format (Subjective, Objective, Assessment, Plan). "
    "Focus on key symptoms, findings, diagnosis, and treatment plan:\n\n"
    "{input}\n\nSOAP Summary:"
)

def generate_summary(input_text: str) -> str:
    prompt = PROMPT_TEMPLATE.format(input=input_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded.replace(prompt, "").strip()


df = pd.read_csv(DATA_PATH).head(100).reset_index(drop=True)
print(f"Loaded {len(df)} SOAP notes.")

generated = []
for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating ({MODEL_TAG})"):
    try:
        summary = generate_summary(str(row["input"]))
    except Exception as e:
        print(f"  Generation error at {row['note_id']}: {e}")
        summary = ""
    generated.append(summary)

    if (idx + 1) % 10 == 0:
        gc.collect()
        torch.cuda.empty_cache()

df["generated_summary"] = generated
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSummaries saved to {OUTPUT_PATH}")


# ──────────────────────────────────────────────
# Metrics (same as existing evaluate_*.py files)
# ──────────────────────────────────────────────
def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())

def compute_bleu(reference, candidate):
    try:
        smoothing = SmoothingFunction().method1
        b1 = sentence_bleu([reference.split()], candidate.split(),
                           weights=(1.0, 0, 0, 0), smoothing_function=smoothing) * 100
        b2 = sentence_bleu([reference.split()], candidate.split(),
                           weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing) * 100
        return b1, b2
    except Exception:
        return 0.0, 0.0

def compute_rouge_l(reference, candidate):
    rouge = PyRouge(rouge_n=(1, 2), rouge_l=True, rouge_w=False,
                    rouge_w_weight=1.2, rouge_s=False, rouge_su=False, skip_gap=4)
    try:
        return rouge.evaluate([candidate], [[reference]])["rouge-l"]["f"] * 100
    except Exception:
        return 0.0

def compute_bertscore_batched(references, candidates, batch_size=32):
    all_f1 = []
    for i in range(0, len(references), batch_size):
        try:
            _, _, F1 = bert_score(candidates[i:i+batch_size], references[i:i+batch_size],
                                  lang="en", verbose=False)
            all_f1.extend([f * 100 for f in F1.tolist()])
        except Exception as e:
            print(f"BERTScore error batch {i}: {e}")
            all_f1.extend([0.0] * len(references[i:i+batch_size]))
    return all_f1


print("\nComputing metrics...")
bleu1_scores, bleu2_scores, rouge_scores = [], [], []

for _, row in tqdm(df.iterrows(), total=len(df), desc="BLEU + ROUGE-L"):
    ref  = clean_text(row["target"])
    cand = clean_text(row["generated_summary"])
    if not ref or not cand:
        bleu1_scores.append(0.0)
        bleu2_scores.append(0.0)
        rouge_scores.append(0.0)
    else:
        b1, b2 = compute_bleu(ref, cand)
        bleu1_scores.append(b1)
        bleu2_scores.append(b2)
        rouge_scores.append(compute_rouge_l(ref, cand))

refs  = [clean_text(t) for t in df["target"]]
cands = [clean_text(t) for t in df["generated_summary"]]
bert_f1 = compute_bertscore_batched(refs, cands)

df["bleu1"]   = bleu1_scores
df["bleu2"]   = bleu2_scores
df["rouge_l"] = rouge_scores
df["bert_f1"] = bert_f1
df.to_csv(OUTPUT_PATH, index=False)

# ──────────────────────────────────────────────
# Print Table 1 row
# ──────────────────────────────────────────────
print("\n" + "="*60)
print(f"TABLE 1 ROW  —  {MODEL_NAME}  (SOAP Summary)")
print("="*60)
print(f"ROUGE-L:    {np.mean(rouge_scores):.2f} ± {np.std(rouge_scores):.2f}")
print(f"BLEU-1:     {np.mean(bleu1_scores):.2f} ± {np.std(bleu1_scores):.2f}")
print(f"BLEU-2:     {np.mean(bleu2_scores):.2f} ± {np.std(bleu2_scores):.2f}")
print(f"BERTScore:  {np.mean(bert_f1):.2f} ± {np.std(bert_f1):.2f}")
print("="*60)
print(f"\nFull results saved to {OUTPUT_PATH}")
