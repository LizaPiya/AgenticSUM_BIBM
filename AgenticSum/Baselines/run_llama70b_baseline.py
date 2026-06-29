"""
Llama-3.1-70B-Instruct single-pass baseline for MIMIC and SOAP.

Uses EXACTLY the same:
  - prompts as DraftAgent (draft_agent.py for MIMIC, draft_agent_soap.py for SOAP)
  - metrics code from Agents/Evaluation.py
  - judge model:  meta-llama/Meta-Llama-3-8B-Instruct in torch.float16
  - judge prompt: Agents/llm_as_a_judge.py  (same as Table 2)

Step 1: Generate summaries with Llama-3.1-70B (4-bit NF4)
Step 2: ROUGE-L, BLEU-1, BLEU-2, BERTScore via Evaluation.py
Step 3: Free 70B → load Meta-Llama-3-8B-Instruct (float16) → LLM-as-judge
"""

import os
import sys
import gc
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── use existing code unchanged ────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../Agents"))
from Evaluation import evaluate_summaries
from llm_as_a_judge import llm_hallucination_evaluation   # same as Table 2

# ── config ─────────────────────────────────────────────────────────────────────
GEN_MODEL   = "meta-llama/Llama-3.1-70B-Instruct"
JUDGE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"       # exact model from paper
HF_TOKEN    = os.environ.get("HUGGINGFACE_HUB_TOKEN")
MAX_NEW_TOKENS = 350                                       # same as DraftAgent
OUTPUT_DIR  = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../outputs/llama70b_baseline")
)

# ── prompts copied verbatim from draft_agent.py and draft_agent_soap.py ───────
def mimic_prompt(text: str) -> str:
    return (
        "Based on the patient record below, write a concise clinical summary "
        "describing the patient's hospital stay. Include reason for admission, "
        "key findings, and treatments given. Use only information from the record. "
        "Do NOT add notes, commentary, or explanations about the summary itself.\n\n"
        f"{text}\n\n"
        "Summary:"
    )

def soap_prompt(text: str) -> str:
    return (
        "You are a helpful medical assistant. Analyze the following patient-doctor dialogue and "
        "create a concise medical summary in SOAP format (Subjective, Objective, Assessment, Plan). "
        "Focus on key symptoms, findings, diagnosis, and treatment plan:\n\n"
        f"{text}\n\nSOAP Summary:"
    )

DATASETS = {
    "mimic": {
        "path": os.path.join(os.path.dirname(__file__), "../Dataset/sample_data_100.csv"),
        "prompt_fn": mimic_prompt,
    },
    "soap": {
        "path": os.path.join(os.path.dirname(__file__), "../Dataset/df_soap_mimic.csv"),
        "prompt_fn": soap_prompt,
    },
}


# ── helpers ────────────────────────────────────────────────────────────────────
def load_4bit(model_name):
    """Load generation model in 4-bit NF4 to fit 70B on GPU."""
    print(f"\nLoading {model_name} in 4-bit NF4...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, use_fast=True)
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    )
    mdl.eval()
    print(f"  GPU mem used: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return mdl, tok


def load_judge(model_name):
    """Load judge in torch.float16 — same as original llm_as_a_judge_llama.py."""
    print(f"\nLoading judge {model_name} in float16...")
    tok = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN,
    ).eval()
    print(f"  GPU mem used: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return mdl, tok


def free_model(mdl):
    del mdl
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  GPU mem after free: {torch.cuda.memory_allocated()/1e9:.1f} GB")


def generate_one(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=3500
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    return decoded.split("Summary:")[-1].strip() if "Summary:" in decoded \
        else decoded[len(prompt):].strip()


def print_results(df, label):
    print(f"\n{'='*60}")
    print(f"RESULTS — {label}")
    print(f"{'='*60}")
    for name_, col in [
        ("ROUGE-L",          "rouge_l"),
        ("BLEU-1",           "bleu1"),
        ("BLEU-2",           "bleu2"),
        ("BERTScore F1",     "bert_f1"),
        ("Hallucination ↓",  "hallucination_score"),
        ("Factual Cons. ↑",  "factual_consistency"),
        ("Completeness ↑",   "completeness"),
        ("Coherence ↑",      "coherence"),
    ]:
        if col in df.columns:
            print(f"  {name_:<20} {df[col].mean():.2f} ± {df[col].std():.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Generate with Llama-3.1-70B
# ══════════════════════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_DIR, exist_ok=True)
assert torch.cuda.is_available(), "CUDA not available"
print("GPU:", torch.cuda.get_device_name(0))

model, tokenizer = load_4bit(GEN_MODEL)

for name, cfg in DATASETS.items():
    df = pd.read_csv(cfg["path"])
    print(f"\nGenerating {name.upper()} ({len(df)} samples)...")

    summaries = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            summary = generate_one(model, tokenizer, cfg["prompt_fn"](str(row["input"])))
        except Exception as e:
            print(f"  Error row {idx}: {e}")
            summary = "ERROR"
        summaries.append(summary)
        torch.cuda.empty_cache()
        gc.collect()

    df["fixed_summary"] = summaries
    df = evaluate_summaries(df, summary_column="fixed_summary", reference_column="target")

    interim = os.path.join(OUTPUT_DIR, f"llama70b_{name}_summaries.csv")
    df.to_csv(interim, index=False)
    print(f"  Saved → {interim}")

print("\nFreeing 70B model...")
free_model(model)
del tokenizer

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — LLM-as-judge with Meta-Llama-3-8B-Instruct in float16
# ══════════════════════════════════════════════════════════════════════════════
judge_model, judge_tok = load_judge(JUDGE_MODEL)

for name in DATASETS:
    interim  = os.path.join(OUTPUT_DIR, f"llama70b_{name}_summaries.csv")
    out_path = os.path.join(OUTPUT_DIR, f"llama70b_{name}_results.csv")

    df = llm_hallucination_evaluation(
        model=judge_model,
        tokenizer=judge_tok,
        csv_path=interim,
        output_path=out_path,
    )
    print_results(df, f"Llama-3.1-70B single-pass — {name.upper()}")

free_model(judge_model)
print("\nAll done. Results in:", OUTPUT_DIR)
