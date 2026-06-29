"""
run_llm_judge_ablation.py
--------------------------
Runs LLM-as-a-judge hallucination evaluation across all ablation conditions.

Evaluates:
  Pipeline ablation  (ablation_5docs.csv):
    C1 - Vanilla LLM
    C2 - +FOCUS
    C3 - +FOCUS +FixAgent
    C4 - Full AgenticSum

  AURA ablation  (aura_ablation_5docs.csv):
    A1 - Semantic Only
    A2 - AURA Only
    A3 - AURA + Semantic

Scores each condition on:
  - Hallucination (1=none, 5=major fabrications)
  - Factual consistency (1=inaccurate, 5=accurate)
  - Completeness (1=missing key info, 5=comprehensive)
  - Coherence (1=poor, 5=excellent)

No existing files are modified.
"""

import os
import gc
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_as_a_judge import llm_hallucination_evaluation


# ======================================================
# Reproducibility
# ======================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

assert torch.cuda.is_available(), "CUDA is not available"
print("=" * 80)
print("Using GPU:", torch.cuda.get_device_name(0))
print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
print("=" * 80)


# ======================================================
# HF token
# ======================================================
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
assert HF_TOKEN is not None, "HUGGINGFACE_HUB_TOKEN not set"


# ======================================================
# Model + Tokenizer
# ======================================================
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

print("Loading model and tokenizer...")
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
print("Model loaded\n")


# ======================================================
# Paths
# ======================================================
output_dir        = "/home/user/MLHC_AgenticSUM/outputs/agenticsum"
aura_mimic_csv    = f"{output_dir}/aura_ablation_100docs_MIMIC.csv"
aura_soap_csv     = f"{output_dir}/aura_ablation_100docs_SOAP.csv"
judge_output_path = f"{output_dir}/llm_judge_aura_100docs_scores.csv"

os.makedirs(output_dir, exist_ok=True)


# ======================================================
# Helper: reshape ablation CSV into the format expected
# by llm_hallucination_evaluation (needs: note_id, input,
# target, fixed_summary columns)
# ======================================================
def reshape_for_judge(df: pd.DataFrame, summary_col: str) -> pd.DataFrame:
    return pd.DataFrame({
        "note_id":       df["note_id"],
        "input":         df["input"],
        "target":        df["target"],
        "fixed_summary": df[summary_col],
    })


# ======================================================
# Define all conditions to evaluate
# Format: (label, source_csv, summary_column)
# ======================================================
all_conditions = []

for dataset_name, aura_csv in [("MIMIC", aura_mimic_csv), ("SOAP", aura_soap_csv)]:
    if os.path.exists(aura_csv):
        aura_df = pd.read_csv(aura_csv)
        print(f"AURA ablation loaded ({dataset_name}): {len(aura_df)} docs")
        all_conditions += [
            (f"A1 - Semantic Only [{dataset_name}]",   aura_df, "A1 - Semantic Only",   dataset_name),
            (f"A2 - AURA Only [{dataset_name}]",       aura_df, "A2 - AURA Only",       dataset_name),
            (f"A3 - AURA + Semantic [{dataset_name}]", aura_df, "A3 - AURA + Semantic", dataset_name),
        ]
    else:
        print(f"AURA ablation CSV not found for {dataset_name} — skipping")

print(f"\nTotal conditions to evaluate: {len(all_conditions)}\n")


# ======================================================
# Run LLM judge on each condition
# ======================================================
print("=" * 80)
print("LLM-AS-A-JUDGE HALLUCINATION EVALUATION")
print("=" * 80 + "\n")

summary_rows = []

for label, source_df, col, dataset_name in all_conditions:
    print(f"\n{'─'*60}")
    print(f" {label}")
    print(f"{'─'*60}")

    # Skip if all errors
    valid = source_df[~source_df[col].astype(str).str.startswith("ERROR", na=False)]
    if valid.empty:
        print(f"  All rows errored — skipping")
        continue

    # Reshape to expected format
    reshaped = reshape_for_judge(valid.reset_index(drop=True), col)

    # Write temp CSV for llm_hallucination_evaluation
    temp_csv    = f"{output_dir}/_tmp_judge_input.csv"
    temp_output = f"{output_dir}/_tmp_judge_output.csv"
    reshaped.to_csv(temp_csv, index=False)

    torch.cuda.empty_cache()
    gc.collect()

    # Run judge
    scored_df = llm_hallucination_evaluation(
        model=model,
        tokenizer=tokenizer,
        csv_path=temp_csv,
        output_path=temp_output,
    )

    # Save per-document scores for this condition
    condition_slug = label.replace(" ", "_").replace("-", "").replace("+", "plus")
    per_doc_path = f"{output_dir}/llm_judge_{condition_slug}.csv"
    scored_df["condition"] = label
    scored_df.to_csv(per_doc_path, index=False)
    print(f"  Per-doc scores saved: {per_doc_path}")

    summary_rows.append({
        "dataset":             dataset_name,
        "condition":           label,
        "n":                   len(scored_df),
        "hallucination_score": round(scored_df["hallucination_score"].mean(), 2),
        "factual_consistency": round(scored_df["factual_consistency"].mean(), 2),
        "completeness":        round(scored_df["completeness"].mean(), 2),
        "coherence":           round(scored_df["coherence"].mean(), 2),
    })

    torch.cuda.empty_cache()
    gc.collect()

# Cleanup temp files
for f in [f"{output_dir}/_tmp_judge_input.csv", f"{output_dir}/_tmp_judge_output.csv"]:
    if os.path.exists(f):
        os.remove(f)


# ======================================================
# Print + save summary
# ======================================================
print("\n" + "=" * 80)
print("LLM JUDGE SUMMARY TABLE")
print("Note: Hallucination 1=none 5=severe | others 1=worst 5=best")
print("=" * 80)

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

summary_df.to_csv(judge_output_path, index=False)
print(f"\nResults saved to: {judge_output_path}")
print("=" * 80)
