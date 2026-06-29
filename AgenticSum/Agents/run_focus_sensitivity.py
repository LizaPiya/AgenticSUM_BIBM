"""
run_focus_sensitivity.py
-------------------------
Sensitivity analysis for the FOCUS retention ratio (r) parameter.

Addresses reviewer concern:
  "Hyperparameter sensitivity — retention ratio r=0.35 was selected on a validation subset."

Sweeps r in {0.2, 0.3, 0.35, 0.5, 0.6} using the full AgenticSum pipeline
(A3: AURA + Semantic, tau=0.42) on a small sample of documents.

Reports ROUGE-L, BLEU-1/2, and BERTScore for each retention ratio.
The default r=0.35 is marked in the output.

No existing files are modified.
"""

import os
import gc
import random

import numpy as np
import pandas as pd
import torch

from focus_agent import FocusAgent
from draft_agent import DraftAgent
from HallucinationDetectorAgent import HallucinationDetectorAgent
from FixAgent import FixAgent
from ClinicalSupervisorAgent import ClinicalSupervisorAgent
from semantic_entailment_judge import SemanticEntailmentJudge
from Evaluation import evaluate_summaries


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
from transformers import AutoTokenizer, AutoModelForCausalLM

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
model.config.output_attentions = True
model.eval()
print("Model loaded\n")


# ======================================================
# Retention ratios to sweep
# ======================================================
RATIOS = [0.2, 0.3, 0.35, 0.5, 0.6]
DEFAULT_RATIO = 0.35

# Fixed AURA threshold and semantic judge (default settings)
AURA_THRESHOLD = 0.42

# ======================================================
# Shared agents (fixed across all ratio conditions)
# ======================================================
draft_agent    = DraftAgent(model=model, tokenizer=tokenizer, max_new_tokens=256)
fix_agent      = FixAgent(model=model, tokenizer=tokenizer, max_new_tokens=150)
semantic_judge = SemanticEntailmentJudge(model=model, tokenizer=tokenizer)

detector = HallucinationDetectorAgent(
    model=model,
    tokenizer=tokenizer,
    semantic_judge=semantic_judge,
    aura_threshold=AURA_THRESHOLD,
)

print("Shared agents initialized\n")


# ======================================================
# Datasets + sample size
# ======================================================
DATASETS = {
    "MIMIC": "../Dataset/sample_data_100.csv",
    "SOAP":  "../Dataset/df_soap_mimic.csv",
}
N_DOCS = 15  # Small sample — sufficient for sensitivity analysis

output_dir = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum"
os.makedirs(output_dir, exist_ok=True)


# ======================================================
# Run sensitivity analysis
# ======================================================
all_eval_rows = []

for dataset_name, data_path in DATASETS.items():
    print("\n" + "=" * 80)
    print(f"FOCUS RETENTION RATIO SENSITIVITY — {dataset_name} — {N_DOCS} documents")
    print("=" * 80 + "\n")

    df = (
        pd.read_csv(data_path)[lambda x: x["input"].str.len() <= 6000]
        .head(N_DOCS)
        .reset_index(drop=True)
    )
    print(f"Loaded {len(df)} documents from {dataset_name}\n")

    results_by_ratio = {r: [] for r in RATIOS}

    with torch.no_grad():
        for idx, row in df.iterrows():
            doc     = row["input"]
            target  = row["target"]
            note_id = row["note_id"]

            print(f"\n[{idx+1}/{len(df)}] {note_id} ({len(doc)} chars)")
            print("-" * 60)

            for r in RATIOS:
                torch.cuda.empty_cache()
                gc.collect()

                # Build FocusAgent and Supervisor for this r
                focus_agent = FocusAgent(
                    model=model,
                    tokenizer=tokenizer,
                    retention_ratio=r,
                    batch_size=8,
                )
                supervisor = ClinicalSupervisorAgent(
                    focus_agent=focus_agent,
                    draft_agent=draft_agent,
                    hallucination_detector_agent=detector,
                    fix_agent=fix_agent,
                    max_iterations=3,
                )

                tag = f"r={r}" + (" [DEFAULT]" if r == DEFAULT_RATIO else "")
                try:
                    print(f"  {tag}...", end=" ", flush=True)
                    out = supervisor.run(doc)
                    summary = out["fixed_summary"]
                    print(f"done ({out['num_iterations']} iter, {out['termination_reason']})")
                except Exception as e:
                    print(f"ERROR: {e}")
                    summary = f"ERROR: {e}"

                results_by_ratio[r].append({
                    "note_id":  note_id,
                    "input":    doc,
                    "target":   target,
                    "summary":  summary,
                })

                del focus_agent, supervisor
                torch.cuda.empty_cache()
                gc.collect()

    # Save raw results for this dataset
    for r in RATIOS:
        r_df = pd.DataFrame(results_by_ratio[r])
        slug = str(r).replace(".", "p")
        raw_path = f"{output_dir}/focus_sensitivity_{dataset_name}_r{slug}.csv"
        r_df.to_csv(raw_path, index=False)

    # Evaluate each ratio
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS — {dataset_name}")
    print("=" * 80)

    for r in RATIOS:
        r_df = pd.DataFrame(results_by_ratio[r])
        valid = r_df[~r_df["summary"].str.startswith("ERROR", na=False)].copy()

        if valid.empty:
            print(f"\nr={r}: all rows errored, skipping\n")
            continue

        tag = f"r={r}" + (" ← DEFAULT" if r == DEFAULT_RATIO else "")
        print(f"\n{'─'*60}")
        print(f"  {tag}  (n={len(valid)})")
        print(f"{'─'*60}")

        scored = evaluate_summaries(valid, summary_column="summary", reference_column="target")

        row_out = {
            "dataset":    dataset_name,
            "r":          r,
            "is_default": r == DEFAULT_RATIO,
            "n":          len(valid),
            "ROUGE-L":    round(scored["rouge_l"].mean(), 4),
            "BLEU-1":     round(scored["bleu1"].mean(), 4),
            "BLEU-2":     round(scored["bleu2"].mean(), 4),
            "BERTScore":  round(scored["bert_f1"].mean(), 4),
        }
        all_eval_rows.append(row_out)
        print(f"  ROUGE-L={row_out['ROUGE-L']}  BLEU-1={row_out['BLEU-1']}  "
              f"BLEU-2={row_out['BLEU-2']}  BERTScore={row_out['BERTScore']}")


# ======================================================
# Summary table
# ======================================================
print("\n" + "=" * 80)
print("FOCUS RETENTION RATIO SENSITIVITY SUMMARY TABLE")
print(f"Default r = {DEFAULT_RATIO}  |  Datasets: {list(DATASETS.keys())}")
print("=" * 80)

summary_df = pd.DataFrame(all_eval_rows)
print(summary_df.to_string(index=False))

scores_path = f"{output_dir}/focus_sensitivity_scores.csv"
summary_df.to_csv(scores_path, index=False)
print(f"\nSummary scores saved to: {scores_path}")
print("=" * 80)
