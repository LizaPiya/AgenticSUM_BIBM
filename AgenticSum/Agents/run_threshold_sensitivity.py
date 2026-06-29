"""
run_threshold_sensitivity.py
-----------------------------
Sensitivity analysis for the AURA threshold (tau) parameter.

Addresses reviewer concern:
  "Threshold selection for AURA appears heuristic and lacks sensitivity analysis."

Runs the full AgenticSum pipeline (A3: AURA + Semantic) on a small sample
of documents using tau in {0.2, 0.3, 0.42, 0.5, 0.6, 0.7}.

Reports ROUGE-L, BLEU-1/2, and BERTScore for each threshold value.
The default tau=0.42 is marked in the output.

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
# Thresholds to sweep
# ======================================================
THRESHOLDS = [0.2, 0.3, 0.42, 0.5, 0.6, 0.7]
DEFAULT_THRESHOLD = 0.42

# ======================================================
# Shared agents (fixed across all threshold conditions)
# ======================================================
focus_agent = FocusAgent(model=model, tokenizer=tokenizer, retention_ratio=0.35, batch_size=8)
draft_agent = DraftAgent(model=model, tokenizer=tokenizer, max_new_tokens=256)
fix_agent   = FixAgent(model=model, tokenizer=tokenizer, max_new_tokens=150)
semantic_judge = SemanticEntailmentJudge(model=model, tokenizer=tokenizer)

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
    print(f"THRESHOLD SENSITIVITY — {dataset_name} — {N_DOCS} documents")
    print("=" * 80 + "\n")

    df = (
        pd.read_csv(data_path)[lambda x: x["input"].str.len() <= 6000]
        .head(N_DOCS)
        .reset_index(drop=True)
    )
    print(f"Loaded {len(df)} documents from {dataset_name}\n")

    # Collect raw summaries per threshold
    # results_by_tau[tau] = list of dicts with note_id, input, target, summary
    results_by_tau = {tau: [] for tau in THRESHOLDS}

    with torch.no_grad():
        for idx, row in df.iterrows():
            doc     = row["input"]
            target  = row["target"]
            note_id = row["note_id"]

            print(f"\n[{idx+1}/{len(df)}] {note_id} ({len(doc)} chars)")
            print("-" * 60)

            for tau in THRESHOLDS:
                torch.cuda.empty_cache()
                gc.collect()

                # Build detector and supervisor for this tau
                detector = HallucinationDetectorAgent(
                    model=model,
                    tokenizer=tokenizer,
                    semantic_judge=semantic_judge,
                    aura_threshold=tau,
                )
                supervisor = ClinicalSupervisorAgent(
                    focus_agent=focus_agent,
                    draft_agent=draft_agent,
                    hallucination_detector_agent=detector,
                    fix_agent=fix_agent,
                    max_iterations=3,
                )

                tag = f"tau={tau}" + (" [DEFAULT]" if tau == DEFAULT_THRESHOLD else "")
                try:
                    print(f"  {tag}...", end=" ", flush=True)
                    out = supervisor.run(doc)
                    summary = out["fixed_summary"]
                    print(f"done ({out['num_iterations']} iter, {out['termination_reason']})")
                except Exception as e:
                    print(f"ERROR: {e}")
                    summary = f"ERROR: {e}"

                results_by_tau[tau].append({
                    "note_id":  note_id,
                    "input":    doc,
                    "target":   target,
                    "summary":  summary,
                })

                # Explicitly free detector/supervisor to avoid memory accumulation
                del detector, supervisor
                torch.cuda.empty_cache()
                gc.collect()

    # Save raw results for this dataset
    for tau in THRESHOLDS:
        tau_df = pd.DataFrame(results_by_tau[tau])
        slug = str(tau).replace(".", "p")
        raw_path = f"{output_dir}/threshold_sensitivity_{dataset_name}_tau{slug}.csv"
        tau_df.to_csv(raw_path, index=False)

    # Evaluate each threshold
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS — {dataset_name}")
    print("=" * 80)

    for tau in THRESHOLDS:
        tau_df = pd.DataFrame(results_by_tau[tau])
        valid = tau_df[~tau_df["summary"].str.startswith("ERROR", na=False)].copy()

        if valid.empty:
            print(f"\ntau={tau}: all rows errored, skipping\n")
            continue

        tag = f"tau={tau}" + (" ← DEFAULT" if tau == DEFAULT_THRESHOLD else "")
        print(f"\n{'─'*60}")
        print(f"  {tag}  (n={len(valid)})")
        print(f"{'─'*60}")

        scored = evaluate_summaries(valid, summary_column="summary", reference_column="target")

        row = {
            "dataset":    dataset_name,
            "tau":        tau,
            "is_default": tau == DEFAULT_THRESHOLD,
            "n":          len(valid),
            "ROUGE-L":    round(scored["rouge_l"].mean(), 4),
            "BLEU-1":     round(scored["bleu1"].mean(), 4),
            "BLEU-2":     round(scored["bleu2"].mean(), 4),
            "BERTScore":  round(scored["bert_f1"].mean(), 4),
        }
        all_eval_rows.append(row)
        print(f"  ROUGE-L={row['ROUGE-L']}  BLEU-1={row['BLEU-1']}  "
              f"BLEU-2={row['BLEU-2']}  BERTScore={row['BERTScore']}")


# ======================================================
# Summary table
# ======================================================
print("\n" + "=" * 80)
print("THRESHOLD SENSITIVITY SUMMARY TABLE")
print(f"Default tau = {DEFAULT_THRESHOLD}  |  Datasets: {list(DATASETS.keys())}")
print("=" * 80)

summary_df = pd.DataFrame(all_eval_rows)
print(summary_df.to_string(index=False))

scores_path = f"{output_dir}/threshold_sensitivity_scores.csv"
summary_df.to_csv(scores_path, index=False)
print(f"\nSummary scores saved to: {scores_path}")
print("=" * 80)
