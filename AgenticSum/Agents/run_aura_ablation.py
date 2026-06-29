"""
run_aura_ablation.py
--------------------
AURA-specific ablation study for AgenticSum (5 documents).
Directly addresses Reviewer nBCZ's comment:
  "it remains unclear how much of the improvement can be attributed
   specifically to the AURA grounding signal versus the broader pipeline."

Conditions (full AgenticSum pipeline, only detection signal varies):
  A1 - Semantic Only  : hallucination detection uses semantic entailment only
                        (AURA threshold = 0.0, so AURA never flags)
  A2 - AURA Only      : hallucination detection uses AURA only
                        (mock semantic judge always returns SUPPORTED)
  A3 - AURA + Semantic: current system (both signals, OR logic)

No existing files are modified.
"""

import os
import gc
import random
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from focus_agent import FocusAgent
from draft_agent import DraftAgent
from HallucinationDetectorAgent import HallucinationDetectorAgent
from FixAgent import FixAgent
from ClinicalSupervisorAgent import ClinicalSupervisorAgent
from semantic_entailment_judge import SemanticEntailmentJudge
from Evaluation import evaluate_summaries


# ======================================================
# Mock semantic judge for AURA-only condition
# Always returns SUPPORTED — disables semantic entailment
# ======================================================
class AlwaysSupportedJudge:
    """Dummy semantic judge that never flags hallucinations.
    Used to isolate the AURA signal in the AURA-only ablation condition."""

    def judge(self, document: str, span: str) -> Dict[str, Any]:
        return {
            "is_supported": True,
            "raw_response": "SUPPORTED",
            "explanation": "",
            "evidence": None,
            "problematic_spans": None,
        }

    def reset(self):
        pass


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
model.config.output_attentions = True
model.eval()
print("Model loaded\n")


# ======================================================
# Shared agents (focus, draft, fix — same across all conditions)
# ======================================================
focus_agent = FocusAgent(model=model, tokenizer=tokenizer, retention_ratio=0.35, batch_size=8)
draft_agent = DraftAgent(model=model, tokenizer=tokenizer, max_new_tokens=256)
fix_agent   = FixAgent(model=model, tokenizer=tokenizer, max_new_tokens=150)

# Real semantic judge (used in A1 and A3)
real_semantic_judge = SemanticEntailmentJudge(model=model, tokenizer=tokenizer)

# Mock semantic judge (used in A2)
mock_semantic_judge = AlwaysSupportedJudge()

print("Shared agents initialized\n")


# ======================================================
# Build three detectors — one per condition
# ======================================================

# A1: Semantic entailment only (AURA threshold = 0.0 → never flags on AURA)
detector_semantic_only = HallucinationDetectorAgent(
    model=model,
    tokenizer=tokenizer,
    semantic_judge=real_semantic_judge,
    aura_threshold=0.0,
)

# A2: AURA only (mock semantic judge → never flags on semantics)
detector_aura_only = HallucinationDetectorAgent(
    model=model,
    tokenizer=tokenizer,
    semantic_judge=mock_semantic_judge,
    aura_threshold=0.42,
)

# A3: Both signals — current system
detector_both = HallucinationDetectorAgent(
    model=model,
    tokenizer=tokenizer,
    semantic_judge=real_semantic_judge,
    aura_threshold=0.42,
)

# Build one supervisor per detector
supervisor_semantic_only = ClinicalSupervisorAgent(
    focus_agent=focus_agent,
    draft_agent=draft_agent,
    hallucination_detector_agent=detector_semantic_only,
    fix_agent=fix_agent,
    max_iterations=3,
)

supervisor_aura_only = ClinicalSupervisorAgent(
    focus_agent=focus_agent,
    draft_agent=draft_agent,
    hallucination_detector_agent=detector_aura_only,
    fix_agent=fix_agent,
    max_iterations=3,
)

supervisor_both = ClinicalSupervisorAgent(
    focus_agent=focus_agent,
    draft_agent=draft_agent,
    hallucination_detector_agent=detector_both,
    fix_agent=fix_agent,
    max_iterations=3,
)

print("All three supervisor configurations initialized\n")


# ======================================================
# Load documents — MIMIC and SOAP, 50 each
# ======================================================
DATASETS = {
    "MIMIC": "../Dataset/sample_data_100.csv",
    "SOAP":  "../Dataset/df_soap_mimic.csv",
}
N_DOCS = 50


# ======================================================
# Output setup
# ======================================================
output_dir = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum"
os.makedirs(output_dir, exist_ok=True)

conditions = [
    ("A1 - Semantic Only",   supervisor_semantic_only),
    ("A2 - AURA Only",       supervisor_aura_only),
    ("A3 - AURA + Semantic", supervisor_both),
]

all_eval_rows = []

# ======================================================
# Run AURA ablation on each dataset
# ======================================================
for dataset_name, data_path in DATASETS.items():
    print("\n" + "=" * 80)
    print(f"AURA ABLATION — {dataset_name} — {N_DOCS} documents")
    print("=" * 80 + "\n")

    df = (
        pd.read_csv(data_path)[lambda x: x["input"].str.len() <= 6000]
        .head(N_DOCS)
        .reset_index(drop=True)
    )

    print(f"Loaded {len(df)} documents from {dataset_name}")
    for _, row in df.iterrows():
        print(f"  {row['note_id']}  ({len(row['input'])} chars)")
    print()

    results = []

    with torch.no_grad():
        for idx, row in df.iterrows():
            doc     = row["input"]
            target  = row["target"]
            note_id = row["note_id"]

            print(f"\n[{idx+1}/{len(df)}] {note_id} ({len(doc)} chars)")
            print("-" * 60)

            result = {"note_id": note_id, "input": doc, "target": target, "dataset": dataset_name}

            for label, supervisor in conditions:
                torch.cuda.empty_cache()
                gc.collect()

                try:
                    print(f"  {label}...", end=" ", flush=True)
                    out = supervisor.run(doc)
                    result[label] = out["fixed_summary"]
                    print(f"done ({out['num_iterations']} iter, {out['termination_reason']})")
                except Exception as e:
                    print(f"ERROR: {e}")
                    result[label] = f"ERROR: {e}"

            results.append(result)
            torch.cuda.empty_cache()
            gc.collect()

    # Save raw results per dataset
    results_df = pd.DataFrame(results)
    raw_path = f"{output_dir}/aura_ablation_50docs_{dataset_name}.csv"
    results_df.to_csv(raw_path, index=False)
    print(f"\nRaw results saved to: {raw_path}\n")

    # Evaluate each condition
    print("=" * 80)
    print(f"EVALUATION RESULTS — {dataset_name}")
    print("=" * 80)

    for label, _ in conditions:
        col   = label
        valid = results_df[~results_df[col].str.startswith("ERROR", na=False)].copy()

        if valid.empty:
            print(f"\n{label}: all rows errored, skipping\n")
            continue

        print(f"\n{'─'*60}")
        print(f" {label}  (n={len(valid)})")
        print(f"{'─'*60}")

        scored = evaluate_summaries(valid, summary_column=col, reference_column="target")

        all_eval_rows.append({
            "dataset":   dataset_name,
            "condition": label,
            "n":         len(valid),
            "ROUGE-L":   round(scored["rouge_l"].mean(), 2),
            "BLEU-1":    round(scored["bleu1"].mean(), 2),
            "BLEU-2":    round(scored["bleu2"].mean(), 2),
            "BERTScore": round(scored["bert_f1"].mean(), 2),
        })


# ======================================================
# Combined summary table
# ======================================================
print("\n" + "=" * 80)
print("AURA ABLATION SUMMARY TABLE — ALL DATASETS")
print("=" * 80)

summary_df = pd.DataFrame(all_eval_rows)
print(summary_df.to_string(index=False))

scores_path = f"{output_dir}/aura_ablation_50docs_scores.csv"
summary_df.to_csv(scores_path, index=False)
print(f"\nSummary scores saved to: {scores_path}")
print("=" * 80)
