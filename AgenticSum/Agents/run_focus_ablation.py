"""
run_focus_ablation.py
---------------------
FOCUS-specific ablation study for AgenticSum.
Directly addresses Reviewer dSaW's comment:
  "It is not clear if FOCUS actually contributes positively to the
   summarization process."

Conditions (full AgenticSum pipeline, only FOCUS varies):
  F1 - No FOCUS   : full document fed directly to DraftAgent (no compression)
  F2 - With FOCUS : compressed document (r=0.35) fed to DraftAgent

All other components (HallucinationDetector, FixAgent, Supervisor) identical.
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
# Passthrough agent — replaces FOCUS, returns full doc
# ======================================================
class PassthroughFocusAgent:
    """Returns the full document unchanged — disables FOCUS compression."""

    def compress(self, document: str) -> dict:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(document.strip())
        return {
            "sentences": sentences,
            "sentence_indices": list(range(len(sentences))),
            "sentence_scores": [1.0] * len(sentences),
            "fallback_used": True,
        }


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
# Shared agents
# ======================================================
focus_agent_real      = FocusAgent(model=model, tokenizer=tokenizer, retention_ratio=0.35, batch_size=8)
focus_agent_passthru  = PassthroughFocusAgent()
draft_agent           = DraftAgent(model=model, tokenizer=tokenizer, max_new_tokens=256)
fix_agent             = FixAgent(model=model, tokenizer=tokenizer, max_new_tokens=150)
semantic_judge        = SemanticEntailmentJudge(model=model, tokenizer=tokenizer)

detector = HallucinationDetectorAgent(
    model=model,
    tokenizer=tokenizer,
    semantic_judge=semantic_judge,
    aura_threshold=0.42,
)

# F1: No FOCUS — passthrough
supervisor_no_focus = ClinicalSupervisorAgent(
    focus_agent=focus_agent_passthru,
    draft_agent=draft_agent,
    hallucination_detector_agent=detector,
    fix_agent=fix_agent,
    max_iterations=3,
)

# F2: With FOCUS (r=0.35) — full system
supervisor_with_focus = ClinicalSupervisorAgent(
    focus_agent=focus_agent_real,
    draft_agent=draft_agent,
    hallucination_detector_agent=detector,
    fix_agent=fix_agent,
    max_iterations=3,
)

print("Both supervisor configurations initialized\n")


# ======================================================
# Load documents — MIMIC only, 2 docs
# ======================================================
N_DOCS = 2
mimic_df = pd.read_csv("../Dataset/sample_data_100.csv")
mimic_df = mimic_df.head(N_DOCS).reset_index(drop=True)

print(f"FOCUS ABLATION — MIMIC — {N_DOCS} documents")
print("=" * 80)

output_dir = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum"
os.makedirs(output_dir, exist_ok=True)

conditions = [
    ("F1 - No FOCUS",   supervisor_no_focus),
    ("F2 - With FOCUS", supervisor_with_focus),
]

all_results = []
all_eval_rows = []

for cond_name, supervisor in conditions:
    print(f"\n{'='*60}")
    print(f" {cond_name}")
    print(f"{'='*60}")

    detector.reset()
    semantic_judge.reset()

    rows = []
    for i, row in mimic_df.iterrows():
        note_id = row.get("note_id", f"doc_{i}")
        source  = str(row["input"])
        target  = str(row["target"])

        print(f"  [{i+1}/{N_DOCS}] {note_id} ({len(source)} chars)")
        print(f"  {cond_name}...", end=" ", flush=True)

        detector.reset()
        semantic_judge.reset()

        try:
            result = supervisor.run(document=source)
            summary = result.get("fixed_summary") or result.get("draft_summary", "")
            n_iter  = result.get("iterations", 0)
            reason  = result.get("termination_reason", "unknown")
            print(f"done ({n_iter} iter, {reason})")
        except Exception as e:
            print(f"ERROR: {e}")
            summary = ""
            n_iter  = -1
            reason  = "error"

        rows.append({
            "note_id":    note_id,
            "condition":  cond_name,
            "input":      source,
            "target":     target,
            "summary":    summary,
            "iterations": n_iter,
            "termination_reason": reason,
        })

    cond_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, f"focus_ablation_MIMIC_{cond_name.replace(' ', '_').replace('/', '')}.csv")
    cond_df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    # Evaluate
    valid = cond_df[cond_df["summary"].str.strip() != ""]
    if len(valid) > 0:
        eval_df = evaluate_summaries(
            df=valid.rename(columns={"summary": "fixed_summary"}),
            summary_column="fixed_summary",
            reference_column="target",
        )
        scores = {
            "condition": cond_name,
            "n": len(valid),
            "ROUGE-L":   round(eval_df["ROUGE-L"].mean(), 4)   if "ROUGE-L"   in eval_df.columns else eval_df.filter(like="rouge").mean().mean(),
            "BLEU-1":    round(eval_df["BLEU-1"].mean(), 4)    if "BLEU-1"    in eval_df.columns else eval_df["bleu1"].mean(),
            "BLEU-2":    round(eval_df["BLEU-2"].mean(), 4)    if "BLEU-2"    in eval_df.columns else eval_df["bleu2"].mean(),
            "BERTScore": round(eval_df["BERTScore"].mean(), 4) if "BERTScore" in eval_df.columns else eval_df["bert_f1"].mean(),
        }
        all_eval_rows.append(scores)
        print(f"  ROUGE-L={scores['ROUGE-L']:.4f} | BLEU-1={scores['BLEU-1']:.4f} | BERTScore={scores['BERTScore']:.4f}")

    all_results.extend(rows)
    gc.collect()
    torch.cuda.empty_cache()

# ======================================================
# Save summary scores
# ======================================================
scores_df = pd.DataFrame(all_eval_rows)[["condition", "n", "ROUGE-L", "BLEU-1", "BLEU-2", "BERTScore"]]
scores_path = os.path.join(output_dir, "focus_ablation_scores.csv")
scores_df.to_csv(scores_path, index=False)

print("\n" + "=" * 80)
print("FOCUS ABLATION COMPLETE")
print("=" * 80)
print(scores_df.to_string(index=False))
print(f"\nSummary scores saved to: {scores_path}")
