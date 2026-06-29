"""
run_ablation.py
---------------
Additive ablation study for AgenticSum (5 documents).

Conditions:
  C1 - Vanilla LLM      : DraftAgent on full document (no compression)
  C2 - +FOCUS           : FocusAgent compression  -> DraftAgent
  C3 - +FOCUS +FixAgent : C2 + one-pass HallucinationDetector + FixAgent (no supervisor loop)
  C4 - Full AgenticSum  : ClinicalSupervisorAgent (C2 + iterative detect + fix)

No existing files are modified. All agents are reused as-is.
"""

import os
import gc
import random

import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM

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
# Hugging Face token
# ======================================================
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
assert HF_TOKEN is not None, "HUGGINGFACE_HUB_TOKEN not set"


# ======================================================
# Model + Tokenizer
# ======================================================
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    use_fast=True,
)
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
# Initialize agents (identical to run_agenticsum.py)
# ======================================================
focus_agent = FocusAgent(
    model=model,
    tokenizer=tokenizer,
    retention_ratio=0.35,
    batch_size=8,
)

semantic_judge = SemanticEntailmentJudge(
    model=model,
    tokenizer=tokenizer,
)

draft_agent = DraftAgent(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
)

hallucination_detector = HallucinationDetectorAgent(
    model=model,
    tokenizer=tokenizer,
    semantic_judge=semantic_judge,
    aura_threshold=0.42,
)

fix_agent = FixAgent(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
)

supervisor = ClinicalSupervisorAgent(
    focus_agent=focus_agent,
    draft_agent=draft_agent,
    hallucination_detector_agent=hallucination_detector,
    fix_agent=fix_agent,
    max_iterations=3,
)

print("All agents initialized\n")


# ======================================================
# Load dataset — pick 5 short-to-mid-length documents
# to avoid OOM across all 4 conditions (especially C1
# which has no input compression).
# ======================================================
data_path = "../Dataset/sample_data_100.csv"
df_all = pd.read_csv(data_path)

# Use same length guard as original run, then take first 5
df = (
    df_all[df_all["input"].str.len() <= 6000]
    .head(20)
    .reset_index(drop=True)
)

print(f"Running ablation on {len(df)} documents")
for _, row in df.iterrows():
    print(f"  {row['note_id']}  ({len(row['input'])} chars)")
print()


# ======================================================
# Output setup
# ======================================================
output_dir = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum"
os.makedirs(output_dir, exist_ok=True)

ablation_output_path = f"{output_dir}/ablation_20docs.csv"


# ======================================================
# Run ablation
# ======================================================
print("=" * 80)
print("ABLATION STUDY — 4 conditions x 5 documents")
print("=" * 80 + "\n")

results = []

with torch.no_grad():
    for idx, row in df.iterrows():
        doc    = row["input"]
        target = row["target"]
        note_id = row["note_id"]

        print(f"\n[{idx+1}/5] {note_id} ({len(doc)} chars)")
        print("-" * 60)

        torch.cuda.empty_cache()
        gc.collect()

        # --------------------------------------------------
        # C1: Vanilla LLM
        #     DraftAgent receives all sentences of the full
        #     document — no compression, no detection/fix.
        # --------------------------------------------------
        try:
            print("  C1 Vanilla LLM...", end=" ")
            c1 = draft_agent.generate(sent_tokenize(doc))
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")
            c1 = f"ERROR: {e}"

        torch.cuda.empty_cache()
        gc.collect()

        # --------------------------------------------------
        # C2: +FOCUS
        #     FocusAgent compresses, DraftAgent generates.
        #     No hallucination detection or correction.
        # --------------------------------------------------
        try:
            print("  C2 +FOCUS...", end=" ")
            focus_out  = focus_agent.compress(doc)
            c2 = draft_agent.generate(focus_out["sentences"])
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")
            c2 = f"ERROR: {e}"

        torch.cuda.empty_cache()
        gc.collect()

        # --------------------------------------------------
        # C3: +FOCUS +FixAgent (single pass)
        #     Same as C2, then one round of hallucination
        #     detection and targeted fix — no iterative loop.
        # --------------------------------------------------
        try:
            print("  C3 +FOCUS +FixAgent (1-pass)...", end=" ")

            # Reset detector state before use
            hallucination_detector.reset()
            semantic_judge.reset()

            detection = hallucination_detector.analyze(
                source_document=doc,
                draft_summary=c2,
            )
            c3 = fix_agent.fix(
                source_document=doc,
                spans=detection["spans"],
                hallucination_mask=detection["hallucination_mask"],
            )
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")
            c3 = f"ERROR: {e}"

        torch.cuda.empty_cache()
        gc.collect()

        # --------------------------------------------------
        # C4: Full AgenticSum
        #     ClinicalSupervisorAgent handles everything:
        #     FOCUS + draft + iterative detect + fix.
        #     supervisor.run() calls reset() internally.
        # --------------------------------------------------
        try:
            print("  C4 Full AgenticSum...", end=" ")
            sup_out = supervisor.run(doc)
            c4 = sup_out["fixed_summary"]
            print(f"done ({sup_out['num_iterations']} iterations, reason: {sup_out['termination_reason']})")
        except Exception as e:
            print(f"ERROR: {e}")
            c4 = f"ERROR: {e}"

        torch.cuda.empty_cache()
        gc.collect()

        results.append({
            "note_id":           note_id,
            "input":             doc,
            "target":            target,
            "vanilla_llm":       c1,
            "focus_draft":       c2,
            "focus_fix_single":  c3,
            "agenticsum":        c4,
        })

# ======================================================
# Save raw results
# ======================================================
results_df = pd.DataFrame(results)
results_df.to_csv(ablation_output_path, index=False)
print(f"\nRaw results saved to: {ablation_output_path}\n")


# ======================================================
# Evaluate each condition against the reference target
# ======================================================
print("=" * 80)
print("EVALUATION RESULTS")
print("=" * 80)

conditions = [
    ("C1 - Vanilla LLM",          "vanilla_llm"),
    ("C2 - +FOCUS",               "focus_draft"),
    ("C3 - +FOCUS +FixAgent",     "focus_fix_single"),
    ("C4 - Full AgenticSum",      "agenticsum"),
]

eval_rows = []

for label, col in conditions:
    # Skip rows with errors in this condition
    valid = results_df[~results_df[col].str.startswith("ERROR", na=False)].copy()

    if valid.empty:
        print(f"\n{label}: all rows errored, skipping\n")
        continue

    print(f"\n{'─'*60}")
    print(f" {label}  (n={len(valid)})")
    print(f"{'─'*60}")

    scored = evaluate_summaries(
        valid,
        summary_column=col,
        reference_column="target",
    )

    eval_rows.append({
        "condition": label,
        "n":         len(valid),
        "ROUGE-L":   round(scored["rouge_l"].mean(), 2),
        "BLEU-1":    round(scored["bleu1"].mean(), 2),
        "BLEU-2":    round(scored["bleu2"].mean(), 2),
        "BERTScore": round(scored["bert_f1"].mean(), 2),
    })

# ======================================================
# Print summary table
# ======================================================
print("\n" + "=" * 80)
print("ABLATION SUMMARY TABLE")
print("=" * 80)

summary_df = pd.DataFrame(eval_rows)
print(summary_df.to_string(index=False))

summary_df.to_csv(f"{output_dir}/ablation_20docs_scores.csv", index=False)
print(f"\nSummary scores saved to: {output_dir}/ablation_20docs_scores.csv")
print("=" * 80)
