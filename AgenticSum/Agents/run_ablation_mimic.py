"""
run_ablation_mimic.py
---------------------
Additive ablation study for AgenticSum — MIMIC-IV (100 documents).

Conditions:
  C1 - Vanilla LLM         : DraftAgent on full document (no compression, no detection)
  C2 - +FOCUS              : FocusAgent compression -> DraftAgent (no detection/fix)
  C3 - +FOCUS +Detect +Fix : C2 + one-pass HallucinationDetector (AURA+Semantic) + FixAgent
                             (single iteration, no ClinicalSupervisorAgent loop)
  C4 - Full AgenticSum     : C2 + ClinicalSupervisorAgent (iterative detect+fix, up to 3 rounds)

Key settings (matching main pipeline):
  retention_ratio = 0.3
  aura_threshold  = 0.42
  max_length      = 4096 (handled inside FocusAgent / HallucinationDetectorAgent)
  No length filter — FOCUS handles long docs via truncation
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


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

assert torch.cuda.is_available(), "CUDA is not available"
print("=" * 80)
print("Using GPU:", torch.cuda.get_device_name(0))
print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
print("=" * 80)

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
assert HF_TOKEN is not None, "HUGGINGFACE_HUB_TOKEN not set"

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

# ── Agents ────────────────────────────────────────────────────────────────────
focus_agent = FocusAgent(model=model, tokenizer=tokenizer, retention_ratio=0.3, batch_size=8)
semantic_judge = SemanticEntailmentJudge(model=model, tokenizer=tokenizer)
draft_agent = DraftAgent(model=model, tokenizer=tokenizer, max_new_tokens=256)
hallucination_detector = HallucinationDetectorAgent(
    model=model, tokenizer=tokenizer,
    semantic_judge=semantic_judge,
    aura_threshold=0.42,
)
fix_agent = FixAgent(model=model, tokenizer=tokenizer, max_new_tokens=150)
supervisor = ClinicalSupervisorAgent(
    focus_agent=focus_agent,
    draft_agent=draft_agent,
    hallucination_detector_agent=hallucination_detector,
    fix_agent=fix_agent,
    max_iterations=3,
)
print("All agents initialized\n")

# ── Data ──────────────────────────────────────────────────────────────────────
data_path = "../Dataset/sample_data_100.csv"
df = pd.read_csv(data_path).head(100).reset_index(drop=True)
print(f"Running ablation on {len(df)} MIMIC documents\n")

# ── Output ────────────────────────────────────────────────────────────────────
output_dir = "/home/user/MLHC_AgenticSUM/outputs/agenticsum"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/checkpoints_ablation_mimic", exist_ok=True)

results = []
BATCH_SIZE = 5

print("=" * 80)
print("ABLATION STUDY — MIMIC-IV (100 documents, 4 conditions)")
print("=" * 80 + "\n")

with torch.no_grad():
    for batch_start in range(0, len(df), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(df))
        batch_df = df.iloc[batch_start:batch_end]

        print(f"\n{'='*80}")
        print(f"BATCH {batch_start//BATCH_SIZE + 1}: notes {batch_start+1}–{batch_end}")
        print(f"{'='*80}\n")

        for idx, row in batch_df.iterrows():
            doc     = row["input"]
            target  = row["target"]
            note_id = row["note_id"]

            print(f"\n[{idx+1}/100] {note_id} ({len(doc)} chars)")
            print("-" * 60)

            torch.cuda.empty_cache()
            gc.collect()

            # C1: Vanilla LLM — no compression, no detection
            try:
                print("  C1 Vanilla LLM...", end=" ", flush=True)
                c1 = draft_agent.generate(sent_tokenize(doc))
                print("done")
            except Exception as e:
                print(f"ERROR: {e}")
                c1 = f"ERROR: {e}"
            torch.cuda.empty_cache(); gc.collect()

            # C2: +FOCUS — compression + draft, no detection
            try:
                print("  C2 +FOCUS...", end=" ", flush=True)
                focus_out = focus_agent.compress(doc)
                c2 = draft_agent.generate(focus_out["sentences"])
                print("done")
            except Exception as e:
                print(f"ERROR: {e}")
                c2 = f"ERROR: {e}"
                focus_out = None
            torch.cuda.empty_cache(); gc.collect()

            # C3: +FOCUS +Detect +Fix (single pass, no supervisor loop)
            try:
                print("  C3 +FOCUS +Detect +Fix (1-pass)...", end=" ", flush=True)
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
            torch.cuda.empty_cache(); gc.collect()

            # C4: Full AgenticSum — iterative supervisor loop
            try:
                print("  C4 Full AgenticSum...", end=" ", flush=True)
                sup_out = supervisor.run(doc)
                c4 = sup_out["fixed_summary"]
                print(f"done ({sup_out['num_iterations']} iter, {sup_out['termination_reason']})")
            except Exception as e:
                print(f"ERROR: {e}")
                c4 = f"ERROR: {e}"
            torch.cuda.empty_cache(); gc.collect()

            results.append({
                "note_id":          note_id,
                "input":            doc,
                "target":           target,
                "vanilla_llm":      c1,
                "focus_draft":      c2,
                "focus_fix_single": c3,
                "agenticsum":       c4,
            })

        checkpoint_path = f"{output_dir}/checkpoints_ablation_mimic/batch_{batch_start//BATCH_SIZE + 1}.csv"
        pd.DataFrame(results).to_csv(checkpoint_path, index=False)
        print(f"\n💾 Checkpoint saved: {len(results)}/100\n")
        torch.cuda.empty_cache(); gc.collect()

results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_dir}/ablation_results_mimic.csv", index=False)
print(f"\nResults saved -> {output_dir}/ablation_results_mimic.csv")
print("=" * 80)
