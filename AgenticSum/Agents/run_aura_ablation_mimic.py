"""
run_aura_ablation_mimic.py
--------------------------
AURA-component ablation — MIMIC-IV (100 documents).

Full AgenticSum pipeline, only the detection signal varies:
  A1 - Semantic Only  : entailment only (AURA threshold = 0.0, never flags on AURA)
  A2 - AURA Only      : AURA only (mock semantic judge always returns SUPPORTED)
  A3 - AURA + Semantic: combined detector — current system (τ = 0.42)

Key settings (matching main pipeline):
  retention_ratio = 0.3
  aura_threshold  = 0.42 (A2, A3)
  No length filter
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


class AlwaysSupportedJudge:
    """Dummy judge that never flags hallucinations — isolates AURA signal."""
    def judge(self, document: str, span: str) -> Dict[str, Any]:
        return {"is_supported": True, "raw_response": "SUPPORTED",
                "explanation": "", "evidence": None, "problematic_spans": None}
    def reset(self):
        pass


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

print("Loading model...")
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

# ── Shared agents ─────────────────────────────────────────────────────────────
focus_agent        = FocusAgent(model=model, tokenizer=tokenizer, retention_ratio=0.3, batch_size=8)
draft_agent        = DraftAgent(model=model, tokenizer=tokenizer, max_new_tokens=256)
fix_agent          = FixAgent(model=model, tokenizer=tokenizer, max_new_tokens=150)
real_semantic_judge = SemanticEntailmentJudge(model=model, tokenizer=tokenizer)
mock_semantic_judge = AlwaysSupportedJudge()

# ── Three detectors ───────────────────────────────────────────────────────────
detector_a1 = HallucinationDetectorAgent(  # Semantic only
    model=model, tokenizer=tokenizer,
    semantic_judge=real_semantic_judge, aura_threshold=0.0,
)
detector_a2 = HallucinationDetectorAgent(  # AURA only
    model=model, tokenizer=tokenizer,
    semantic_judge=mock_semantic_judge, aura_threshold=0.42,
)
detector_a3 = HallucinationDetectorAgent(  # AURA + Semantic (full system)
    model=model, tokenizer=tokenizer,
    semantic_judge=real_semantic_judge, aura_threshold=0.42,
)

supervisor_a1 = ClinicalSupervisorAgent(focus_agent=focus_agent, draft_agent=draft_agent,
    hallucination_detector_agent=detector_a1, fix_agent=fix_agent, max_iterations=3)
supervisor_a2 = ClinicalSupervisorAgent(focus_agent=focus_agent, draft_agent=draft_agent,
    hallucination_detector_agent=detector_a2, fix_agent=fix_agent, max_iterations=3)
supervisor_a3 = ClinicalSupervisorAgent(focus_agent=focus_agent, draft_agent=draft_agent,
    hallucination_detector_agent=detector_a3, fix_agent=fix_agent, max_iterations=3)

print("All configurations initialized\n")

conditions = [
    ("A1 - Semantic Only",   supervisor_a1),
    ("A2 - AURA Only",       supervisor_a2),
    ("A3 - AURA + Semantic", supervisor_a3),
]

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("../Dataset/sample_data_100.csv").head(100).reset_index(drop=True)
print(f"Running AURA ablation on {len(df)} MIMIC documents\n")

# ── Output ────────────────────────────────────────────────────────────────────
output_dir = "/home/user/MLHC_AgenticSUM/outputs/agenticsum"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/checkpoints_aura_mimic", exist_ok=True)

results = []
BATCH_SIZE = 5

print("=" * 80)
print("AURA ABLATION — MIMIC-IV (100 documents, 3 conditions)")
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

            result = {"note_id": note_id, "input": doc, "target": target}

            for label, supervisor in conditions:
                torch.cuda.empty_cache(); gc.collect()
                try:
                    print(f"  {label}...", end=" ", flush=True)
                    out = supervisor.run(doc)
                    result[label] = out["fixed_summary"]
                    print(f"done ({out['num_iterations']} iter, {out['termination_reason']})")
                except Exception as e:
                    print(f"ERROR: {e}")
                    result[label] = f"ERROR: {e}"

            results.append(result)
            torch.cuda.empty_cache(); gc.collect()

        checkpoint_path = f"{output_dir}/checkpoints_aura_mimic/batch_{batch_start//BATCH_SIZE + 1}.csv"
        pd.DataFrame(results).to_csv(checkpoint_path, index=False)
        print(f"\n💾 Checkpoint saved: {len(results)}/100\n")
        torch.cuda.empty_cache(); gc.collect()

results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_dir}/aura_ablation_100docs_MIMIC.csv", index=False)
print(f"\nResults saved -> {output_dir}/aura_ablation_100docs_MIMIC.csv")
print("=" * 80)
