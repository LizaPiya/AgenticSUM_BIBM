"""
AgenticSum with Mistral-7B-Instruct-v0.3 — SOAP only.
Rerun after MIMIC completed successfully in job 594.
OOM fix: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set in sbatch.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM

from focus_agent import FocusAgent
from draft_agent import DraftAgent
from HallucinationDetectorAgent import HallucinationDetectorAgent
from FixAgent import FixAgent
from ClinicalSupervisorAgent import ClinicalSupervisorAgent
from semantic_entailment_judge import SemanticEntailmentJudge
from Evaluation import evaluate_summaries
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

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
assert HF_TOKEN is not None, "HUGGINGFACE_HUB_TOKEN not set"

# ======================================================
# Model
# ======================================================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

print(f"Loading {MODEL_NAME}...")
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
print(f"✅ Model loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB\n")

# ======================================================
# Agents — identical parameters to original
# ======================================================
focus_agent = FocusAgent(model=model, tokenizer=tokenizer, retention_ratio=0.3, batch_size=8)
semantic_judge = SemanticEntailmentJudge(model=model, tokenizer=tokenizer)
draft_agent = DraftAgent(model=model, tokenizer=tokenizer, max_new_tokens=256)
hallucination_detector_agent = HallucinationDetectorAgent(
    model=model, tokenizer=tokenizer, semantic_judge=semantic_judge
)
fix_agent = FixAgent(model=model, tokenizer=tokenizer, max_new_tokens=150)
supervisor = ClinicalSupervisorAgent(
    focus_agent=focus_agent,
    draft_agent=draft_agent,
    hallucination_detector_agent=hallucination_detector_agent,
    fix_agent=fix_agent,
    max_iterations=3,
)
print("✅ All agents initialized\n")

# ======================================================
# Output
# ======================================================
OUTPUT_DIR = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/mistral7b_agenticsum"
os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)

BATCH_SIZE = 5

# ======================================================
# Run SOAP pipeline
# ======================================================
df = pd.read_csv("../Dataset/df_soap_mimic.csv")
print(f"Starting AgenticSum-Mistral7B [SOAP] — {len(df)} samples\n")

results = []

with torch.no_grad():
    for batch_start in range(0, len(df), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(df))
        batch_df = df.iloc[batch_start:batch_end]

        print(f"\nBATCH {batch_start//BATCH_SIZE + 1}: rows {batch_start+1}-{batch_end}")

        for idx, row in batch_df.iterrows():
            try:
                print(f"[{idx+1}/{len(df)}] Processing {row['note_id']}...", end=" ")
                torch.cuda.empty_cache()
                gc.collect()

                output = supervisor.run(row["input"])
                results.append({
                    "note_id": row["note_id"],
                    "input": row["input"],
                    "target": row["target"],
                    "draft_summary": output["draft_summary"],
                    "fixed_summary": output["fixed_summary"],
                })
                print("✅")

            except Exception as e:
                print(f"❌ ERROR: {str(e)[:80]}")
                results.append({
                    "note_id": row.get("note_id", "NA"),
                    "input": row.get("input", ""),
                    "target": row.get("target", ""),
                    "draft_summary": f"ERROR: {str(e)}",
                    "fixed_summary": "ERROR",
                })

        ckpt = f"{OUTPUT_DIR}/checkpoints/soap_batch_{batch_start//BATCH_SIZE+1}.csv"
        pd.DataFrame(results).to_csv(ckpt, index=False)
        print(f"💾 Checkpoint: {len(results)} notes\n")

        torch.cuda.empty_cache()
        gc.collect()

results_df = pd.DataFrame(results)
soap_path = f"{OUTPUT_DIR}/mistral7b_soap_summaries.csv"
results_df.to_csv(soap_path, index=False)

success = len([r for r in results if "ERROR" not in str(r.get("fixed_summary", ""))])
print(f"\n✅ SOAP COMPLETE — {success}/{len(df)} successful")

# ======================================================
# Evaluation metrics
# ======================================================
df_out = pd.read_csv(soap_path)
df_out = evaluate_summaries(df_out, summary_column="fixed_summary", reference_column="target")
df_out.to_csv(soap_path, index=False)

# ======================================================
# LLM-as-judge — Meta-Llama-3-8B-Instruct float16
# ======================================================
print("\nFreeing Mistral, loading judge...")
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
print(f"GPU mem after free: {torch.cuda.memory_allocated()/1e9:.1f} GB")

from transformers import AutoTokenizer, AutoModelForCausalLM
JUDGE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
judge_tok = AutoTokenizer.from_pretrained(JUDGE_MODEL, token=HF_TOKEN)
judge_tok.pad_token = judge_tok.eos_token
judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL, device_map="auto", torch_dtype=torch.float16, token=HF_TOKEN
).eval()
print(f"Judge loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

out_path = f"{OUTPUT_DIR}/mistral7b_soap_results.csv"
df_final = llm_hallucination_evaluation(
    model=judge_model,
    tokenizer=judge_tok,
    csv_path=soap_path,
    output_path=out_path,
)

print(f"\n{'='*60}")
print("FINAL RESULTS — AgenticSum-Mistral7B — SOAP")
print(f"{'='*60}")
for label, col in [
    ("ROUGE-L",         "rouge_l"),
    ("BLEU-1",          "bleu1"),
    ("BLEU-2",          "bleu2"),
    ("BERTScore F1",    "bert_f1"),
    ("Hallucination ↓", "hallucination_score"),
    ("Factual Cons. ↑", "factual_consistency"),
    ("Completeness ↑",  "completeness"),
    ("Coherence ↑",     "coherence"),
]:
    if col in df_final.columns:
        print(f"  {label:<20} {df_final[col].mean():.2f} ± {df_final[col].std():.2f}")

print("\nDone. Results in:", OUTPUT_DIR)
