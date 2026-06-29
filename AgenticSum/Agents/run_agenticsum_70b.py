"""
AgenticSum with Llama-3.1-70B-Instruct backbone.

Identical to run_agenticsum.py and Soap_run_agenticsum.py in every way EXCEPT
the backbone model name (70B in 4-bit NF4) and attn_implementation.
All agent parameters, prompts, evaluation metrics, and judge setup unchanged.

Note: output_attentions=True is required for AURA. The HallucinationDetectorAgent
already handles 70B memory safely — only keeps the last attention layer and
immediately moves it to CPU.

Runs MIMIC and SOAP sequentially, then evaluation + LLM-as-judge.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

# ======================================================
# Hugging Face token (from environment, sbatch-safe)
# ======================================================
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
assert HF_TOKEN is not None, "HUGGINGFACE_HUB_TOKEN not set"

# ======================================================
# Model — 70B in 4-bit NF4 to fit on single GPU
# ======================================================
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"

print(f"Loading {MODEL_NAME} in 4-bit NF4...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    use_fast=True,
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_cfg,
    device_map="auto",
    low_cpu_mem_usage=True,
    token=HF_TOKEN,
    attn_implementation="eager",
)

# REQUIRED: attention access for FocusAgent + HallucinationDetectorAgent
model.config.output_attentions = True
model.eval()
print(f"✅ Model loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB\n")

# ======================================================
# Initialize agents — identical parameters to original
# ======================================================
print("Initializing agents...")

focus_agent = FocusAgent(
    model=model,
    tokenizer=tokenizer,
    retention_ratio=0.3,
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

hallucination_detector_agent = HallucinationDetectorAgent(
    model=model,
    tokenizer=tokenizer,
    semantic_judge=semantic_judge,
)

fix_agent = FixAgent(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
)

supervisor = ClinicalSupervisorAgent(
    focus_agent=focus_agent,
    draft_agent=draft_agent,
    hallucination_detector_agent=hallucination_detector_agent,
    fix_agent=fix_agent,
    max_iterations=3,
)

print("✅ All agents initialized\n")

# ======================================================
# Output setup
# ======================================================
OUTPUT_DIR = "/home/user/MLHC_AgenticSUM/outputs/agenticsum/agenticsum_70b"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)

DATASETS = {
    "mimic": "../Dataset/sample_data_100.csv",
    "soap":  "../Dataset/df_soap_mimic.csv",
}

BATCH_SIZE = 5


# ======================================================
# Run pipeline on a dataset
# ======================================================
def run_pipeline(dataset_name, data_path):
    df = pd.read_csv(data_path)
    print(f"\n{'='*80}")
    print(f"Starting AgenticSum-70B [{dataset_name.upper()}] — {len(df)} samples")
    print(f"{'='*80}\n")

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

            ckpt = f"{OUTPUT_DIR}/checkpoints/{dataset_name}_batch_{batch_start//BATCH_SIZE+1}.csv"
            pd.DataFrame(results).to_csv(ckpt, index=False)
            print(f"💾 Checkpoint: {len(results)} notes\n")

            torch.cuda.empty_cache()
            gc.collect()

    results_df = pd.DataFrame(results)
    out_path = f"{OUTPUT_DIR}/agenticsum70b_{dataset_name}_summaries.csv"
    results_df.to_csv(out_path, index=False)

    success = len([r for r in results if "ERROR" not in str(r.get("fixed_summary", ""))])
    print(f"\n✅ {dataset_name.upper()} COMPLETE — {success}/{len(df)} successful")
    print(f"Saved → {out_path}")
    return out_path


# ======================================================
# Step 1 — Run pipeline on both datasets
# ======================================================
for name, path in DATASETS.items():
    run_pipeline(name, path)
    torch.cuda.empty_cache()
    gc.collect()

# ======================================================
# Step 2 — Evaluation metrics (same Evaluation.py)
# ======================================================
print("\n" + "="*80)
print("Computing evaluation metrics...")

for name in DATASETS:
    summary_path = f"{OUTPUT_DIR}/agenticsum70b_{name}_summaries.csv"
    df = pd.read_csv(summary_path)
    df = evaluate_summaries(df, summary_column="fixed_summary", reference_column="target")
    df.to_csv(summary_path, index=False)

# ======================================================
# Step 3 — LLM-as-judge: Meta-Llama-3-8B-Instruct float16
#          Same model and function as Table 2
# ======================================================
print("\n" + "="*80)
print("Freeing 70B model, loading judge...")

del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
print(f"GPU mem after free: {torch.cuda.memory_allocated()/1e9:.1f} GB")

from transformers import AutoTokenizer, AutoModelForCausalLM
JUDGE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
judge_tok = AutoTokenizer.from_pretrained(JUDGE_MODEL, token=HF_TOKEN)
judge_tok.pad_token = judge_tok.eos_token
judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HF_TOKEN,
).eval()
print(f"Judge loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.1f} GB")

for name in DATASETS:
    summary_path = f"{OUTPUT_DIR}/agenticsum70b_{name}_summaries.csv"
    out_path     = f"{OUTPUT_DIR}/agenticsum70b_{name}_results.csv"

    df = llm_hallucination_evaluation(
        model=judge_model,
        tokenizer=judge_tok,
        csv_path=summary_path,
        output_path=out_path,
    )

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — AgenticSum-70B — {name.upper()}")
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
        if col in df.columns:
            print(f"  {label:<20} {df[col].mean():.2f} ± {df[col].std():.2f}")

print("\nAll done. Results in:", OUTPUT_DIR)
