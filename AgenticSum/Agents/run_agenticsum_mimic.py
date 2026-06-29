import os
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


# ======================================================
# Reproducibility
# ======================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

assert torch.cuda.is_available(), "CUDA is not available"
print("="*80)
print("Using GPU:", torch.cuda.get_device_name(0))
print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
print("="*80)


# ======================================================
# Hugging Face token (from environment, sbatch-safe)
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

# REQUIRED: attention access for FocusAgent, DraftAgent, HallucinationDetector
model.config.output_attentions = True
model.eval()
print("✅ Model loaded with attention outputs enabled\n")


# ======================================================
# Initialize agents
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
# Load dataset — all 100 MIMIC notes
# ======================================================
data_path = "../Dataset/sample_data_100.csv"
df = pd.read_csv(data_path)

print(f"Loaded {len(df)} notes from {data_path}\n")


# ======================================================
# Output setup
# ======================================================
output_dir = "/home/user/MLHC_AgenticSUM/outputs/agenticsum"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/checkpoints_mimic", exist_ok=True)

final_output_path = f"{output_dir}/agenticsum_results_mimic.csv"

results = []
BATCH_SIZE = 5


# ======================================================
# Run AgenticSum pipeline
# ======================================================
print("="*80)
print("Starting AgenticSum Pipeline — MIMIC-IV (100 notes)")
print("="*80 + "\n")

with torch.no_grad():
    for batch_start in range(0, len(df), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(df))
        batch_df  = df.iloc[batch_start:batch_end]

        print(f"\n{'='*80}")
        print(f"BATCH {batch_start//BATCH_SIZE + 1}: notes {batch_start+1}–{batch_end}")
        print(f"{'='*80}\n")

        for idx, row in batch_df.iterrows():
            try:
                print(f"[{idx+1}/{len(df)}] {row['note_id']} ({len(row['input'])} chars)...", end=" ")

                torch.cuda.empty_cache()
                gc.collect()

                output = supervisor.run(row["input"])

                results.append({
                    "note_id":       row["note_id"],
                    "input":         row["input"],
                    "target":        row["target"],
                    "draft_summary": output["draft_summary"],
                    "fixed_summary": output["fixed_summary"],
                })

                print("✅")

            except Exception as e:
                print(f"❌ ERROR: {str(e)[:80]}")
                results.append({
                    "note_id":       row.get("note_id", "NA"),
                    "input":         row.get("input", ""),
                    "target":        row.get("target", ""),
                    "draft_summary": f"ERROR: {str(e)}",
                    "fixed_summary": "ERROR",
                })

            torch.cuda.empty_cache()
            gc.collect()

        # Checkpoint after each batch
        checkpoint_path = f"{output_dir}/checkpoints_mimic/batch_{batch_start//BATCH_SIZE + 1}.csv"
        pd.DataFrame(results).to_csv(checkpoint_path, index=False)
        print(f"\n💾 Checkpoint saved: {len(results)}/{len(df)} notes\n")

        torch.cuda.empty_cache()
        gc.collect()


# ======================================================
# Final save
# ======================================================
results_df = pd.DataFrame(results)
results_df.to_csv(final_output_path, index=False)

success = len([r for r in results if "ERROR" not in str(r.get("fixed_summary", ""))])

print("\n" + "="*80)
print("✅ PIPELINE COMPLETE — MIMIC-IV")
print(f"Total: {len(results)}/{len(df)}  |  Successful: {success}  |  Errors: {len(results)-success}")
print(f"Results saved → {final_output_path}")
print("="*80)
