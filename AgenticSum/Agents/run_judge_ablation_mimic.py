import os, gc, random
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_as_a_judge import llm_hallucination_evaluation

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

assert torch.cuda.is_available(), "CUDA required"
print("=" * 60)
print("LLM-as-a-Judge Ablation (C1-C4) — MIMIC-IV")
print(f"GPU : {torch.cuda.get_device_name(0)}")
print("=" * 60)

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
assert HF_TOKEN, "HUGGINGFACE_HUB_TOKEN not set"

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    low_cpu_mem_usage=True, token=HF_TOKEN, attn_implementation="eager",
)
model.eval()
print("Model loaded\n")

OUTPUT_DIR  = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum"
RESULTS_CSV = f"{OUTPUT_DIR}/ablation_results_mimic.csv"

df = pd.read_csv(RESULTS_CSV)
print(f"Loaded {len(df)} notes\n")

conditions = {
    "C1_Vanilla_LLM":  "vanilla_llm",
    "C2_FOCUS":        "focus_draft",
    "C3_FOCUS_Fix":    "focus_fix_single",
    "C4_AgenticSum":   "agenticsum",
}

summary_rows = []

for label, col in conditions.items():
    print(f"\n{'─'*50}\n {label}\n{'─'*50}")
    tmp = df[["note_id","input","target",col]].rename(columns={col:"fixed_summary"})
    tmp = tmp.dropna(subset=["fixed_summary"]).reset_index(drop=True)

    temp_in  = f"{OUTPUT_DIR}/_tmp_judge_in.csv"
    temp_out = f"{OUTPUT_DIR}/_tmp_judge_out.csv"
    tmp.to_csv(temp_in, index=False)

    torch.cuda.empty_cache(); gc.collect()
    scored = llm_hallucination_evaluation(model=model, tokenizer=tokenizer,
                                          csv_path=temp_in, output_path=temp_out)
    scored["condition"] = label
    out_path = f"{OUTPUT_DIR}/judge_ablation_mimic_{label}.csv"
    scored.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}")

    summary_rows.append({
        "condition":           label,
        "n":                   len(scored),
        "hallucination_score": round(scored["hallucination_score"].mean(), 2),
        "factual_consistency": round(scored["factual_consistency"].mean(), 2),
        "completeness":        round(scored["completeness"].mean(), 2),
        "coherence":           round(scored["coherence"].mean(), 2),
    })
    torch.cuda.empty_cache(); gc.collect()

for f in [temp_in, temp_out]:
    if os.path.exists(f): os.remove(f)

summary_df = pd.DataFrame(summary_rows)
print("\n" + "=" * 60)
print(summary_df.to_string(index=False))
summary_path = f"{OUTPUT_DIR}/judge_ablation_mimic_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary saved -> {summary_path}")
