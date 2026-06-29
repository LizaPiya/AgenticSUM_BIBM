import os
import pandas as pd
import torch

from Evaluation import evaluate_summaries

RESULTS_CSV  = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/agenticsum_results_mimic.csv"
OUTPUT_CSV   = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/eval_metrics_mimic.csv"

print("=" * 60)
print("AgenticSum Evaluation — MIMIC-IV")
print("=" * 60)

df = pd.read_csv(RESULTS_CSV)
print(f"Loaded {len(df)} results\n")

results_df = evaluate_summaries(df, summary_column="fixed_summary", reference_column="target")
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved -> {OUTPUT_CSV}")
