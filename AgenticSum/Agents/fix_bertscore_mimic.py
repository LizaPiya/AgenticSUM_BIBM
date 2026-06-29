"""
Recompute BERTScore for AgenticSum-Mistral7B MIMIC results.
Summaries already saved from job 594 — just computing missing metric.
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from Evaluation import compute_bert_score_batched, clean_text

CSV_PATH = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/mistral7b_agenticsum/mistral7b_mimic_summaries.csv"

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} samples")

references = [clean_text(t) for t in df["target"]]
candidates = [clean_text(t) for t in df["fixed_summary"]]

print("Computing BERTScore...")
bert_p, bert_r, bert_f1 = compute_bert_score_batched(references, candidates)

df["bert_p"]  = bert_p
df["bert_r"]  = bert_r
df["bert_f1"] = bert_f1

df.to_csv(CSV_PATH, index=False)

import numpy as np
print(f"\nBERTScore F1: {df['bert_f1'].mean():.2f} ± {df['bert_f1'].std():.2f}")
print(f"Saved → {CSV_PATH}")
