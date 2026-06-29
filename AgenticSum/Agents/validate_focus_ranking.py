#!/usr/bin/env python3
"""
Validation: FOCUS attention-received ranking on 10 MIMIC notes.

Formula:
    beta_j = (1 / H|T_j|) * sum_h sum_{k in T_j} C_{h,k}
    C_{h,k} = sum_{i >= k} A^(L)_{h,i,k}   (last-layer column sum)

Purpose: verify whether top-ranked sentences are clinically sensible
         or simply reflect positional ordering (early = high score).
         Run this BEFORE committing to full pipeline reruns.

How to read the output:
    - "Retained positions: [1, 2, 3]" every time → positional bias dominates, stop.
    - "Retained positions: [1, 5, 12, 3, ...]" with clinical content → content signal present, proceed.
"""

import os
import math
import torch
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Config ────────────────────────────────────────────────────────────────────
N_NOTES     = 300
MAX_LENGTH  = 2048       # covers ~95% of MIMIC notes
RETENTION   = 0.3        # matches pipeline default
MODEL_NAME  = "meta-llama/Llama-3.2-3B-Instruct"
DATA_PATH   = "../Dataset/mimic-iv-bhc.csv"
OUTPUT_FILE = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/focus_validation_300.txt"

# ── Auth + GPU ────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
assert HF_TOKEN, "HUGGINGFACE_HUB_TOKEN not set"
assert torch.cuda.is_available(), "CUDA required"
print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# ── Model ─────────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading model (bfloat16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    token=HF_TOKEN,
    attn_implementation="eager",   # required for output_attentions=True
)
model.eval()
print("Model ready.\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_char_spans(document: str, sentences: list) -> list:
    """Return (char_start, char_end) for each sentence in document order."""
    spans = []
    cursor = 0
    for sent in sentences:
        idx = document.find(sent, cursor)
        if idx == -1:
            idx = cursor
        spans.append((idx, idx + len(sent)))
        cursor = idx + len(sent)
    return spans


def score_document(document: str):
    """
    Compute attention-received salience score for each sentence.

    Returns list of (sent_idx, sentence_text, beta_score) for all sentences.
    sent_idx is 0-indexed position in original document.
    """
    sentences = sent_tokenize(document)
    if not sentences:
        return []

    # Tokenize full document, keep offset map for sentence→token alignment
    encoding = tokenizer(
        document,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )
    offset_mapping = encoding["offset_mapping"][0].tolist()   # list of (char_s, char_e)
    n_tokens       = encoding["input_ids"].shape[1]
    input_ids      = encoding["input_ids"].to(model.device)
    attn_mask      = encoding["attention_mask"].to(model.device)

    # Forward pass — attn_implementation="eager" returns attention weights
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)

    if outputs.attentions is None:
        print("  [WARN] No attention weights returned — check attn_implementation.")
        return []

    # Extract last-layer attention: (1, H, T, T) → (H, T, T) in float32
    last_attn = outputs.attentions[-1][0].float().cpu()   # keep only last layer
    del outputs
    torch.cuda.empty_cache()

    # Column sums: C[h, k] = sum_i A[h, i, k]
    # Causal masking zeroes out A[h,i,k] for k>i, so summing all rows is safe.
    col_sums = last_attn.sum(dim=1)   # (H, T)
    del last_attn

    # Map each sentence to its token indices
    char_spans = get_char_spans(document, sentences)
    results    = []

    for sent_idx, (sent, (c_start, c_end)) in enumerate(zip(sentences, char_spans)):
        token_indices = [
            i for i, (ts, te) in enumerate(offset_mapping)
            if te > ts            # skip zero-width special tokens (BOS, padding)
            and ts < c_end
            and te > c_start
        ]

        if not token_indices:
            results.append((sent_idx, sent, 0.0))
            continue

        # beta_j = mean of col_sums over all heads and sentence token positions
        beta = col_sums[:, token_indices].mean().item()
        results.append((sent_idx, sent, beta))

    return results, n_tokens


# ── Data ──────────────────────────────────────────────────────────────────────
df     = pd.read_csv(DATA_PATH)
sample = df.head(N_NOTES)

# ── Run & collect output ──────────────────────────────────────────────────────
lines = []

def out(s=""):
    print(s)
    lines.append(s)

for _, row in sample.iterrows():
    note_id  = str(row["note_id"])
    document = str(row["input"]).strip()
    sentences = sent_tokenize(document)
    n_sents   = len(sentences)
    k         = max(1, int(math.floor(RETENTION * n_sents)))

    out(f"\n{'='*80}")
    out(f"NOTE: {note_id}  |  {n_sents} sentences  |  retain top {k}")
    out('='*80)

    result = score_document(document)
    if not result:
        out("  [SKIP] scoring failed.")
        continue

    scored, n_tokens = result
    truncated = n_tokens >= MAX_LENGTH
    if truncated:
        out(f"  [NOTE] Document truncated to {MAX_LENGTH} tokens — some sentences may have score 0.")

    ranked = sorted(scored, key=lambda x: x[2], reverse=True)

    retained_positions = [pos + 1 for pos, _, _ in ranked[:k]]
    out(f"Retained positions (1-indexed out of {n_sents}): {retained_positions}")
    out(f"  → Are these sequential from the top? If yes, positional bias is dominating.\n")

    out("RETAINED:")
    for rank, (pos, sent, beta) in enumerate(ranked[:k], 1):
        marker = "←EARLY" if pos < n_sents * 0.3 else ("←LATE" if pos >= n_sents * 0.7 else "")
        out(f"  Rank {rank:2d} | pos {pos+1:3d}/{n_sents} {marker:8s} | β={beta:.4f} | {sent[:100]}")

    out("\nDISCARDED:")
    for pos, sent, beta in ranked[k:]:
        out(f"         | pos {pos+1:3d}/{n_sents}          | β={beta:.4f} | {sent[:100]}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(lines))

print(f"\nResults saved → {OUTPUT_FILE}")
