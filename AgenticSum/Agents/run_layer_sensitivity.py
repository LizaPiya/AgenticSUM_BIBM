"""
run_layer_sensitivity.py
-------------------------
Sensitivity analysis for AURA layer selection and head aggregation strategy.

Addresses reviewer comment:
  "How sensitive are AURA scores to layer selection and head aggregation strategy?
   Does performance remain consistent if computed from earlier layers instead of the final layer?"

Tests the following configurations:
  Layer selection:
    - last       : final transformer layer (current default)
    - middle     : middle transformer layer
    - all_avg    : average across all layers

  Head aggregation:
    - mean (current default)
    - max

Combined into 6 conditions:
  last_mean, last_max, middle_mean, middle_max, all_avg_mean, all_avg_max

Runs on N_DOCS documents from MIMIC and SOAP.
Reports ROUGE-L, BLEU-1/2, BERTScore per configuration.

No existing files are modified.
"""

import os
import gc
import random
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from focus_agent import FocusAgent
from draft_agent import DraftAgent
from FixAgent import FixAgent
from ClinicalSupervisorAgent import ClinicalSupervisorAgent
from semantic_entailment_judge import SemanticEntailmentJudge
from HallucinationDetectorAgent import HallucinationDetectorAgent
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
# HF token
# ======================================================
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
assert HF_TOKEN is not None, "HUGGINGFACE_HUB_TOKEN not set"


# ======================================================
# Model + Tokenizer
# ======================================================
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


# ======================================================
# Subclass HallucinationDetectorAgent to vary layer + head
# ======================================================
class AURAVariantDetector(HallucinationDetectorAgent):
    """
    Extends HallucinationDetectorAgent to support different
    layer selection and head aggregation strategies.

    layer_mode: 'last' | 'middle' | 'all_avg'
    head_mode:  'mean' | 'max'
    """

    def __init__(self, model, tokenizer, semantic_judge,
                 layer_mode="last", head_mode="mean",
                 aura_threshold=0.42, **kwargs):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            semantic_judge=semantic_judge,
            aura_threshold=aura_threshold,
            **kwargs,
        )
        self.layer_mode = layer_mode
        self.head_mode  = head_mode

    def _compute_token_aura(self, source_document: str, draft_summary: str) -> List[float]:
        prompt = source_document + "\n\nSummary:\n"

        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", max_length=4096, truncation=True,
        ).input_ids.to(self.device)

        summary_ids = self.tokenizer(
            draft_summary, return_tensors="pt",
            add_special_tokens=False, max_length=512, truncation=True,
        ).input_ids.to(self.device)

        input_ids = torch.cat([prompt_ids, summary_ids], dim=-1)
        T_prompt   = prompt_ids.shape[-1]
        T_summary  = summary_ids.shape[-1]

        del prompt_ids, summary_ids
        torch.cuda.empty_cache()

        outputs = self.model(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False,
        )

        # --- Layer selection ---
        all_attn = outputs.attentions  # tuple of (1, n_heads, seq, seq)
        n_layers = len(all_attn)

        if self.layer_mode == "last":
            attn = all_attn[-1].squeeze(0).cpu()           # (n_heads, seq, seq)
        elif self.layer_mode == "middle":
            mid = n_layers // 2
            attn = all_attn[mid].squeeze(0).cpu()
        elif self.layer_mode == "all_avg":
            attn = torch.stack(
                [a.squeeze(0).cpu() for a in all_attn], dim=0
            ).mean(dim=0)                                  # (n_heads, seq, seq)
        else:
            raise ValueError(f"Unknown layer_mode: {self.layer_mode}")

        del outputs, input_ids
        torch.cuda.empty_cache()

        # --- Head aggregation ---
        token_aura_scores: List[float] = []
        for t in range(T_summary):
            pos   = T_prompt + t
            attn_t = attn[:, pos, :]                       # (n_heads, seq)

            numerator   = attn_t[:, :T_prompt].sum(dim=1) # (n_heads,)
            denominator = attn_t.sum(dim=1) + self.epsilon

            head_scores = numerator / denominator          # (n_heads,)

            if self.head_mode == "mean":
                score = head_scores.mean().item()
            elif self.head_mode == "max":
                score = head_scores.max().item()
            else:
                raise ValueError(f"Unknown head_mode: {self.head_mode}")

            token_aura_scores.append(score)

        del attn
        torch.cuda.empty_cache()
        return token_aura_scores


# ======================================================
# Configurations to test
# ======================================================
CONFIGS = [
    ("last_mean",    "last",    "mean"),   # current default
    ("last_max",     "last",    "max"),
    ("middle_mean",  "middle",  "mean"),
    ("middle_max",   "middle",  "max"),
    ("all_avg_mean", "all_avg", "mean"),
    ("all_avg_max",  "all_avg", "max"),
]
DEFAULT_CONFIG = "last_mean"


# ======================================================
# Datasets + sample size
# ======================================================
DATASETS = {
    "MIMIC": "../Dataset/sample_data_100.csv",
    "SOAP":  "../Dataset/df_soap_mimic.csv",
}
N_DOCS = 15

output_dir = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum"
os.makedirs(output_dir, exist_ok=True)


# ======================================================
# Shared agents
# ======================================================
focus_agent    = FocusAgent(model=model, tokenizer=tokenizer, retention_ratio=0.35, batch_size=8)
draft_agent    = DraftAgent(model=model, tokenizer=tokenizer, max_new_tokens=256)
fix_agent      = FixAgent(model=model, tokenizer=tokenizer, max_new_tokens=150)
semantic_judge = SemanticEntailmentJudge(model=model, tokenizer=tokenizer)
print("Shared agents initialized\n")


# ======================================================
# Run sensitivity analysis
# ======================================================
all_eval_rows = []

for dataset_name, data_path in DATASETS.items():
    print("\n" + "=" * 80)
    print(f"LAYER/HEAD SENSITIVITY — {dataset_name} — {N_DOCS} documents")
    print("=" * 80 + "\n")

    df = (
        pd.read_csv(data_path)[lambda x: x["input"].str.len() <= 6000]
        .head(N_DOCS)
        .reset_index(drop=True)
    )
    print(f"Loaded {len(df)} documents\n")

    results_by_config = {cfg[0]: [] for cfg in CONFIGS}

    with torch.no_grad():
        for idx, row in df.iterrows():
            doc     = row["input"]
            target  = row["target"]
            note_id = row["note_id"]

            print(f"\n[{idx+1}/{len(df)}] {note_id} ({len(doc)} chars)")
            print("-" * 60)

            for config_name, layer_mode, head_mode in CONFIGS:
                torch.cuda.empty_cache()
                gc.collect()

                detector = AURAVariantDetector(
                    model=model,
                    tokenizer=tokenizer,
                    semantic_judge=semantic_judge,
                    layer_mode=layer_mode,
                    head_mode=head_mode,
                    aura_threshold=0.42,
                )
                supervisor = ClinicalSupervisorAgent(
                    focus_agent=focus_agent,
                    draft_agent=draft_agent,
                    hallucination_detector_agent=detector,
                    fix_agent=fix_agent,
                    max_iterations=3,
                )

                tag = config_name + (" [DEFAULT]" if config_name == DEFAULT_CONFIG else "")
                try:
                    print(f"  {tag}...", end=" ", flush=True)
                    out = supervisor.run(doc)
                    summary = out["fixed_summary"]
                    print(f"done ({out['num_iterations']} iter, {out['termination_reason']})")
                except Exception as e:
                    print(f"ERROR: {e}")
                    summary = f"ERROR: {e}"

                results_by_config[config_name].append({
                    "note_id": note_id,
                    "input":   doc,
                    "target":  target,
                    "summary": summary,
                })

                del detector, supervisor
                torch.cuda.empty_cache()
                gc.collect()

    # Save raw results
    for config_name, _, _ in CONFIGS:
        cfg_df = pd.DataFrame(results_by_config[config_name])
        cfg_df.to_csv(f"{output_dir}/layer_sensitivity_{dataset_name}_{config_name}.csv", index=False)

    # Evaluate
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS — {dataset_name}")
    print("=" * 80)

    for config_name, layer_mode, head_mode in CONFIGS:
        cfg_df = pd.DataFrame(results_by_config[config_name])
        valid  = cfg_df[~cfg_df["summary"].str.startswith("ERROR", na=False)].copy()

        if valid.empty:
            print(f"\n{config_name}: all rows errored, skipping")
            continue

        tag = config_name + (" ← DEFAULT" if config_name == DEFAULT_CONFIG else "")
        print(f"\n{'─'*60}")
        print(f"  {tag}  (layer={layer_mode}, heads={head_mode}, n={len(valid)})")
        print(f"{'─'*60}")

        scored = evaluate_summaries(valid, summary_column="summary", reference_column="target")

        row = {
            "dataset":    dataset_name,
            "config":     config_name,
            "layer":      layer_mode,
            "heads":      head_mode,
            "is_default": config_name == DEFAULT_CONFIG,
            "n":          len(valid),
            "ROUGE-L":    round(scored["rouge_l"].mean(), 4),
            "BLEU-1":     round(scored["bleu1"].mean(), 4),
            "BLEU-2":     round(scored["bleu2"].mean(), 4),
            "BERTScore":  round(scored["bert_f1"].mean(), 4),
        }
        all_eval_rows.append(row)
        print(f"  ROUGE-L={row['ROUGE-L']}  BLEU-1={row['BLEU-1']}  "
              f"BLEU-2={row['BLEU-2']}  BERTScore={row['BERTScore']}")


# ======================================================
# Summary table
# ======================================================
print("\n" + "=" * 80)
print("LAYER / HEAD SENSITIVITY SUMMARY TABLE")
print(f"Default config: {DEFAULT_CONFIG}  |  tau=0.42 fixed across all")
print("=" * 80)

summary_df = pd.DataFrame(all_eval_rows)
print(summary_df.to_string(index=False))

scores_path = f"{output_dir}/layer_sensitivity_scores.csv"
summary_df.to_csv(scores_path, index=False)
print(f"\nSummary scores saved to: {scores_path}")
print("=" * 80)
