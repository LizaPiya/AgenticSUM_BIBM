import pandas as pd
from Evaluation import evaluate_summaries

OUTPUT_DIR  = "/home/user/MLHC_AgenticSUM/outputs/agenticsum"
RESULTS_CSV = f"{OUTPUT_DIR}/ablation_results_mimic.csv"

print("=" * 60)
print("Ablation Evaluation (ROUGE/BERTScore) — MIMIC-IV")
print("=" * 60)

df = pd.read_csv(RESULTS_CSV)
print(f"Loaded {len(df)} notes\n")

conditions = {
    "C1_Vanilla_LLM":      "vanilla_llm",
    "C2_FOCUS":            "focus_draft",
    "C3_FOCUS_Fix":        "focus_fix_single",
    "C4_AgenticSum":       "agenticsum",
}

for label, col in conditions.items():
    tmp = df[["note_id", "input", "target", col]].rename(columns={col: "fixed_summary"})
    tmp = tmp.dropna(subset=["fixed_summary"])
    print(f"\n--- {label} (n={len(tmp)}) ---")
    out = evaluate_summaries(tmp, summary_column="fixed_summary", reference_column="target")
    out["condition"] = label
    out_path = f"{OUTPUT_DIR}/eval_ablation_mimic_{label}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

print("\nDone.")
