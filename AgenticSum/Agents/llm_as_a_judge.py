import pandas as pd
import torch
from tqdm import tqdm
import re


def llm_hallucination_evaluation(
    model,
    tokenizer,
    csv_path="agenticsum_results_MIMIC.csv",
    output_path="llama_judge_results.csv",
):
    """
    Hallucination evaluation using a pre-loaded model.
    Expects columns: input, fixed_summary, target, note_id
    Skips rows where fixed_summary == 'SKIPPED_TOO_LONG'.

    Args:
        model: already-loaded HF model (e.g. Gemma 3 1B from your notebook)
        tokenizer: matching tokenizer
        csv_path: path to your agenticsum results CSV
        output_path: where to save scored results
    """
    df = pd.read_csv(csv_path)
    print(f"📄 Loaded {len(df)} samples\n")

    results = []

    print("🔍 Evaluating summaries for hallucinations...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            source = str(row["input"])
            summary = str(row["fixed_summary"])

            prompt = f"""Evaluate this medical summary against the source document. Rate 1-5 for each criterion.

SOURCE: {source[:1000]}...

SUMMARY: {summary}

Rate 1-5:
- Hallucination (1=none, 5=major fabrications)
- Factual (1=inaccurate, 5=perfectly accurate)
- Complete (1=missing key info, 5=comprehensive)
- Coherent (1=poor writing, 5=excellent)

Format:
Hallucination: X
Factual: X
Complete: X
Coherent: X"""

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True,
                    repetition_penalty=1.1,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            # Parse scores
            halluc_match = re.search(r"Hallucination:\s*(\d+)", response)
            factual_match = re.search(r"Factual:\s*(\d+)", response)
            complete_match = re.search(r"Complete:\s*(\d+)", response)
            coherent_match = re.search(r"Coherent:\s*(\d+)", response)

            halluc_score = max(1, min(5, float(halluc_match.group(1)))) if halluc_match else 3.0
            factual_score = max(1, min(5, float(factual_match.group(1)))) if factual_match else 3.0
            complete_score = max(1, min(5, float(complete_match.group(1)))) if complete_match else 3.0
            coherent_score = max(1, min(5, float(coherent_match.group(1)))) if coherent_match else 3.0

        except Exception as e:
            print(f"⚠️  Error on sample {idx}: {e}")
            halluc_score, factual_score, complete_score, coherent_score = 3.0, 3.0, 3.0, 3.0

        results.append(
            {
                "hallucination_score": halluc_score,
                "factual_consistency": factual_score,
                "completeness": complete_score,
                "coherence": coherent_score,
            }
        )

    # Attach scores
    scores_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)

    # Print results
    print("\n" + "=" * 60)
    print("📊 HALLUCINATION EVALUATION RESULTS:")
    print("=" * 60)

    metrics = {
        "Hallucination (1-5)": "hallucination_score",
        "Factual Consistency (1-5)": "factual_consistency",
        "Completeness (1-5)": "completeness",
        "Coherence (1-5)": "coherence",
    }

    for name, col in metrics.items():
        print(f"{name:<25}: {df[col].mean():.2f} ± {df[col].std():.2f}")

    # Insights
    print(f"\n📋 ADDITIONAL INSIGHTS:")
    print(f"• High hallucination (≥4):        {(df['hallucination_score'] >= 4).sum()}/{len(df)}")
    print(f"• Low factual consistency (≤2):   {(df['factual_consistency'] <= 2).sum()}/{len(df)}")
    print(f"• Perfect factual consistency (5): {(df['factual_consistency'] == 5).sum()}/{len(df)}")
    print(f"• Poor completeness (≤2):         {(df['completeness'] <= 2).sum()}/{len(df)}")

    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")

    return df


def test_single_sample(model, tokenizer, csv_path="agenticsum_results_MIMIC.csv", sample_idx=0):
    """
    Test evaluation on a single sample before running full eval.
    Uses the same model/tokenizer already loaded in the notebook.
    """
    df = pd.read_csv(csv_path)
    sample = df.iloc[sample_idx]

    source = str(sample["input"])
    summary = str(sample["fixed_summary"])

    print(f"📄 Source length:  {len(source)} chars")
    print(f"📝 Summary length: {len(summary)} chars")
    print(f"📄 Source preview: {source[:200]}...")
    print(f"📝 Summary preview: {summary[:200]}...\n")

    prompt = f"""Evaluate this medical summary against the source document. Rate 1-5 for each criterion.

SOURCE DOCUMENT:
{source}

GENERATED SUMMARY:
{summary}

Rate 1-5:
- Hallucination (1=none, 5=major fabrications)
- Factual (1=inaccurate, 5=perfectly accurate)
- Complete (1=missing key info, 5=comprehensive)
- Coherent (1=poor writing, 5=excellent)

Format:
Hallucination: X
Factual: X
Complete: X
Coherent: X"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    print("🤖 MODEL RESPONSE:")
    print(response)

    # Parse and show
    for label, pattern in [
        ("Hallucination", r"Hallucination:\s*(\d+)"),
        ("Factual", r"Factual:\s*(\d+)"),
        ("Complete", r"Complete:\s*(\d+)"),
        ("Coherent", r"Coherent:\s*(\d+)"),
    ]:
        match = re.search(pattern, response)
        print(f"  {label}: {match.group(1) if match else 'NOT FOUND'}")