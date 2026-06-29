import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

def evaluate_medalpaca_summaries(
    csv_path="medalpaca_summaries_full_dataset.csv",
    output_path="medalpaca_judge_results.csv"
):
    """
    Hallucination evaluation of MedAlpaca summaries using LLaMA 3 8B as judge.
    """
    print("🔄 Loading LLaMA 3 8B model as judge...")

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float16
    ).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load summaries
    df = pd.read_csv(csv_path)
    print(f"📄 Loaded {len(df)} summaries from: {csv_path}")

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating summaries"):
        try:
            source = str(row["original_input"])
            summary = str(row["generated_summary"])

            if summary.startswith("ERROR"):
                results.append({'hallucination_score': 5.0, 'factual_consistency': 1.0, 'completeness': 1.0, 'coherence': 1.0})
                continue

            # Prompt
            prompt = f"""Evaluate this medical summary against the source clinical note. Rate 1-5 for each criterion.

SOURCE CLINICAL NOTE:
{source[:2000]}...

GENERATED SUMMARY:
{summary}

Rate 1-5:
- Hallucination (1=none, 5=major fabrications)
- Factual: (1=inaccurate, 5=perfectly accurate)
- Complete: (1=missing key info, 5=comprehensive)
- Coherent: (1=poor writing, 5=excellent)

Format:
Hallucination: X
Factual: X
Complete: X
Coherent: X"""

            # Tokenize and generate
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True
                )

            response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            halluc = int(re.search(r'Hallucination:\s*(\d)', response).group(1))
            factual = int(re.search(r'Factual:\s*(\d)', response).group(1))
            complete = int(re.search(r'Complete:\s*(\d)', response).group(1))
            coherent = int(re.search(r'Coherent:\s*(\d)', response).group(1))

        except Exception as e:
            print(f"⚠️ Error on index {idx}: {e}")
            halluc, factual, complete, coherent = 3, 3, 3, 3

        results.append({
            'hallucination_score': halluc,
            'factual_consistency': factual,
            'completeness': complete,
            'coherence': coherent
        })

    # Append results
    for i, res in enumerate(results):
        for key in res:
            df.loc[i, key] = res[key]

    # Save results
    df.to_csv(output_path, index=False)
    print(f"\n💾 Evaluation saved to: {output_path}")

    # Display aggregate results
    print("\n📊 MedAlpaca LLM-Judge Evaluation:")
    for col in ["hallucination_score", "factual_consistency", "completeness", "coherence"]:
        mean = df[col].mean()
        std = df[col].std()
        print(f"{col.replace('_', ' ').title()}: {mean:.2f} ± {std:.2f}")

    return df
