import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

def llama_judge_mistral_evaluation(csv_path="mistral_7b_summaries_full_dataset.csv", output_path="mistral_7b_judge_results.csv"):
    """
    Hallucination evaluation of Mistral-7B summaries using Llama 3 8B as judge
    """
    print("🔄 Loading Llama 3 8B model as judge...")
    
    # Load Llama 3 8B model
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float16
    ).eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load your Mistral-7B results
    df = pd.read_csv(csv_path)
    print(f"📄 Loaded {len(df)} Mistral-7B generated summaries")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Store results
    results = []
    
    print("🔍 Evaluating Mistral-7B summaries for hallucinations...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            # Get your data columns - adapted for your CSV structure
            source = str(row['original_input'])  # The clinical note
            summary = str(row['generated_summary'])  # Mistral-7B generated summary
            
            # Skip if summary is an error
            if summary.startswith("ERROR:"):
                print(f"⚠️ Skipping error summary at index {idx}")
                results.append({
                    'hallucination_score': 5.0,  # Max hallucination for errors
                    'factual_consistency': 1.0,
                    'completeness': 1.0,
                    'coherence': 1.0
                })
                continue
            
            # Create evaluation prompt
            prompt = f"""Evaluate this medical summary against the source clinical note. Rate 1-5 for each criterion.

SOURCE CLINICAL NOTE:
{source[:2000]}...

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
            
            # Generate evaluation
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True,
                    repetition_penalty=1.1
                )
            
            # Extract response
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # Parse scores using regex
            halluc_match = re.search(r'Hallucination:\s*(\d+)', response)
            factual_match = re.search(r'Factual:\s*(\d+)', response)
            complete_match = re.search(r'Complete:\s*(\d+)', response)
            coherent_match = re.search(r'Coherent:\s*(\d+)', response)
            
            # Extract scores with fallbacks
            halluc_score = float(halluc_match.group(1)) if halluc_match else 3.0
            factual_score = float(factual_match.group(1)) if factual_match else 3.0
            complete_score = float(complete_match.group(1)) if complete_match else 3.0
            coherent_score = float(coherent_match.group(1)) if coherent_match else 3.0
            
            # Validate ranges (all 1-5 now)
            halluc_score = max(1, min(5, halluc_score))
            factual_score = max(1, min(5, factual_score))
            complete_score = max(1, min(5, complete_score))
            coherent_score = max(1, min(5, coherent_score))
            
        except Exception as e:
            print(f"⚠️ Error on sample {idx}: {e}")
            halluc_score, factual_score, complete_score, coherent_score = 3.0, 3.0, 3.0, 3.0
        
        # Store results
        results.append({
            'hallucination_score': halluc_score,
            'factual_consistency': factual_score,
            'completeness': complete_score,
            'coherence': coherent_score
        })
    
    # Add scores to dataframe
    for i, result in enumerate(results):
        for key, value in result.items():
            df.loc[i, key] = value
    
    # Calculate and print statistics
    print("\n📊 MISTRAL-7B HALLUCINATION EVALUATION RESULTS:")
    print("=" * 70)
    
    metrics = {
        'Hallucination (1-5)': 'hallucination_score',
        'Factual Consistency (1-5)': 'factual_consistency', 
        'Completeness (1-5)': 'completeness',
        'Coherence (1-5)': 'coherence'
    }
    
    print(f"{'Metric':<25} {'Mean ± Std':<15} {'Min':<6} {'Max':<6} {'Perfect Scores'}")
    print("-" * 70)
    
    for name, col in metrics.items():
        values = df[col]
        mean_val = values.mean()
        std_val = values.std()
        min_val = values.min()
        max_val = values.max()
        perfect = len(values[values == 5]) if col != 'hallucination_score' else len(values[values == 1])
        
        print(f"{name:<25} {mean_val:.2f} ± {std_val:.2f}   {min_val:<6.1f} {max_val:<6.1f} {perfect}/{len(df)}")
    
    # Additional analysis specific to baseline evaluation
    print(f"\n📋 BASELINE QUALITY INSIGHTS:")
    high_halluc = len(df[df['hallucination_score'] >= 4])
    low_factual = len(df[df['factual_consistency'] <= 2])
    good_complete = len(df[df['completeness'] >= 4])
    good_coherent = len(df[df['coherence'] >= 4])
    
    print(f"• High hallucination (≥4): {high_halluc}/{len(df)} ({high_halluc/len(df)*100:.1f}%)")
    print(f"• Low factual consistency (≤2): {low_factual}/{len(df)} ({low_factual/len(df)*100:.1f}%)")
    print(f"• Good completeness (≥4): {good_complete}/{len(df)} ({good_complete/len(df)*100:.1f}%)")
    print(f"• Good coherence (≥4): {good_coherent}/{len(df)} ({good_coherent/len(df)*100:.1f}%)")
    
    # Summary for baseline table
    print(f"\n📊 FOR BASELINE TABLE:")
    print(f"Hallucination: {df['hallucination_score'].mean():.2f} ± {df['hallucination_score'].std():.2f}")
    print(f"Factual Consistency: {df['factual_consistency'].mean():.2f} ± {df['factual_consistency'].std():.2f}")
    print(f"Completeness: {df['completeness'].mean():.2f} ± {df['completeness'].std():.2f}")
    print(f"Coherence: {df['coherence'].mean():.2f} ± {df['coherence'].std():.2f}")
    
    # Save results
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    # Run evaluation directly
    print("Running Mistral-7B evaluation...")
    results = llama_judge_mistral_evaluation()
    print("Done!")