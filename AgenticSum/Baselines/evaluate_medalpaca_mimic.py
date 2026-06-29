from tqdm import tqdm
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
from rouge_metric import PyRouge  # You can also use `rouge_score` from Hugging Face

def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return ' '.join(text.strip().lower().split())

def compute_bleu_scores(reference, candidate):
    try:
        smoothing = SmoothingFunction().method1
        bleu1 = sentence_bleu([reference.split()], candidate.split(), weights=(1.0, 0, 0, 0), smoothing_function=smoothing)
        bleu2 = sentence_bleu([reference.split()], candidate.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        return bleu1 * 100, bleu2 * 100
    except Exception as e:
        print(f"BLEU Error: {e}")
        return 0.0, 0.0

def compute_rouge_l(reference, candidate):
    rouge = PyRouge(rouge_n=(1, 2), rouge_l=True)
    try:
        scores = rouge.evaluate([candidate], [[reference]])
        return scores['rouge-l']['f'] * 100
    except Exception as e:
        print(f"ROUGE-L Error: {e}")
        return 0.0

def compute_bert_score_batched(references, candidates, batch_size=32):
    all_P, all_R, all_F1 = [], [], []
    for i in range(0, len(references), batch_size):
        refs = references[i:i + batch_size]
        cands = candidates[i:i + batch_size]
        try:
            P, R, F1 = score(cands, refs, lang="en", verbose=False)
            all_P.extend([p * 100 for p in P.tolist()])
            all_R.extend([r * 100 for r in R.tolist()])
            all_F1.extend([f * 100 for f in F1.tolist()])
        except Exception as e:
            print(f"BERTScore Error in batch {i}: {e}")
            all_P.extend([0.0] * len(refs))
            all_R.extend([0.0] * len(refs))
            all_F1.extend([0.0] * len(refs))
    return all_P, all_R, all_F1

def evaluate_medalpaca_summaries(df, summary_column='generated_summary', reference_column='target_summary'):
    print(f"📊 Evaluating {len(df)} MedAlpaca-generated summaries...")
    print(f"📝 Summary column: {summary_column}")
    print(f"🎯 Reference column: {reference_column}")
    
    # Check if columns exist
    if summary_column not in df.columns:
        raise ValueError(f"Summary column '{summary_column}' not found. Available columns: {list(df.columns)}")
    if reference_column not in df.columns:
        raise ValueError(f"Reference column '{reference_column}' not found. Available columns: {list(df.columns)}")
    
    bleu1_scores, bleu2_scores, rouge_l_scores = [], [], []
    print("\n🔢 Computing BLEU and ROUGE-L scores...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
        reference = clean_text(row[reference_column])
        candidate = clean_text(row[summary_column])

        if not reference or not candidate:
            bleu1_scores.append(0.0)
            bleu2_scores.append(0.0)
            rouge_l_scores.append(0.0)
        else:
            bleu1, bleu2 = compute_bleu_scores(reference, candidate)
            rouge_l = compute_rouge_l(reference, candidate)
            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            rouge_l_scores.append(rouge_l)

    print("\n🧠 Computing BERTScore...")
    references = [clean_text(text) for text in df[reference_column]]
    candidates = [clean_text(text) for text in df[summary_column]]
    bert_p, bert_r, bert_f1 = compute_bert_score_batched(references, candidates)

    # Add scores to dataframe
    df['bleu1'] = bleu1_scores
    df['bleu2'] = bleu2_scores
    df['rouge_l'] = rouge_l_scores
    df['bert_p'] = bert_p
    df['bert_r'] = bert_r
    df['bert_f1'] = bert_f1

    # Summary stats
    print("\n" + "="*80)
    print("📊 MEDALPACA-7B EVALUATION RESULTS")
    print("="*80)
    
    metrics = ['bleu1', 'bleu2', 'rouge_l', 'bert_p', 'bert_r', 'bert_f1']
    print(f"{'Metric':<12} {'Mean ± Std':<20} {'Min':<8} {'Max':<8} {'Median':<8}")
    print("-" * 70)
    
    for metric in metrics:
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        min_val = df[metric].min()
        max_val = df[metric].max()
        median_val = df[metric].median()
        print(f"{metric.upper():<12} {mean_val:.3f} ± {std_val:.3f}    {min_val:<8.2f} {max_val:<8.2f} {median_val:<8.2f}")

    print("\n" + "="*80)
    print("📈 BASELINE METRICS SUMMARY (copy this into your paper!)")
    print("="*80)
    print(f"BLEU-1: {df['bleu1'].mean():.2f} ± {df['bleu1'].std():.2f}")
    print(f"BLEU-2: {df['bleu2'].mean():.2f} ± {df['bleu2'].std():.2f}")
    print(f"ROUGE-L: {df['rouge_l'].mean():.2f} ± {df['rouge_l'].std():.2f}")
    print(f"BERTScore-F1: {df['bert_f1'].mean():.2f} ± {df['bert_f1'].std():.2f}")

    return df

def analyze_token_lengths(df):
    print("\n" + "="*80)
    print("📏 TOKEN LENGTH ANALYSIS")
    print("="*80)

    if 'summary_token_count' in df.columns:
        token_counts = df['summary_token_count']
        print(f"Generated Summary Tokens:")
        print(f"  Mean ± Std: {token_counts.mean():.1f} ± {token_counts.std():.1f}")
        print(f"  Target: 150 tokens")
        print(f"  Range: {token_counts.min()} - {token_counts.max()}")
        print(f"  Within 140-160: {len(df[(token_counts >= 140) & (token_counts <= 160)])}/{len(df)} "
              f"({len(df[(token_counts >= 140) & (token_counts <= 160)])/len(df)*100:.1f}%)")

    if 'target_tokens' in df.columns:
        target_tokens = df['target_tokens']
        print(f"\nTarget Summary Tokens:")
        print(f"  Mean ± Std: {target_tokens.mean():.1f} ± {target_tokens.std():.1f}")
        print(f"  Range: {target_tokens.min()} - {target_tokens.max()}")

def run_medalpaca_evaluation(csv_path="medalpaca_summaries_full_dataset.csv", output_path=None):
    print("🚀 Starting MedAlpaca-7B MIMIC Evaluation")
    print("="*80)

    df = pd.read_csv(csv_path)
    print(f"📂 Loaded {len(df)} samples from {csv_path}")
    print(f"📋 Columns: {list(df.columns)}")

    if output_path is None:
        output_path = csv_path.replace('.csv', '_evaluation_results.csv')

    df_evaluated = evaluate_medalpaca_summaries(df)
    analyze_token_lengths(df_evaluated)

    df_evaluated.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")

    print("\n" + "="*80)
    print("📋 SAMPLE RESULTS")
    print("="*80)
    print(f"Note ID: {df_evaluated.iloc[0]['note_id']}")
    print(f"Generated Summary Token Count: {df_evaluated.iloc[0]['summary_token_count']}")
    print(f"BLEU-1: {df_evaluated.iloc[0]['bleu1']:.2f}")
    print(f"ROUGE-L: {df_evaluated.iloc[0]['rouge_l']:.2f}")
    print(f"BERTScore-F1: {df_evaluated.iloc[0]['bert_f1']:.2f}")

    return df_evaluated

if __name__ == "__main__":
    df_results = run_medalpaca_evaluation("medalpaca_summaries_full_dataset.csv")
    print("\n🎉 MedAlpaca evaluation completed!")
