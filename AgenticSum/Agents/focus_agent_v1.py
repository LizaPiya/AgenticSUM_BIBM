import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)


class FocusAgent:
    """
    FocusAgent: Sentence-level input compression via FOCUS.

    Uses self-attention to compute deterministic sentence salience scores.

    DESIGN GOALS:
    - GPU-accelerated
    - Batched processing
    - Deterministic
    - No generation
    - No truncation of selected sentences

    GUARANTEE:
    - Never returns empty output
    """

    def __init__(
        self,
        model,
        tokenizer,
        retention_ratio: float = 0.7,
        batch_size: int = 8,
        verbose: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.retention_ratio = retention_ratio
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
        self.verbose = verbose

    def _batch_sentence_scores(self, sentences: List[str]) -> List[float]:
        """
        Compute attention-based salience scores for a batch of sentences.
        
        Returns a list of scores (one per sentence).
        """
        scores = []
        
        num_batches = (len(sentences) + self.batch_size - 1) // self.batch_size
        
        iterator = range(0, len(sentences), self.batch_size)
        if self.verbose:
            iterator = tqdm(
                iterator, 
                total=num_batches,
                desc="[FocusAgent] Scoring sentences",
                leave=False
            )
        
        for i in iterator:
            batch = sentences[i:i + self.batch_size]
            
            try:
                # Tokenize batch
                encoding = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                
                # Move to device
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                
                attentions = outputs.attentions
                
                if attentions is None or len(attentions) == 0:
                    scores.extend([0.0] * len(batch))
                    continue
                
                # Last-layer attention: (batch_size, H, T, T)
                last_attn = attentions[-1]  # (B, H, T, T)
                
                # Compute per-sentence score
                for j in range(last_attn.size(0)):
                    # (H, T, T) -> mean over heads and tokens
                    token_importance = last_attn[j].mean(dim=0).mean(dim=0)
                    score = float(token_importance.mean().item())
                    
                    if not np.isfinite(score):
                        score = 0.0
                    
                    scores.append(score)
                
                # Cleanup
                del outputs, attentions, last_attn
                
            except RuntimeError as e:
                # OOM or other runtime error
                if self.verbose:
                    print(f"\n[FocusAgent] RuntimeError in batch {i}: {e}")
                scores.extend([0.0] * len(batch))
            
            except Exception as e:
                # Unexpected error
                if self.verbose:
                    print(f"\n[FocusAgent] Unexpected error in batch {i}: {e}")
                scores.extend([0.0] * len(batch))
        
        torch.cuda.empty_cache()
        return scores

    def compress(self, document: str) -> Dict[str, Any]:
        """
        Perform sentence-level input compression.

        RETURNS:
        - sentences: List[str]
        - sentence_indices: List[int]
        - sentence_scores: List[float]
        - fallback_used: bool
        """
        document = document.strip()
        fallback_used = False

        # Sentence segmentation
        sentences = sent_tokenize(document)

        if not sentences:
            return {
                "sentences": [document],
                "sentence_indices": [0],
                "sentence_scores": [1.0],
                "fallback_used": True,
            }

        if self.verbose:
            print(f"[FocusAgent] Processing {len(sentences)} sentences...")

        # Score sentences (batched)
        sentence_scores = self._batch_sentence_scores(sentences)
        
        # Sanitize scores
        sentence_scores = np.nan_to_num(
            sentence_scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).tolist()

        # Top-k selection
        m = len(sentences)
        k = max(1, int(np.floor(self.retention_ratio * m)))

        ranked = sorted(
            enumerate(sentence_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        selected_indices = sorted(idx for idx, _ in ranked[:k])
        selected_sentences = [sentences[i] for i in selected_indices]

        if self.verbose:
            print(f"[FocusAgent] Retained {len(selected_sentences)}/{len(sentences)} sentences")

        if not selected_sentences:
            return {
                "sentences": [document],
                "sentence_indices": [0],
                "sentence_scores": [1.0],
                "fallback_used": True,
            }

        return {
            "sentences": selected_sentences,
            "sentence_indices": selected_indices,
            "sentence_scores": [sentence_scores[i] for i in selected_indices],
            "fallback_used": fallback_used,
        }