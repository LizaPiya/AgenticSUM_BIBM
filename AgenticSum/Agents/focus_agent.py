import torch
import numpy as np
from typing import List, Dict, Any, Tuple

import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)


class FocusAgent:
    """
    FocusAgent: Sentence-level input compression via FOCUS.

    Scores sentences by attention received from subsequent tokens in the document.
    A sentence scores higher when its tokens attract greater attention from later
    parts of the document, reflecting their role as reference points for downstream
    contextual processing.

    Formula:
        beta_j = (1 / H|T_j|) * sum_h sum_{k in T_j} C_{h,k}
        C_{h,k} = sum_{i >= k} A^(L)_{h,i,k}   (last-layer column sum)

    where A^(L)_{h,i,k} is the attention weight from position i to position k
    in head h of the last transformer layer. In a causal decoder,
    A_{h,i,k} = 0 for k > i, so column sums accumulate only from
    subsequent tokens — not a constant, unlike row-sum aggregation.

    DESIGN GOALS:
    - Single forward pass over full document (document-level context)
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
        batch_size: int = 8,    # kept for API compatibility, not used
        verbose: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.retention_ratio = retention_ratio
        self.device = next(model.parameters()).device
        self.verbose = verbose

    def _get_char_spans(
        self, document: str, sentences: List[str]
    ) -> List[Tuple[int, int]]:
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

    def _document_scores(
        self, document: str, sentences: List[str]
    ) -> List[float]:
        """
        Score all sentences in one forward pass over the full document.

        Processes the entire document as a single sequence so each sentence
        is scored in its full document context, not in isolation.
        """
        try:
            # Tokenize full document — offset_mapping maps tokens to char positions
            encoding = self.tokenizer(
                document,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                return_offsets_mapping=True,
            )
            offset_mapping = encoding["offset_mapping"][0].tolist()   # (T, 2)
            input_ids      = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )

            if outputs.attentions is None or len(outputs.attentions) == 0:
                del outputs
                torch.cuda.empty_cache()
                return [0.0] * len(sentences)

            # Last-layer attention: (1, H, T, T) → move to CPU in float32
            last_attn = outputs.attentions[-1][0].float().cpu()   # (H, T, T)
            del outputs
            torch.cuda.empty_cache()

            # Column sums: C[h, k] = sum_i A[h, i, k]
            # Causal masking zeroes A[h,i,k] for k > i, so this equals
            # sum_{i >= k} A[h,i,k] without any extra masking needed.
            col_sums = last_attn.sum(dim=1)   # (H, T)
            del last_attn

            # Map each sentence's char span to token indices, then score
            char_spans = self._get_char_spans(document, sentences)
            scores     = []

            for (c_start, c_end) in char_spans:
                token_indices = [
                    i for i, (ts, te) in enumerate(offset_mapping)
                    if te > ts          # skip zero-width special tokens
                    and ts < c_end
                    and te > c_start
                ]
                if not token_indices:
                    # Sentence fell outside the truncated token window
                    scores.append(0.0)
                    continue

                beta = float(col_sums[:, token_indices].mean().item())
                scores.append(beta if np.isfinite(beta) else 0.0)

            return scores

        except RuntimeError as e:
            torch.cuda.empty_cache()
            if self.verbose:
                print(f"\n[FocusAgent] RuntimeError: {e}")
            return [0.0] * len(sentences)

        except Exception as e:
            torch.cuda.empty_cache()
            if self.verbose:
                print(f"\n[FocusAgent] Unexpected error: {e}")
            return [0.0] * len(sentences)

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

        sentence_scores = self._document_scores(document, sentences)

        sentence_scores = np.nan_to_num(
            sentence_scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).tolist()

        m = len(sentences)
        k = max(1, int(np.floor(self.retention_ratio * m)))

        ranked = sorted(
            enumerate(sentence_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        selected_indices   = sorted(idx for idx, _ in ranked[:k])
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
