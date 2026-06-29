import torch
import nltk
from typing import List, Dict, Any, Optional

nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize


class HallucinationDetectorAgent:
    """
    Evaluates a draft summary using:
    - Attention-based grounding (AURA)
    - Strict semantic entailment

    Returns detailed hallucination information including explanations.
    """

    def __init__(
        self,
        model,
        tokenizer,
        semantic_judge,
        device: str = "cuda",
        epsilon: float = 1e-8,
        aura_threshold: float = 0.42,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.semantic_judge = semantic_judge
        self.device = device
        self.epsilon = epsilon
        self.aura_threshold = aura_threshold

        # Internal caches
        self._cached_spans: Optional[List[str]] = None
        self._cached_token_aura: Optional[List[float]] = None
        self._cached_span_aura: Optional[Dict[int, float]] = None
        self._cached_document: Optional[str] = None
        self._aura_computed: bool = False

        # Store detailed hallucination information
        self._hallucination_details: Dict[int, Dict[str, Any]] = {}

        print("[Init] HallucinationDetectorAgent initialized")
        print(f"[Init] AURA threshold = {self.aura_threshold}")

    # FIX 1: Removed blank line so @torch.no_grad() actually decorates analyze()
    @torch.no_grad()
    def analyze(
        self,
        source_document: str,
        draft_summary: str,
    ) -> Dict[str, Any]:

        # FIX 4: Auto-reset caches if the source document has changed
        if self._cached_document is not None and self._cached_document != source_document:
            print("[Detector] New document detected — resetting caches")
            self.reset()

        # SKIP AURA for long documents - use semantic judge only
        doc_tokens = len(self.tokenizer.encode(source_document, add_special_tokens=False))
        if doc_tokens > 2048:
            print(f"[Detector] Long doc ({doc_tokens} tokens), skipping AURA - semantic judge only")
            spans = sent_tokenize(draft_summary)
            semantic_mask = self._semantic_entailment(source_document, spans)

            # Use semantic mask directly as hallucination mask
            hallucination_mask = semantic_mask

            print(
                f"[Detector] Spans={len(spans)} | "
                f"Hallucinated={sum(hallucination_mask.values())}"
            )

            return {
                "spans": spans,
                "span_aura_scores": {i: 0.5 for i in range(len(spans))},
                "hallucination_mask": hallucination_mask,
                "token_aura_scores": [],
                "hallucination_details": self._hallucination_details,
            }

        # Span freezing
        if self._cached_spans is None:
            self._cached_spans = sent_tokenize(draft_summary)
            print(f"[Detector] Cached {len(self._cached_spans)} spans")

        spans = self._cached_spans

        # AURA computation (once per document)
        if not self._aura_computed:
            self._cached_token_aura = self._compute_token_aura(
                source_document, draft_summary
            )
            self._cached_span_aura = self._aggregate_to_spans(
                draft_summary, spans, self._cached_token_aura
            )
            self._cached_document = source_document
            self._aura_computed = True
            print("[Detector] AURA computed and cached")
        else:
            print("[Detector] Using cached AURA from iteration 0")

        token_aura_scores = self._cached_token_aura
        span_aura_scores = self._cached_span_aura

        # Semantic entailment with detailed explanations
        semantic_mask = self._semantic_entailment(
            source_document, spans
        )

        # Final hallucination logic (OR)
        hallucination_mask: Dict[int, int] = {}

        for j in range(len(spans)):
            semantic_fail = semantic_mask[j] == 1
            weak_grounding = span_aura_scores[j] < self.aura_threshold

            hallucination_mask[j] = int(
                semantic_fail or weak_grounding
            )

        print(
            f"[Detector] Spans={len(spans)} | "
            f"Hallucinated={sum(hallucination_mask.values())}"
        )

        return {
            "spans": spans,
            "span_aura_scores": span_aura_scores,
            "hallucination_mask": hallucination_mask,
            "token_aura_scores": token_aura_scores,
            "hallucination_details": self._hallucination_details,
        }

    def _compute_token_aura(
        self,
        source_document: str,
        draft_summary: str,
    ) -> List[float]:
        """Compute token-level AURA scores with truncation to prevent OOM."""

        prompt = source_document + "\n\nSummary:\n"

        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=4096,
            truncation=True,
        ).input_ids.to(self.device)

        summary_ids = self.tokenizer(
            draft_summary,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=512,
            truncation=True,
        ).input_ids.to(self.device)

        input_ids = torch.cat([prompt_ids, summary_ids], dim=-1)

        T_prompt = prompt_ids.shape[-1]
        T_summary = summary_ids.shape[-1]

        # Free memory before forward pass
        del prompt_ids, summary_ids
        torch.cuda.empty_cache()

        outputs = self.model(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False,
        )

        # Move attention to CPU immediately
        attn = outputs.attentions[-1].squeeze(0).cpu()
        del outputs, input_ids
        torch.cuda.empty_cache()

        token_aura_scores: List[float] = []

        for t in range(T_summary):
            pos = T_prompt + t
            attn_t = attn[:, pos, :]

            numerator = attn_t[:, :T_prompt].sum(dim=1)
            denominator = attn_t.sum(dim=1) + self.epsilon

            token_aura_scores.append(
                (numerator / denominator).mean().item()
            )

        del attn
        torch.cuda.empty_cache()
        return token_aura_scores

    # FIX 2: Rewrote span aggregation to use character offsets correctly.
    # The old version counted only stripped token lengths against the full
    # span length (which includes whitespace), causing token index drift
    # across span boundaries.
    def _aggregate_to_spans(
        self,
        summary_text: str,
        spans: List[str],
        token_aura_scores: List[float],
    ) -> Dict[int, float]:
        """Aggregate token-level AURA to span-level using offset boundaries."""

        encoding = self.tokenizer(
            summary_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        offsets = encoding["offset_mapping"]

        # Build a mapping from each span to its character range in summary_text
        span_char_ranges: List[tuple] = []
        search_start = 0
        for span in spans:
            idx = summary_text.find(span, search_start)
            if idx == -1:
                # Fallback: if exact match fails, assign empty range
                span_char_ranges.append((search_start, search_start))
            else:
                span_char_ranges.append((idx, idx + len(span)))
                search_start = idx + len(span)

        # Assign each token to its span based on where its offset falls
        span_aura: Dict[int, float] = {j: 0.0 for j in range(len(spans))}
        span_token_counts: Dict[int, int] = {j: 0 for j in range(len(spans))}

        for token_idx, (tok_start, tok_end) in enumerate(offsets):
            if token_idx >= len(token_aura_scores):
                break  # Don't exceed available AURA scores

            # Skip pure-whitespace tokens
            if summary_text[tok_start:tok_end].strip() == "":
                continue

            # Find which span this token belongs to
            for j, (span_start, span_end) in enumerate(span_char_ranges):
                if tok_start >= span_start and tok_end <= span_end:
                    span_aura[j] += token_aura_scores[token_idx]
                    span_token_counts[j] += 1
                    break

        # Average per span
        for j in range(len(spans)):
            if span_token_counts[j] > 0:
                span_aura[j] /= span_token_counts[j]
            else:
                span_aura[j] = 0.0  # No tokens mapped — treat as ungrounded

        return span_aura

    def _semantic_entailment(
        self,
        source_document: str,
        spans: List[str],
    ) -> Dict[int, int]:
        """
        Evaluate semantic entailment and store detailed explanations.

        Returns:
            Dict mapping span index to 1 (unsupported) or 0 (supported)
        """

        hallucination_mask = {}
        self._hallucination_details = {}  # Reset

        for j, span in enumerate(spans):
            result = self.semantic_judge.judge(
                document=source_document,
                span=span,
            )

            # Mark as hallucinated if NOT supported
            is_hallucinated = int(not result["is_supported"])
            hallucination_mask[j] = is_hallucinated

            # Store detailed information for unsupported spans
            if is_hallucinated:
                self._hallucination_details[j] = {
                    "span": span,
                    "explanation": result.get("explanation", ""),
                    "problematic_spans": result.get("problematic_spans", ""),
                    "verdict": "UNSUPPORTED"
                }

        return hallucination_mask

    def reset(self):
        """Reset all caches for new document."""
        self._cached_spans = None
        self._cached_token_aura = None
        self._cached_span_aura = None
        self._cached_document = None
        self._aura_computed = False
        self._hallucination_details = {}
        print("[Detector] State reset")