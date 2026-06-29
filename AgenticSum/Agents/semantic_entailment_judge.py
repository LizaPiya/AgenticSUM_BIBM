import torch
from typing import Dict, Any


class SemanticEntailmentJudge:
    """
    Strict textual entailment judge with role-based prompting.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        max_new_tokens: int = 250,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._cache: Dict[str, Dict[str, Any]] = {}
        print("[Init] SemanticEntailmentJudge initialized")

    @torch.no_grad()
    def judge(self, document: str, span: str) -> Dict[str, Any]:
        """Returns entailment judgment for a single summary span."""
        
        cache_key = f"{hash(document)}::{hash(span)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = f"""Verify if this sentence is supported by the document.

Document:
{document}

Sentence:
{span}

Answer ONLY: SUPPORTED or NOT SUPPORTED"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Simple parsing: check if "not supported" appears anywhere
        is_supported = "not supported" not in response.lower()

        result = {
            "is_supported": is_supported,
            "raw_response": response,
            "explanation": response,
            "evidence": None,
            "problematic_spans": None,
        }
        
        self._cache[cache_key] = result
        return result

    def reset(self):
        """Reset cache for new document."""
        self._cache.clear()
        print("[SemanticJudge] Cache reset")