import torch
from typing import List


class DraftAgent:
    """
    DraftAgent: Deterministic clinical summary generation.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 350
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.device = next(self.model.parameters()).device
        print("[Init] DraftAgent initialized (deterministic mode)")
    
    def generate(self, compressed_sentences: List[str]) -> str:
        if not compressed_sentences:
            return ""
        
        context = " ".join(compressed_sentences)
        
        prompt = (
            "Based on the patient record below, write a concise clinical summary "
            "describing the patient's hospital stay. Include reason for admission, "
            "key findings, and treatments given. Use only information from the record. "
            "Do NOT add notes, commentary, or explanations about the summary itself.\n\n"
            
            f"{context}\n\n"
            
            "Summary:"
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        decoded = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        
        return decoded.split("Summary:", 1)[-1].strip()