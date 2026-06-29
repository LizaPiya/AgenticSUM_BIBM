import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class DraftAgent:
    def __init__(self, model, tokenizer, max_new_tokens=300):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.device = model.device
    
    def act(self, input_text: str) -> str:
        # Updated prompt for medical dialogue summarization in SOAP format
        prompt = (
            "You are a helpful medical assistant. Analyze the following patient-doctor dialogue and "
            "create a concise medical summary in SOAP format (Subjective, Objective, Assessment, Plan). "
            "Focus on key symptoms, findings, diagnosis, and treatment plan:\n\n"
            f"{input_text}\n\nSOAP Summary:"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the prompt prefix from the decoded text
        return summary.replace(prompt, "").strip()