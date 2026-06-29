import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_as_a_judge import llm_hallucination_evaluation

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

assert torch.cuda.is_available(), "CUDA required"
print("=" * 60)
print("LLM-as-a-Judge Evaluation — SOAP")
print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
assert HF_TOKEN, "HUGGINGFACE_HUB_TOKEN not set"

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    token=HF_TOKEN,
    attn_implementation="eager",
)
model.eval()
print("Model loaded\n")

RESULTS_CSV = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/agenticsum_results_soap.csv"
OUTPUT_CSV  = "/home/lizapiya/MLHC_AgenticSUM/outputs/agenticsum/judge_scores_soap_8b.csv"

llm_hallucination_evaluation(
    model=model,
    tokenizer=tokenizer,
    csv_path=RESULTS_CSV,
    output_path=OUTPUT_CSV,
)
