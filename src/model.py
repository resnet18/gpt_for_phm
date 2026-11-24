import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    return model, tokenizer