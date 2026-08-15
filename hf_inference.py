import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_REPO_ID = "shubhamprakash108/gpt-model-2-decoder-100000-tiny-stories-fp16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading tokenizer and model weights from {HF_REPO_ID}...")

tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(HF_REPO_ID, trust_remote_code=True).to(DEVICE)
model.eval()

def generate(prompt, max_new_tokens=100, temperature=0.8, top_p=0.95, top_k=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(generate("The transformer architecture is"))
