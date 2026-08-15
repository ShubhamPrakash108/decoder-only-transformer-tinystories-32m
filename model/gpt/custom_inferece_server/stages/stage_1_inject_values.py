import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
sys.path.append(project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from model.gpt.custom_inferece_server.stages.stage_0_mha_fix import _GPTBlock
import shutil
import torch
import warnings
from pathlib import Path
from threading import Thread
from huggingface_hub import snapshot_download
from transformers import (
    TextIteratorStreamer,
    logging,
)
from utils.file_utils import load_yaml
config = load_yaml('config/config.yml')

warnings.filterwarnings("ignore")
# logging.set_verbosity_error()
# logging.disable_progress_bar()


model_id = config["Inference"]["hf_model_link"]
model_directory = Path.cwd() / "downloaded_hf_model"

if not os.path.exists(model_directory):
    snapshot_download(
        repo_id=model_id,
        local_dir=model_directory,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_directory,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"Model: {model}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    prompt = "The transformer is based on"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 128,
        "do_sample": False,
        # "temperature": 0.2,
        "pad_token_id": tokenizer.eos_token_id,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print(f"\033[96m{prompt}\033[92m", end="", flush=True)

    for text in streamer:
        print(text, end="", flush=True)

    thread.join()
    print("\033[0m")





# 1. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_directory)

# 2. Load the custom model architecture and safetensors weights
# trust_remote_code=True is required to load modeling_gpt_custom.py
model = AutoModelForCausalLM.from_pretrained(model_directory, trust_remote_code=True)

print(f"Successfully loaded model with {sum(p.numel() for p in model.parameters())} parameters!")

mha = model.transformer_blocks[0].multihead_attention

d = mha.embed_dim 

W_q, W_k, W_v = mha.in_proj_weight.chunk(3, dim=0)   
b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0)     

W_out = mha.out_proj.weight   
b_out = mha.out_proj.bias    

print(f"Q Shape: {W_q.shape}")
print(f"K Shape: {W_k.shape}")
print(f"V Shape: {W_v.shape}")
print(f"Out Shape: {W_out.shape}")

print("\n--- Starting Automatic Weight Conversion ---")
from model.gpt.custom_inferece_server.stages.stage_0_mha_fix import GPTCustomConfig, GPTCustomForCausalLM

print("1. Building new custom architecture...")
new_config = GPTCustomConfig.from_pretrained(model_directory)
new_model = GPTCustomForCausalLM(new_config)

print("2. Copying base weights (Embeddings, LayerNorms, MLPs)...")
# strict=False allows us to ignore the missing MHA weights for now
new_model.load_state_dict(model.state_dict(), strict=False)

print("3. Injecting MHA weights for all transformer blocks...")
for i in range(new_config.number_of_transformer_block):
    old_mha = model.transformer_blocks[i].multihead_attention
    new_mha = new_model.transformer_blocks[i].multihead_attention
    
    # Extract from old
    W_q, W_k, W_v = old_mha.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = old_mha.in_proj_bias.chunk(3, dim=0)
    
    # INJECT into new
    new_mha.query_proj.weight.data.copy_(W_q)
    new_mha.query_proj.bias.data.copy_(b_q)
    
    new_mha.key_proj.weight.data.copy_(W_k)
    new_mha.key_proj.bias.data.copy_(b_k)
    
    new_mha.value_proj.weight.data.copy_(W_v)
    new_mha.value_proj.bias.data.copy_(b_v)
    
    # Copy the output projection directly
    new_mha.out_proj.weight.data.copy_(old_mha.out_proj.weight)
    new_mha.out_proj.bias.data.copy_(old_mha.out_proj.bias)

print("Successfully copied all weights into the new CustomCausalMHA!")

converted_directory = Path.cwd() / "converted_model"
print(f"4. Saving brand new model to: {converted_directory}...")

# Fix: Hugging Face safetensors format crashes if two layers point to the exact same memory tensor 
# unless it's strictly registered. By cloning it here, we save it as two separate identical tensors.
# When the model loads later, the `tie_weights()` function will automatically tie them back together in memory!
new_model.final_linear_layer.weight = torch.nn.Parameter(new_model.token_embedding.weight.clone())

new_model.save_pretrained(converted_directory)
tokenizer.save_pretrained(converted_directory)
print("Done! You have successfully completed Stage 0.")

print("\nTesting the Converted Model")
device = "cuda" if torch.cuda.is_available() else "cpu"
new_model.to(device)
new_model.eval()

prompt = "The transformer is based on"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

generation_kwargs = {
    **inputs,
    "streamer": streamer,
    "max_new_tokens": 128,
    "do_sample": False,
    "pad_token_id": tokenizer.eos_token_id,
}

thread = Thread(target=new_model.generate, kwargs=generation_kwargs)
thread.start()

print(f"\n\033[96m[Prompt]: {prompt}\033[92m", end="", flush=True)

for text in streamer:
    print(text, end="", flush=True)

thread.join()
print("\033[0m\n")
print("If the generated text above makes sense, your new MHA architecture is mathematically identical to the original!")
