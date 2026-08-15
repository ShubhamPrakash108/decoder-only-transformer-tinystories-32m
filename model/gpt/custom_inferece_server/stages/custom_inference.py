from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

input_text = "Transformer is based on ss"

tokenizer = AutoTokenizer.from_pretrained(r".\converted_model")

input_tokens = tokenizer(input_text, return_tensors='pt')['input_ids']
print("Input Tokens (Token IDs): ")
print(input_tokens[0])
print("Input Tokens Shape: ")
print(input_tokens.shape)

model = AutoModelForCausalLM.from_pretrained(r".\converted_model", trust_remote_code=True)
# print("Model: ")
# print(model)
output, cache = model(input_tokens, use_cache=True) # This call previously crashed downstream in the custom GPT block due to the Dropout tuple/tensor mismatch
print("Output Logits Shape: ")
print(output.logits.detach().shape)

print("Output Logits: ")
print(output.logits.detach())

print("Output Logits argmax: ")
print(output.logits.detach().argmax(dim=-1)[0])

print("Output Tokens: ")
print(tokenizer.decode(output.logits.detach().argmax(dim=-1)[0][-1]))


import psutil
from tqdm import tqdm
import os
import time

sequence_lengths = [128, 256, 512]
process = psutil.Process(os.getpid())

import platform

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_system_specs():
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== System Specifications ==={Colors.ENDC}")
    
    # RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Total System RAM: {ram_gb:.2f} GB")
    
    # CPU
    print(f"CPU: {platform.processor()} ({os.cpu_count()} logical cores)")
    
    # GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU(s) Available: {gpu_count}")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("GPU: Not Available (Using CPU)")
    print(f"{Colors.HEADER}{Colors.BOLD}============================={Colors.ENDC}\n")

print_system_specs()

print(f"Initial Prompt: {input_text}") 

analytics_results = []

for max_output_len in sequence_lengths:
    print(f"\n{Colors.CYAN}{Colors.BOLD}--- Testing Sequence Length: {max_output_len} ---{Colors.ENDC}")
    
    # Reset input tokens for each test
    current_input_tokens = tokenizer(input_text, return_tensors='pt')['input_ids']
    
    pbar = tqdm(range(max_output_len), desc=f"Generating {max_output_len} tokens")
    
    start_time = time.time()
    first_token_time = None
    ttft = 0.0
    
    for i in pbar:
        # wanted to see how kv cache looks like :) 
        output, cache = model(current_input_tokens, use_cache=True)
        if i == 0:
            import json
            cache_serializable = {}
            for idx, (block, (k, v)) in enumerate(cache.items()):
                cache_serializable[f"block_{idx}"] = {
                    "k_shape": list(k.shape),
                    "v_shape": list(v.shape),
                    "k_values": k.tolist(),
                    "v_values": v.tolist()
                }
            with open("cache.json", "w") as f:
                json.dump(cache_serializable, f, indent=2)
        # Get the next token ID
        next_token = output.logits.detach().argmax(dim=-1)[0][-1]
        
        # Append the new token to the sequence
        next_token_reshaped = next_token.unsqueeze(0).unsqueeze(0)
        current_input_tokens = torch.cat([current_input_tokens, next_token_reshaped], dim=-1)
        
        # Record Time to First Token (TTFT)
        if i == 0:
            first_token_time = time.time()
            ttft = first_token_time - start_time
            
        current_time = time.time()
        elapsed_since_first = current_time - first_token_time if first_token_time else 0
        
        # Calculate Tokens Per Second (TPS) ignoring the TTFT phase
        if elapsed_since_first > 0 and i > 0:
            tps = i / elapsed_since_first
        else:
            tps = 0.0
            
        # Update progress bar with CPU, RAM, TTFT, and TPS usage
        mem_info = process.memory_info()
        ram_mb = mem_info.rss / (1024 * 1024)
        cpu_percent = process.cpu_percent()
        
        pbar.set_postfix({
            'RAM (MB)': f'{ram_mb:.1f}', 
            'CPU (%)': f'{cpu_percent:.1f}',
            'TTFT (s)': f'{ttft:.2f}',
            'TPS': f'{tps:.2f}'
        })

    end_time = time.time()
    total_time = end_time - start_time
    avg_tps = max_output_len / total_time
    
    print(f"\n{Colors.GREEN}Stats for sequence length {max_output_len}:{Colors.ENDC}")
    print(f"{Colors.GREEN}Total Time: {total_time:.2f}s | TTFT: {ttft:.2f}s | Overall Avg Tokens/s: {avg_tps:.2f}{Colors.ENDC}")

    # print(f"--- Final Generated Text (Truncated) ---")
    final_text = tokenizer.decode(current_input_tokens[0])
    # Print the last 150 characters to avoid flooding the terminal
    # print("..." + final_text[-150:])
    
    # Save analytics
    analytics_results.append({
        'length': max_output_len,
        'ttft': ttft,
        'avg_tps': avg_tps,
        'total_time': total_time,
        'final_ram': ram_mb
    })

# Print Final Analytics Summary Table
print(f"\n{Colors.BLUE}{Colors.BOLD}" + "="*60)
print(" FINAL INFERENCE ANALYTICS SUMMARY ".center(60, "="))
print("="*60 + f"{Colors.ENDC}")
print(f"{Colors.WARNING}{'Seq Length':<12} | {'TTFT (s)':<10} | {'Avg TPS':<10} | {'Total Time (s)':<15} | {'Peak RAM (MB)':<15}{Colors.ENDC}")
print("-" * 60)
for res in analytics_results:
    print(f"{res['length']:<12} | {res['ttft']:<10.3f} | {res['avg_tps']:<10.2f} | {res['total_time']:<15.2f} | {res['final_ram']:<15.1f}")
print(f"{Colors.BLUE}{Colors.BOLD}" + "="*60 + f"{Colors.ENDC}\n")