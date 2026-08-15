import torch
from safetensors.torch import load_file

d = load_file(r'e:\llm_pretraining_accelerator_best\downloaded_hf_model\model.safetensors')
for k in d.keys():
    if "attention" in k:
        print(f"{k}: {d[k].shape}")
