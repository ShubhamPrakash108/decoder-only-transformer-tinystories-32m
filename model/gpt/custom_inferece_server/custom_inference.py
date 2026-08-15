import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from pathlib import Path
from threading import Thread

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    logging,
)
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.file_utils import load_yaml
config = load_yaml('config/config.yml')

warnings.filterwarnings("ignore")
# logging.set_verbosity_error()
# logging.disable_progress_bar()

model_id = config["Inference"]["hf_model_link"]
model_directory = Path.cwd() / "downloaded_hf_model"

snapshot_download(
    repo_id=model_id,
    local_dir=model_directory,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_directory,
    trust_remote_code=True,
    local_files_only=True,
)

# 1. Copy the file
source_file = r"e:\llm_pretraining_accelerator_best\model\gpt\custom_inferece_server\stages\0_mha_fix.py"
target_file = r"e:\llm_pretraining_accelerator_best\downloaded_hf_model\modeling_gpt_custom.py"

os.makedirs(os.path.dirname(target_file), exist_ok=True)
shutil.copy(source_file, target_file)
print(f"Copied {source_file} to {target_file}")

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