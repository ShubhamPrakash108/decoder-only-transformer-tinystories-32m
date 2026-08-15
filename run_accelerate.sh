#!/usr/bin/env bash
# run_accelerate.sh
# Runs HF Accelerate training script.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
echo "Working directory: $PROJECT_ROOT"
echo ""

section() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

section "STEP 0/4 · Installing dependencies from requirements.txt"
pip install -q -r requirements.txt
echo ""
echo "✓  Dependencies installed"

section "STEP 1/4 · Downloading raw data"
python corpus/download_data/download_tinystories.py
echo ""
echo "✓  Raw + cleaned text ready"

section "STEP 2/4 · Training BPE tokenizer"
python model/gpt/tokenizer/hf_tokenizer_training.py
echo ""
echo "✓  Tokenizer saved"

section "STEP 3/4 · Preparing pretraining dataset (tokenise → .pt tensors)"
python corpus/data/pretraining_data/prepare_dataset.py
echo ""
echo "✓  Tensors saved"

section "STEP 4/4 · Training GPT-2 with HF Accelerate (train_gpt_accelerate.py)"
accelerate launch model/gpt/train_gpt_accelerate.py
echo ""
echo "✓  Model saved"

section "Pipeline complete"
echo ""
echo "Artifacts produced:"
echo "  • Cleaned corpus  : corpus/data/pretraining_data/clean/"
echo "  • Tokenizer       : artifacts/tokenizer/v1/tokenizer.json"
echo "  • Dataset tensors : corpus/data/pretraining_data/pt_data_file/pretraining/"
echo "  • Model weights   : artifacts/model/gpt2/model.safetensors"
echo ""
