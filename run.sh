#!/usr/bin/env bash

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

section "STEP 1/4 · Downloading raw novels from Project Gutenberg"

python corpus/download_data/download_tinystories.py

echo ""
echo "✓  Raw + cleaned text saved to corpus/data/pretraining_data/{raw,clean}/"

section "STEP 2/4 · Training BPE tokenizer"

python model/gpt/tokenizer/hf_tokenizer_training.py

echo ""
echo "✓  Tokenizer saved to artifacts/tokenizer/v1/tokenizer.json"

section "STEP 3/4 · Preparing pretraining dataset (tokenise → .pt tensors)"

python corpus/data/pretraining_data/prepare_dataset.py

echo ""
echo "✓  Tensors saved to corpus/data/pretraining_data/pt_data_file/pretraining/"

section "STEP 4/4 · Training GPT-2 model"

python model/gpt/train_gpt.py

echo ""
echo "✓  Model saved to artifacts/model/gpt2/gpt2_model.safetensors"

section "Pipeline complete"
echo ""
echo "Artifacts produced:"
echo "  • Cleaned corpus  : corpus/data/pretraining_data/clean/"
echo "  • Tokenizer       : artifacts/tokenizer/v1/tokenizer.json"
echo "  • Dataset tensors : corpus/data/pretraining_data/pt_data_file/pretraining/"
echo "  • Model weights   : artifacts/model/gpt2/gpt2_model.safetensors"
echo ""
echo "To generate text, run:"
echo "  python model/gpt/generate.py --prompt \"Call me Ishmael\""
echo ""
