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
