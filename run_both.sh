#!/usr/bin/env bash
#  run_both.sh
#  Runs BOTH training scripts (vanilla PyTorch vs HF Accelerate) and
#  prints a wall-clock time comparison at the end.


set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
echo "Working directory: $PROJECT_ROOT"
echo ""

# helpers

section() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# Format seconds as "Xm Ys"
fmt_duration() {
    local secs=$1
    local mins=$(( secs / 60 ))
    local rem=$(( secs % 60 ))
    if (( mins > 0 )); then
        echo "${mins}m ${rem}s"
    else
        echo "${rem}s"
    fi
}

# Step 0 · Install dependencies

section "STEP 0 · Installing dependencies from requirements.txt"
pip install -q -r requirements.txt
echo ""
echo "✓  Dependencies installed"

# Step 1 · Data preparation (shared by both runs)

section "STEP 1/3 · Downloading raw data"
python corpus/download_data/download_tinystories.py
echo ""
echo "✓  Raw + cleaned text ready"

section "STEP 2/3 · Training BPE tokenizer"
python model/gpt/tokenizer/hf_tokenizer_training.py
echo ""
echo "✓  Tokenizer saved"

section "STEP 3/3 · Preparing pretraining dataset (tokenise → .pt tensors)"
python corpus/data/pretraining_data/prepare_dataset.py
echo ""
echo "✓  Tensors saved"

# Run A · Vanilla PyTorch training

section "RUN A · Training GPT-2 with vanilla PyTorch (train_gpt.py)"

start_a=$SECONDS
python model/gpt/train_gpt.py
end_a=$SECONDS
duration_a=$(( end_a - start_a ))

echo ""
echo "✓  Vanilla training finished in $(fmt_duration $duration_a)"

# Run B · HF Accelerate training

section "RUN B · Training GPT-2 with HF Accelerate (train_gpt_accelerate.py)"

start_b=$SECONDS
accelerate launch model/gpt/train_gpt_accelerate.py
end_b=$SECONDS
duration_b=$(( end_b - start_b ))

echo ""
echo "✓  Accelerate training finished in $(fmt_duration $duration_b)"

# Summary

section "⏱  Training Time Comparison"

echo ""
printf "  %-35s %10s\n" "Method" "Time"
printf "  %-35s %10s\n" "-----------------------------------" "----------"
printf "  %-35s %10s\n" "Vanilla PyTorch (train_gpt.py)"    "$(fmt_duration $duration_a)"
printf "  %-35s %10s\n" "HF Accelerate (train_gpt_accelerate.py)" "$(fmt_duration $duration_b)"
echo ""

if (( duration_a > duration_b )); then
    diff_secs=$(( duration_a - duration_b ))
    echo "  Accelerate was FASTER by $(fmt_duration $diff_secs)"
elif (( duration_b > duration_a )); then
    diff_secs=$(( duration_b - duration_a ))
    echo "  Vanilla PyTorch was FASTER by $(fmt_duration $diff_secs)"
else
    echo "  Both methods took the same time"
fi

echo ""
echo "Artifacts produced:"
echo "  • Model weights : artifacts/model/gpt2/gpt2_model.safetensors"
echo ""
