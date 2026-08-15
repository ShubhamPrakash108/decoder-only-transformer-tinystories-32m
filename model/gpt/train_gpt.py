import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.file_utils import load_yaml
from model.gpt.model.dataloader import get_dataloaders
from model.gpt.model.gpt2_model import GPTModel, count_parameters

# Load config from config/config.yml
_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yml"))
GPT_CONFIG = load_yaml(_config_path)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build model and move to primary device
model = GPTModel(GPT_CONFIG).to(DEVICE)
total_params = count_parameters(model)
print(f"TOTAL PARAMETERS: {total_params:,}")
print(f"DEVICE: {DEVICE}")

# Multi-GPU: wrap with DataParallel when more than one GPU is available.
# DataParallel splits each batch across all visible GPUs automatically —
# no changes needed inside the training loop.
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
    model = nn.DataParallel(model)

# Training hyper-parameters
EPOCHS_TO_TRAIN_LLM = 1
EARLY_STOPPING_PATIENCE = 3   # stop if val loss doesn't improve for N epochs
LOSS_FN = nn.CrossEntropyLoss()
MAX_GRAD_NORM = 1.0

OPTIMIZER = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
)

# Cosine annealing LR scheduler with warmup via linear phase
SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR(
    OPTIMIZER,
    T_max=EPOCHS_TO_TRAIN_LLM,
    eta_min=1e-5,
)


def train_gpt2(train_loader: DataLoader, val_loader: DataLoader, epochs: int = EPOCHS_TO_TRAIN_LLM):
    """Train the GPT-2 model and run validation after each epoch.

    Saves the checkpoint with the lowest validation loss (best generalisation)
    rather than the final-epoch checkpoint. Also applies early stopping so
    training halts automatically once val loss stops improving.
    """
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Outer bar — one tick per epoch, shows overall training progress.
    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch", file=sys.stdout)

    for epoch in epoch_bar:
        # Training
        model.train()
        train_losses = []

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Batches", leave=False, file=sys.stdout)
        for input_ids, targets in batch_bar:
            input_ids = input_ids.to(DEVICE)
            targets = targets.to(DEVICE)

            OPTIMIZER.zero_grad()
            logits = model(input_ids)
            loss = LOSS_FN(logits.reshape(-1, GPT_CONFIG["vocab_size"]), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            OPTIMIZER.step()
            loss_val = loss.item()
            train_losses.append(loss_val)
            batch_bar.set_postfix(loss=f"{loss_val:.4f}")

        SCHEDULER.step()
        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for input_ids, targets in val_loader:
                input_ids = input_ids.to(DEVICE)
                targets = targets.to(DEVICE)

                logits = model(input_ids)
                loss = LOSS_FN(logits.reshape(-1, GPT_CONFIG["vocab_size"]), targets.reshape(-1))
                val_losses.append(loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        current_lr = OPTIMIZER.param_groups[0]["lr"]

        # Best-checkpoint saving
        # Save only when val loss improves — this is the model that generalises
        # best, not the one that merely memorises the training set the most.
        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            save_model()  # overwrites artifacts/model/gpt2/gpt2_model.safetensors
            best_marker = " ← best val loss — checkpoint saved"
        else:
            epochs_without_improvement += 1
            best_marker = f" (no improvement for {epochs_without_improvement} epoch(s))"

        # Update the outer epoch bar with a summary of this epoch.
        epoch_bar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}",
            val_loss=f"{avg_val_loss:.4f}",
            lr=f"{current_lr:.2e}",
        )
        tqdm.write(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
            f"{best_marker}"
        )

        # Early stopping
        # If val loss hasn't improved for EARLY_STOPPING_PATIENCE epochs,
        # the model is overfitting — stop before it gets worse.
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            tqdm.write(
                f"\nEarly stopping triggered after {epoch + 1} epochs "
                f"(val loss hasn't improved for {EARLY_STOPPING_PATIENCE} epochs)."
            )
            tqdm.write(f"Best val loss: {best_val_loss:.4f}")
            break


def save_model(save_dir: str = "artifacts/model/gpt2"):
    """Save model weights in safetensors format.

    GPT-2 uses weight tying: final_linear_layer.weight shares the same
    underlying tensor as token_embedding.weight. safetensors rejects
    shared-memory tensors, so we drop the duplicate key before saving.
    The tie is restored automatically in load_model.

    When DataParallel is active the real GPTModel lives under model.module —
    we unwrap it so the saved checkpoint is portable and loadable without
    DataParallel.
    """
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", save_dir))
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, "gpt2_model.safetensors")

    # Unwrap DataParallel so we always save a plain GPTModel state dict.
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    # Build a deduplicated state dict — drop the tied output-projection key
    # so only one copy of the shared tensor is written to disk.
    state_dict = {
        k: v for k, v in raw_model.state_dict().items()
        if k != "final_linear_layer.weight"
    }
    save_file(state_dict, filepath)
    print(f"Model saved to: {filepath}")


def load_model(save_dir: str = "artifacts/model/gpt2"):
    """Load model weights from safetensors format.

    After loading, re-tie final_linear_layer.weight → token_embedding.weight
    to restore the weight-sharing that was removed during saving.
    """
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", save_dir))
    filepath = os.path.join(save_path, "gpt2_model.safetensors")
    state_dict = load_file(filepath)
    # strict=False because final_linear_layer.weight is absent from the file
    model.load_state_dict(state_dict, strict=False)
    # Re-tie: point output projection back to the embedding weight
    model.final_linear_layer.weight = model.token_embedding.weight
    model.to(DEVICE)
    print(f"Model loaded from: {filepath}")


def generate(prompt_tokens: list[int], max_new_tokens: int = 50, temperature: float = 1.0, top_k: int = 50) -> list[int]:
    """Generate new tokens given a list of prompt token IDs."""
    model.eval()
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if the sequence gets too long
            input_cropped = input_ids[:, -GPT_CONFIG["max_seq_len"]:]

            logits = model(input_cropped)
            # Take logits for the last token only
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids[0].tolist()


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(GPT_CONFIG)
    train_gpt2(train_loader, val_loader)
    save_model()