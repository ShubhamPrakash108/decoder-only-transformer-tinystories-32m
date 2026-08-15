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
from accelerate import Accelerator

accelerator = Accelerator(step_scheduler_with_optimizer=False)

# Load config from config/config.yml
_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yml"))
GPT_CONFIG = load_yaml(_config_path)

# Device setup
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build model and move to primary device
model = GPTModel(GPT_CONFIG)
total_params = count_parameters(model)
accelerator.print(f"TOTAL PARAMETERS: {total_params:,}")
accelerator.print(f"DEVICE: {accelerator.device}")

if accelerator.num_processes > 1:
    accelerator.print(f"Using {accelerator.num_processes} GPUs ")

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

# learning rate scheduler
SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR(
    OPTIMIZER,
    T_max=EPOCHS_TO_TRAIN_LLM,
    eta_min=1e-5,
)

train_loader, val_loader, test_loader = get_dataloaders(GPT_CONFIG)

# Prepare everything with accelerator (wraps model, optimizer, dataloaders)
gpt_model, OPTIMIZER, train_loader, val_loader, SCHEDULER = accelerator.prepare(
    model, OPTIMIZER, train_loader, val_loader, SCHEDULER
)

def train_gpt2(gpt_model, OPTIMIZER, train_loader: DataLoader, val_loader: DataLoader, SCHEDULER, epochs: int = EPOCHS_TO_TRAIN_LLM):
    """Train the GPT-2 model and run validation after each epoch.

    Saves the checkpoint with the lowest validation loss (best generalisation)
    rather than the final-epoch checkpoint. Also applies early stopping so
    training halts automatically once val loss stops improving.
    """
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    # Outer bar — one tick per epoch, shows overall training progress.
    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch", file=sys.stdout, disable=not accelerator.is_local_main_process)

    for epoch in epoch_bar:
        # Training
        gpt_model.train()
        train_losses = []

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} Batches", leave=False, file=sys.stdout, disable=not accelerator.is_local_main_process)
        for input_ids, targets in batch_bar:

            OPTIMIZER.zero_grad()
            logits = gpt_model(input_ids)
            loss = LOSS_FN(logits.reshape(-1, GPT_CONFIG["vocab_size"]), targets.reshape(-1))
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(gpt_model.parameters(), MAX_GRAD_NORM)
            OPTIMIZER.step()
            
            # Synchronize loss across all processes to get the global average
            gathered_loss = accelerator.gather(loss.detach().unsqueeze(0)).mean().item()
            train_losses.append(gathered_loss)
            batch_bar.set_postfix(loss=f"{gathered_loss:.4f}")
        
        current_lr = OPTIMIZER.param_groups[0]["lr"] # It's safe to get this from any process Otherwise the displayed LR is not the LR used during that epoch.
        SCHEDULER.step()
        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation
        gpt_model.eval()
        val_losses = []
        with torch.no_grad():
            for input_ids, targets in val_loader:

                logits = gpt_model(input_ids)
                loss = LOSS_FN(logits.reshape(-1, GPT_CONFIG["vocab_size"]), targets.reshape(-1))
                
                # Synchronize validation loss as well
                gathered_val_loss = accelerator.gather(loss.detach().unsqueeze(0)).mean().item()
                val_losses.append(gathered_val_loss)

        avg_val_loss = sum(val_losses) / len(val_losses)
        

        # Best-checkpoint saving
        # Save only when val loss improves — this is the model that generalises
        # best, not the one that merely memorises the training set the most.
        improved = avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            save_model_accelerate(gpt_model)  # overwrites artifacts/model/gpt2/model.safetensors
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
        
        if accelerator.is_local_main_process:
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
            if accelerator.is_local_main_process:
                tqdm.write(
                    f"\nEarly stopping triggered after {epoch + 1} epochs "
                    f"(val loss hasn't improved for {EARLY_STOPPING_PATIENCE} epochs)."
                )
                tqdm.write(f"Best val loss: {best_val_loss:.4f}")
            break


def save_model_accelerate(model, save_dir: str = "artifacts/model/gpt2"):
    """Save model weights in safetensors format.

    GPT-2 uses weight tying: final_linear_layer.weight shares the same
    underlying tensor as token_embedding.weight. safetensors rejects
    shared-memory tensors, so we drop the duplicate key before saving.
    The tie is restored automatically in load_model.

    When using Accelerate, the model is wrapped. We use accelerator.unwrap_model()
    so the saved checkpoint is portable and loadable without Accelerate.
    """
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", save_dir))
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, "model.safetensors")

    # wait for everyone
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = unwrapped_model.state_dict()

    if accelerator.is_main_process:
        # Build a deduplicated state dict — drop the tied output-projection key
        # so only one copy of the shared tensor is written to disk.
        state_dict_dedup = {
            k: v.detach().cpu() # This is safer 
            for k, v in state_dict.items()
            if k != "final_linear_layer.weight"
        }
        save_file(state_dict_dedup, filepath)
        print(f"Model saved to: {filepath}")


def load_model_accelerate(model, save_dir: str = "artifacts/model/gpt2"):
    """Load model weights from safetensors format.

    After loading, re-tie final_linear_layer.weight → token_embedding.weight
    to restore the weight-sharing that was removed during saving.
    """
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", save_dir))
    filepath = os.path.join(save_path, "model.safetensors")
    
    # Load the safetensors file
    state_dict = load_file(filepath)
    
    # Unwrap the model to safely load the raw GPT2 state dict
    unwrapped_model = accelerator.unwrap_model(model)
    
    # strict=False because final_linear_layer.weight is absent from the file
    unwrapped_model.load_state_dict(state_dict, strict=False)
    
    # Re-tie: point output projection back to the embedding weight
    unwrapped_model.final_linear_layer.weight = unwrapped_model.token_embedding.weight
    
    print(f"Model loaded from: {filepath}")


if __name__ == "__main__":
    train_gpt2(gpt_model, OPTIMIZER, train_loader, val_loader, SCHEDULER )