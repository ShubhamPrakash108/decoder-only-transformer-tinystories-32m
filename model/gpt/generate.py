import argparse
import logging
import os
import sys
from typing import List

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

# Path Setup
# Determine the project root so we can import shared utilities and load config.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.file_utils import load_yaml
from model.gpt.model.gpt2_model import GPTModel

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yml")
CONFIG = load_yaml(CONFIG_PATH)

# Paths derived from project layout & config
TOKENIZER_PATH = os.path.join(ROOT_DIR, CONFIG.get("save_dir", "artifacts/tokenizer/v1"), "tokenizer.json")
MODEL_DIR      = os.path.join(ROOT_DIR, "artifacts", "model", "gpt2")
MODEL_PATH     = os.path.join(MODEL_DIR, "gpt2_model.safetensors")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Tokenizer

def load_tokenizer(tokenizer_path: str = TOKENIZER_PATH) -> Tokenizer:
    """
    Load the pre-trained BPE tokenizer from disk.

    Args:
        tokenizer_path: Absolute or relative path to the ``tokenizer.json`` file.

    Returns:
        A ready-to-use HuggingFace ``Tokenizer`` instance.

    Raises:
        FileNotFoundError: If the tokenizer file does not exist.
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at '{tokenizer_path}'. "
            "Run the tokenizer training script first "
            "(model/gpt/tokenizer/hf_tokenizer_training.py)."
        )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    logger.info("Tokenizer loaded from '%s'  (vocab_size=%d).", tokenizer_path, tokenizer.get_vocab_size())
    return tokenizer


# Model Loading

def load_model(model_path: str = MODEL_PATH, device: torch.device = DEVICE) -> GPTModel:
    """
    Instantiate a GPTModel from config and load pre-trained weights.

    Args:
        model_path: Path to the ``.safetensors`` checkpoint file.
        device:     Target device (``cpu`` or ``cuda``).

    Returns:
        The GPTModel with weights loaded, set to eval mode on the target device.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{model_path}'. "
            "Train the model first (model/gpt/train_gpt.py)."
        )

    model = GPTModel(CONFIG).to(device)
    state_dict = load_file(model_path, device=str(device))
    # strict=False because final_linear_layer.weight is intentionally absent
    # from the checkpoint (weight tying — it shares token_embedding.weight).
    model.load_state_dict(state_dict, strict=False)
    # Re-tie: restore the shared reference that existed before saving.
    model.final_linear_layer.weight = model.token_embedding.weight
    model.eval()

    logger.info("Model loaded from '%s'  (device=%s).", model_path, device)
    return model


# Text Generation    

@torch.no_grad()
def generate(
    model: GPTModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.3,
) -> str:
    """
    Generate text by autoregressively sampling from the model.

    Args:
        model:              The loaded GPTModel in eval mode.
        tokenizer:          The loaded BPE tokenizer.
        prompt:             The text prompt to condition generation on.
        max_new_tokens:     Maximum number of new tokens to generate.
        temperature:        Sampling temperature (lower = more deterministic).
        top_k:              Only sample from the top-k highest-probability tokens.
        repetition_penalty: Penalty > 1.0 discourages repeating tokens that
                            already appear in the context. 1.0 = no penalty.
                            Values around 1.2–1.5 work well in practice.

    Returns:
        The generated text string (prompt + continuation).
    """
    # Encode the prompt into token IDs
    encoded = tokenizer.encode(prompt)
    prompt_ids: List[int] = encoded.ids

    if not prompt_ids:
        logger.warning("Prompt encoded to zero tokens — using <|endoftext|> as seed.")
        eos_id = tokenizer.token_to_id("<|endoftext|>")
        prompt_ids = [eos_id] if eos_id is not None else [0]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    max_seq_len = CONFIG["max_seq_len"]

    # End-of-text token ID for optional early stopping
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")

    for _ in range(max_new_tokens):
        # Crop the context to the maximum sequence length the model supports
        input_cropped = input_ids[:, -max_seq_len:]

        logits = model(input_cropped)

        # Focus on the logits for the very last position in the sequence
        next_token_logits = logits[:, -1, :] / temperature

        # Repetition penalty: divide (positive) or multiply (negative) logits
        # for tokens that already appear in the current context window.
        # This makes the model less likely to repeat itself in a loop.
        if repetition_penalty != 1.0:
            for token_id in set(input_ids[0].tolist()):
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= repetition_penalty
                else:
                    next_token_logits[0, token_id] *= repetition_penalty

        # Top-k filtering: zero out everything outside the top-k
        if top_k > 0:
            top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            next_token_logits = torch.where(
                next_token_logits < min_top_k,
                torch.full_like(next_token_logits, float("-inf")),
                next_token_logits,
            )

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Stop early if <|endoftext|> is generated
        if eos_token_id is not None and next_token.item() == eos_token_id:
            logger.info("Encountered <|endoftext|> — stopping generation.")
            break

    generated_ids = input_ids[0].tolist()
    return tokenizer.decode(generated_ids)


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with a pre-trained GPT-2 model.",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt to condition generation on. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=100,
        help="Maximum number of new tokens to generate (default: 100).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8). Lower values are more deterministic.",
    )
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Top-k filtering threshold (default: 50). Set to 0 to disable.",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.3,
        help="Repetition penalty (default: 1.3). 1.0 = no penalty; higher = less repetition.",
    )
    parser.add_argument(
        "--model_path", type=str, default=MODEL_PATH,
        help=f"Path to the model checkpoint (default: {MODEL_PATH}).",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=TOKENIZER_PATH,
        help=f"Path to the tokenizer JSON (default: {TOKENIZER_PATH}).",
    )
    return parser.parse_args()


def interactive_mode(model: GPTModel, tokenizer: Tokenizer, args: argparse.Namespace) -> None:
    """Run an interactive REPL for text generation."""
    print("\n" + "=" * 60)
    print("  GPT-2 Text Generation — Interactive Mode")
    print("  Type your prompt and press Enter to generate.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not prompt:
            continue

        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"\n{output}\n")


# Entry Point

if __name__ == "__main__":
    args = parse_args()

    # Load tokenizer and model
    tokenizer = load_tokenizer(args.tokenizer_path)
    model = load_model(args.model_path)

    print(f"Device : {DEVICE}")
    print(f"Config : max_seq_len={CONFIG['max_seq_len']}, vocab_size={CONFIG['vocab_size']}")

    if args.prompt is not None:
        # Single-shot mode
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"\n{'─' * 60}")
        print(f"Prompt : {args.prompt}")
        print(f"Output : {output}")
        print(f"{'─' * 60}")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, args)
