import logging
import os
import sys
from typing import List, Tuple

import torch

# Path Setup
# Insert the project root into sys.path so that the 'utils' package is
# importable regardless of the working directory from which this script runs.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tokenizers import Tokenizer
from utils.file_utils import load_yaml


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration & Tokenizer
config = load_yaml("config/config.yml")

# Load the pre-trained BPE tokenizer from the versioned artifact path.
_tokenizer_path = "artifacts/tokenizer/v1/tokenizer.json"
if not os.path.exists(_tokenizer_path):
    raise FileNotFoundError(
        f"Tokenizer not found at '{_tokenizer_path}'. "
        "Run the tokenizer training script before preparing the dataset."
    )
tokenizer: Tokenizer = Tokenizer.from_file(_tokenizer_path)
logger.info("Tokenizer loaded from '%s'.", _tokenizer_path)

# Corpus Loading
# Read and concatenate every `.txt` file in the dataset directory.
# Files are sorted to ensure a deterministic corpus ordering across runs.
_dataset_dir: str = config["dataset_dir"]

if not os.path.isdir(_dataset_dir):
    raise NotADirectoryError(
        f"Dataset directory '{_dataset_dir}' does not exist. "
        "Check the 'dataset_dir' key in config/config.yml."
    )

_txt_files = sorted(f for f in os.listdir(_dataset_dir) if f.endswith(".txt"))

if not _txt_files:
    raise RuntimeError(
        f"No .txt files found in '{_dataset_dir}'. "
        "Ensure the corpus has been cleaned and placed in the correct directory."
    )

# Tokenization Helpers

def encode_text(text: str) -> List[int]:
    """
    Encode a raw text string into a list of token IDs using the loaded tokenizer.

    Args:
        text: The raw input string to tokenize.

    Returns:
        A list of integer token IDs.
    """
    encoded = tokenizer.encode(text)
    return encoded.ids


def decode_text(ids: List[int]) -> str:
    """
    Decode a list of token IDs back into a human-readable string.

    Args:
        ids: A list of integer token IDs produced by encode_text().

    Returns:
        The decoded string.
    """
    decoded = tokenizer.decode(ids)
    return decoded

# Sequence Block Size
# block_size defines the context window length (number of tokens) used to
# build each training sample.  This must match the model's positional
# embedding dimension configured during architecture design.
block_size: int = config["block_size"]

# Reference: Naive (Inefficient) Dataset Construction — kept for documentation
# The approach below was an earlier, sub-optimal implementation.  It is
# retained as a reference to illustrate *why* the efficient version is
# preferable:
#
# Problems with the naive approach:
#   1. Inner loop: for each context window of length `block_size`, it
#      iterates over every prefix length, producing O(n * block_size) samples
#      instead of O(n).  This is orders of magnitude more data than needed for
#      next-token prediction training.
#   2. Variable-length sequences: X samples have lengths 0 … block_size-1,
#      requiring padding before they can be stacked into a tensor.
#   3. No alignment: y samples are single tokens rather than full shifted
#      sequences, which is incompatible with cross-entropy over a full context.
#
# def create_llm_dataset(text):
#     encoded_texts_ids = encode_text(text)

#     X = []
#     y = []

#     for i in range(len(encoded_texts_ids)-block_size):
#         context_data = encoded_texts_ids[i:i+block_size]
#
#         small_context = 0
#
#         for IDs in range(len(context_data)):
#
#             X.append(context_data[:IDs])
#             y.append(context_data[IDs:IDs+1])
#
#
#     return X, y

# Efficient Dataset Construction
def create_llm_dataset_efficient(encoded_texts_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build sliding-window (input, target) sequence pairs for next-token prediction
    using PyTorch's memory-efficient unfold operation.

    For each position ``idx`` in the token sequence, a sample is formed as:
        X[idx] = tokens[idx : idx + block_size]          (context)
        y[idx] = tokens[idx + 1 : idx + block_size + 1]  (shifted-by-one target)

    This aligns with the standard causal language-model objective where every
    output token at position ``t`` predicts position ``t+1``.

    Args:
        encoded_texts_ids: The full list of tokenized integers.

    Returns:
        A tuple ``(X, y)`` where:
            - X is a torch.Tensor of shape (N, block_size).
            - y is a torch.Tensor of shape (N, block_size),
              shifted one position to the right relative to the corresponding X.

    Raises:
        ValueError: If the encoded corpus is shorter than ``block_size + 1``
                    tokens, making it impossible to form even one sample.
    """

    num_tokens = len(encoded_texts_ids)
    if num_tokens <= block_size:
        raise ValueError(
            f"Corpus contains only {num_tokens} tokens, but block_size={block_size}. "
            "The corpus must have at least block_size + 1 tokens to create any sample."
        )

    # Convert the 1D list of tokens directly into a PyTorch tensor
    tokens_tensor = torch.tensor(encoded_texts_ids, dtype=torch.long)
    
    # Create sliding window views using unfold.
    # .unfold(dimension, size, step)
    X_tensor = tokens_tensor[:-1].unfold(0, block_size, 1)
    y_tensor = tokens_tensor[1:].unfold(0, block_size, 1)

    logger.info(
        "Created %d sliding-window samples (block_size=%d, total tokens=%d).",
        X_tensor.shape[0],
        block_size,
        num_tokens,
    )
    return X_tensor, y_tensor



# Entry Point

if __name__ == "__main__":
    logger.info("Starting dataset preparation. Encoding text chunks to avoid out-of-memory errors...")

    encoded_texts_ids = []
    for fname in _txt_files:
        fpath = os.path.join(_dataset_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                encoded_texts_ids.extend(encode_text(line))
    
    logger.info("Loaded and encoded %d corpus file(s) from '%s'.", len(_txt_files), _dataset_dir)

    # Build the (X, y) dataset from the loaded corpus.
    X_tensor, y_tensor = create_llm_dataset_efficient(encoded_texts_ids)

    # Output directory for serialized PyTorch tensors.
    out_dir = os.path.join("corpus", "data", "pretraining_data", "pt_data_file", "pretraining")
    os.makedirs(out_dir, exist_ok=True)

    # Persist tensors to disk; these files are consumed directly by the
    # training DataLoader via a custom Dataset class.
    X_path = os.path.join(out_dir, "X.pt")
    y_path = os.path.join(out_dir, "y.pt")
    torch.save(X_tensor, X_path)
    torch.save(y_tensor, y_path)

    # Report summary statistics for quick sanity-checking.
    x_size_mb = os.path.getsize(X_path) / (1024 ** 2)
    y_size_mb = os.path.getsize(y_path) / (1024 ** 2)

    logger.info("Dataset saved to '%s'.", out_dir)
    logger.info("  Samples : %d", len(X_tensor))
    logger.info("  X shape : %s  (%.2f MB)", tuple(X_tensor.shape), x_size_mb)
    logger.info("  y shape : %s  (%.2f MB)", tuple(y_tensor.shape), y_size_mb)