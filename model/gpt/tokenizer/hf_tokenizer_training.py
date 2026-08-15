# Imports
import os
import glob
import logging
from typing import List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from utils.file_utils import load_yaml

_config = load_yaml("config/config.yml")
DATASET_DIR    = _config.get("dataset_dir", r"corpus\data\pretraining_data")
SAVE_DIR       = _config.get("save_dir", "tokenizer/")
VOCAB_SIZE     = _config.get("vocab_size", 32000)
SPECIAL_TOKENS = _config.get("special_tokens", ["<|endoftext|>", "<|pad|>", "<|unk|>"])

# Set up basic logging for production-grade monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def train_bpe_tokenizer(dataset_dir: str, save_dir: str, vocab_size: int, special_tokens: List[str]) -> Tokenizer:
    """
    Trains a Byte-Pair Encoding (BPE) tokenizer on all text files within a given directory.

    Args:
        dataset_dir (str): Directory containing '.txt' files to train the tokenizer on.
        save_dir (str): Directory where the trained tokenizer should be saved.
        vocab_size (int): The target vocabulary size.
        special_tokens (List[str]): A list of special tokens to add to the vocabulary.

    Returns:
        Tokenizer: The trained HuggingFace Tokenizer instance.
        
    Raises:
        FileNotFoundError: If no text files are found in the specified directory.
    """
    logger.info("Initializing BPE Tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

    # ByteLevel pre-tokenizer is crucial for GPT-style models to handle spacing correctly
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    # Collect all text files in the target directory
    dataset_files = glob.glob(os.path.join(dataset_dir, "*.txt"))
    if not dataset_files:
        raise FileNotFoundError(f"No '.txt' files found in dataset directory: '{dataset_dir}'")

    logger.info(f"Found {len(dataset_files)} file(s) for training. Starting training...")
    tokenizer.train(files=dataset_files, trainer=trainer)
    logger.info("Training completed.")

    # Post-processor to ensure decoded tokens reconstruct the original text precisely
    tokenizer.post_processor = ByteLevelProcessor(
        trim_offsets=True,
        add_prefix_space=False,
    )

    # Decoder to reverse the ByteLevel encoding back into human-readable text
    tokenizer.decoder = ByteLevelDecoder()

    # Ensure the save directory exists and save the models
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.model.save(save_dir)
    
    save_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(save_path)

    logger.info(f"Tokenizer saved to '{save_dir}'")
    logger.info(f"Final vocabulary size: {tokenizer.get_vocab_size()}")
    
    return tokenizer


def load_and_test(save_dir: str) -> None:
    """
    Loads a saved tokenizer and performs a basic encoding/decoding test.

    Args:
        save_dir (str): Directory containing the saved 'tokenizer.json'.
    """
    load_path = os.path.join(save_dir, "tokenizer.json")
    if not os.path.exists(load_path):
        logger.error(f"Tokenizer file not found at {load_path}")
        return

    logger.info("Loading tokenizer for testing...")
    tokenizer = Tokenizer.from_file(load_path)
    
    sample_text = "The transformer is based on the attention mechanism."
    encoded = tokenizer.encode(sample_text)
    
    logger.info(f"Test sentence : {sample_text}")
    logger.info(f"Tokens        : {encoded.tokens}")
    logger.info(f"Token IDs     : {encoded.ids}")
    logger.info(f"Decoded       : {tokenizer.decode(encoded.ids)}")


if __name__ == "__main__":
    try:
        tok = train_bpe_tokenizer(DATASET_DIR, SAVE_DIR, VOCAB_SIZE, SPECIAL_TOKENS)
        load_and_test(SAVE_DIR)
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}", exc_info=True)