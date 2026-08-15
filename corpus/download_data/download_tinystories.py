import logging
import sys
import os
from pathlib import Path

# Add project root to sys.path so we can import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from utils.file_utils import load_yaml
_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yml"))
config = load_yaml(_config_path)

try:
    from datasets import load_dataset
except ImportError:
    print("[ERROR] 'datasets' library not found. Run: pip install datasets")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

number_of_stories_rows = config['download_data']['number_of_stories_rows']

def main() -> None:
    BASE_DIR  = Path(__file__).resolve().parent
    DATA_DIR  = BASE_DIR.parent / "data" / "pretraining_data"
    CLEAN_DIR = DATA_DIR / "clean"
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    output_path = CLEAN_DIR / "tinystories_clean.txt"
    
    # Skip if already downloaded
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info(f"Already exists, skipping: {output_path.name}")
        return

    logger.info(f"Loading roneneldan/TinyStories dataset (first {number_of_stories_rows:,} stories)...")
    try:
        # 'datasets' library doesn't parse float percentages like 0.05% well. 
        # Using an absolute slice is safer.
        ds = load_dataset("roneneldan/TinyStories", split=f"train[:{number_of_stories_rows}]")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    logger.info(f"Saving to {output_path}...")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            count = 0
            for row in ds:
                text = row.get("text", "").strip()
                if text:
                    # Write each story separated by two newlines
                    f.write(text + "\n\n")
                    count += 1
        logger.info(f"Saved {count:,} stories successfully to {output_path.name}")
        
        # Delete HF cache to keep only the txt file, as requested
        logger.info("Cleaning up Hugging Face cache files for this dataset...")
        cache_cleared = ds.cleanup_cache_files()
        logger.info(f"Cleanup complete. Cleared {cache_cleared} cache files.")

    except IOError as e:
        logger.error(f"Failed to write file: {e}")


if __name__ == "__main__":
    main()
