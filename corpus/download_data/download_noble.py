import logging
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' library not found. Run: pip install requests")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# HTTP headers — required by Gutenberg; plain urllib gets blocked/throttled
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; LLMCorpusBuilder/1.0; "
        "+https://github.com/your-repo) Python-requests"
    )
}

# Constants — exactly 25 modern novels (early–mid 20th century, clean contemporary prose)
NOVELS = {
    # F. Scott Fitzgerald — Jazz Age / Lost Generation
    "the_great_gatsby":          "https://www.gutenberg.org/cache/epub/64317/pg64317.txt",
    "this_side_of_paradise":     "https://www.gutenberg.org/cache/epub/805/pg805.txt",
    "the_beautiful_and_damned":  "https://www.gutenberg.org/cache/epub/9830/pg9830.txt",

    # Sinclair Lewis — sharp American social satire
    "main_street":               "https://www.gutenberg.org/cache/epub/543/pg543.txt",
    "babbitt":                   "https://www.gutenberg.org/cache/epub/1156/pg1156.txt",

    # Edith Wharton — modern psychological realism
    "the_age_of_innocence":      "https://www.gutenberg.org/cache/epub/541/pg541.txt",
    "the_house_of_mirth":        "https://www.gutenberg.org/cache/epub/284/pg284.txt",
    "ethan_frome":               "https://www.gutenberg.org/cache/epub/4517/pg4517.txt",

    # D.H. Lawrence — raw modern sensibility
    "sons_and_lovers":           "https://www.gutenberg.org/cache/epub/59/pg59.txt",
    "the_rainbow":               "https://www.gutenberg.org/cache/epub/28948/pg28948.txt",

    # Joseph Conrad — psychological depth, modern narration
    "heart_of_darkness":         "https://www.gutenberg.org/cache/epub/219/pg219.txt",
    "lord_jim":                  "https://www.gutenberg.org/cache/epub/5658/pg5658.txt",
    "the_secret_agent":          "https://www.gutenberg.org/cache/epub/974/pg974.txt",

    # Virginia Woolf — modernist stream-of-consciousness
    "mrs_dalloway":              "https://www.gutenberg.org/cache/epub/63107/pg63107.txt",
    "a_room_with_a_view":        "https://www.gutenberg.org/cache/epub/2641/pg2641.txt",

    # Jack London — modern adventure + social commentary
    "martin_eden":               "https://www.gutenberg.org/cache/epub/1056/pg1056.txt",
    "the_sea_wolf":              "https://www.gutenberg.org/cache/epub/1074/pg1074.txt",

    # Willa Cather — clean modern American frontier voice
    "my_antonia":                "https://www.gutenberg.org/cache/epub/242/pg242.txt",
    "o_pioneers":                "https://www.gutenberg.org/cache/epub/24/pg24.txt",

    # Upton Sinclair — gritty social realism
    "the_jungle":                "https://www.gutenberg.org/cache/epub/140/pg140.txt",

    # H.G. Wells — conversational modern social fiction
    "tono_bungay":               "https://www.gutenberg.org/cache/epub/718/pg718.txt",
    "mr_polly":                  "https://www.gutenberg.org/cache/epub/4825/pg4825.txt",

    # Booth Tarkington — Pulitzer Prize, early 20th century American
    "the_magnificent_ambersons": "https://www.gutenberg.org/cache/epub/5765/pg5765.txt",
    "alice_adams":               "https://www.gutenberg.org/cache/epub/9327/pg9327.txt",

    # Theodore Dreiser — American gritty realism
    "sister_carrie":             "https://www.gutenberg.org/cache/epub/233/pg233.txt",
}

START_MARKER = "*** START OF THE PROJECT GUTENBERG"
END_MARKER   = "*** END OF THE PROJECT GUTENBERG"

MAX_RETRIES = 3
TIMEOUT_SEC = 30  # seconds per request attempt


def download_file(url: str, output_path: Path) -> bool:
    """Downloads a file from a URL with retries and proper headers.
    
    Returns True on success, False on failure (does NOT exit the process).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"  Attempt {attempt}/{MAX_RETRIES}: GET {url}")
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SEC, stream=True)
            response.raise_for_status()  # raises for 4xx / 5xx

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"  Download complete → {output_path.name}")
            return True

        except requests.exceptions.ConnectionError as e:
            logger.error(f"  Network unreachable: {e}")
            logger.error("  ⚠  Make sure internet is ENABLED in your Kaggle session:")
            logger.error("     Notebook → Settings → Internet → Turn ON")
            # No point retrying a network-level failure — fail fast
            return False

        except requests.exceptions.Timeout:
            logger.warning(f"  Timeout after {TIMEOUT_SEC}s.")

        except requests.exceptions.HTTPError as e:
            logger.warning(f"  HTTP error: {e}")

        except Exception as e:
            logger.warning(f"  Unexpected error: {e}")

        if attempt < MAX_RETRIES:
            wait = 2 ** attempt  # exponential back-off: 2s, 4s
            logger.info(f"  Retrying in {wait}s...")
            time.sleep(wait)

    logger.error(f"  All {MAX_RETRIES} attempts failed for {url}")
    return False


def clean_gutenberg_text(text: str) -> str:
    """Removes Gutenberg headers/footers and extra whitespace."""
    logger.info("  Cleaning Gutenberg boilerplate...")

    start_idx = text.find(START_MARKER)
    end_idx   = text.find(END_MARKER)

    if start_idx != -1 and end_idx != -1:
        start_idx = text.find('\n', start_idx) + 1
        text = text[start_idx:end_idx]
    else:
        logger.warning("  Gutenberg markers not found; using full text.")

    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def process_corpus(name: str, url: str, raw_path: Path, clean_path: Path) -> bool:
    """Downloads, cleans, and saves one novel. Returns True on success."""
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded
    if clean_path.exists() and clean_path.stat().st_size > 0:
        logger.info(f"  Already exists, skipping: {clean_path.name}")
        return True

    # Download
    ok = download_file(url, raw_path)
    if not ok:
        return False

    # Read
    try:
        with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except IOError as e:
        logger.error(f"  Failed to read raw file: {e}")
        return False

    # Clean
    cleaned_text = clean_gutenberg_text(text)

    # Save
    try:
        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
    except IOError as e:
        logger.error(f"  Failed to save cleaned file: {e}")
        return False

    logger.info(f"  Saved ✓  |  chars: {len(cleaned_text):,}  |  words: {len(cleaned_text.split()):,}")
    return True


def main() -> None:
    BASE_DIR   = Path(__file__).resolve().parent
    DATA_DIR   = BASE_DIR.parent / "data" / "pretraining_data"
    RAW_DIR    = DATA_DIR / "raw"
    CLEAN_DIR  = DATA_DIR / "clean"

    total   = len(NOVELS)
    success = 0
    failed  = []

    for idx, (name, url) in enumerate(NOVELS.items(), start=1):
        logger.info(f"[{idx}/{total}] Processing: {name}")
        ok = process_corpus(name, url, RAW_DIR / f"{name}_raw.txt", CLEAN_DIR / f"{name}_clean.txt")
        if ok:
            success += 1
        else:
            failed.append(name)

    logger.info("=" * 60)
    logger.info(f"Done. {success}/{total} novels downloaded successfully.")
    if failed:
        logger.warning(f"Failed ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()