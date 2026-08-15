import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("[ERROR] 'datasets' library not found. Run: pip install datasets")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

MIN_TURNS = 2  # need at least one user + one assistant turn to be useful


def save_jsonl(conversations: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# OpenAssistant OASST1 — real human-written assistant conversations.
# Data is a flat tree (message_id / parent_id); each assistant reply + its
# ancestor chain becomes one training conversation.
# ---------------------------------------------------------------------------
def process_oasst1() -> list[dict]:
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    by_id = {row["message_id"]: row for row in ds if row["lang"] == "en"}

    conversations = []
    for row in by_id.values():
        if row["role"] != "assistant":
            continue
        chain = []
        node = row
        while node is not None:
            chain.append(node)
            parent_id = node["parent_id"]
            node = by_id.get(parent_id) if parent_id else None
        chain.reverse()  # root -> ... -> this assistant reply

        messages = []
        for n in chain:
            role = "user" if n["role"] == "prompter" else "assistant"
            text = n["text"].strip()
            if text:
                messages.append({"role": role, "content": text})

        if len(messages) >= MIN_TURNS:
            conversations.append({"messages": messages})

    return conversations


# ---------------------------------------------------------------------------
# DailyDialog — casual two-person conversations, alternating turns.
# ---------------------------------------------------------------------------
def process_daily_dialog() -> list[dict]:
    ds = load_dataset("daily_dialog", split="train", trust_remote_code=True)

    conversations = []
    for row in ds:
        utterances = [u.strip() for u in row["dialog"] if u.strip()]
        if len(utterances) < MIN_TURNS:
            continue
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": utt}
            for i, utt in enumerate(utterances)
        ]
        conversations.append({"messages": messages})

    return conversations


# ---------------------------------------------------------------------------
# EmpatheticDialogues — supportive exchanges. Rows are individual utterances
# sharing a conv_id; group + order by utterance_idx to rebuild each thread.
# Note: this dataset replaces literal commas with "_comma_" — undo that.
# ---------------------------------------------------------------------------
def process_empathetic_dialogues() -> list[dict]:
    ds = load_dataset("facebook/empathetic_dialogues", split="train", trust_remote_code=True)

    grouped = defaultdict(list)
    for row in ds:
        grouped[row["conv_id"]].append(row)

    conversations = []
    for rows in grouped.values():
        rows.sort(key=lambda r: r["utterance_idx"])
        messages = []
        for i, row in enumerate(rows):
            text = row["utterance"].replace("_comma_", ",").strip()
            if text:
                messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": text})
        if len(messages) >= MIN_TURNS:
            conversations.append({"messages": messages})

    return conversations


# ---------------------------------------------------------------------------
# Cornell Movie-Dialogs — each row is already one exchange; utterance.text
# is the ordered list of lines spoken back and forth.
# ---------------------------------------------------------------------------
def process_cornell_movie() -> list[dict]:
    ds = load_dataset("cornell-movie-dialog/cornell_movie_dialog", split="train", trust_remote_code=True)

    conversations = []
    for row in ds:
        lines = [t.strip() for t in row["utterance"]["text"] if t.strip()]
        if len(lines) < MIN_TURNS:
            continue
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": line}
            for i, line in enumerate(lines)
        ]
        conversations.append({"messages": messages})

    return conversations


SOURCES = {
    "oasst1": process_oasst1,
    "daily_dialog": process_daily_dialog,
    "empathetic_dialogues": process_empathetic_dialogues,
    "cornell_movie_dialog": process_cornell_movie,
}


def main() -> None:
    BASE_DIR  = Path(__file__).resolve().parent
    DATA_DIR  = BASE_DIR.parent / "data" / "sft_data"   # separate from pretraining_data on purpose
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_conversations = []
    summary = {}

    for name, fn in SOURCES.items():
        logger.info(f"Processing: {name}")
        try:
            conversations = fn()
            save_jsonl(conversations, DATA_DIR / f"{name}.jsonl")
            summary[name] = len(conversations)
            all_conversations.extend(conversations)
            logger.info(f"  Saved {len(conversations):,} conversations -> {name}.jsonl")
        except Exception as e:
            logger.error(f"  Failed to process {name}: {e}")
            logger.error(f"  Skipping this source, continuing with the rest...")
            summary[name] = 0

    save_jsonl(all_conversations, DATA_DIR / "combined_dialogue.jsonl")

    logger.info("=" * 60)
    logger.info("Done. Conversation counts:")
    for name, count in summary.items():
        logger.info(f"  {name}: {count:,}")
    logger.info(f"  combined total: {len(all_conversations):,}")
    logger.info(f"Output -> {DATA_DIR}")


if __name__ == "__main__":
    main()