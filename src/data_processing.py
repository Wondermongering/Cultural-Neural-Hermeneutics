import json
import random
import re
from pathlib import Path
from typing import List, Tuple

try:
    import requests
except ImportError:  # pragma: no cover - requests not installed
    requests = None


def download_text(url: str, dest: Path) -> None:
    """Download text from a URL to a destination file."""
    if requests is None:
        raise ImportError("requests package is required for download_text")
    resp = requests.get(url)
    resp.raise_for_status()
    dest.write_text(resp.text, encoding="utf-8")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_dataset(text: str, train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    random.shuffle(sentences)
    split = int(len(sentences) * train_ratio)
    return sentences[:split], sentences[split:]


def export_jsonl(lines: List[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            json.dump({"text": line}, f)
            f.write("\n")


def export_csv(lines: List[str], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
