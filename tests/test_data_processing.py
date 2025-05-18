import re
from pathlib import Path

import pytest

from src import data_processing as dp


def test_clean_text():
    dirty = "Hello\n\nWorld\t!"
    assert dp.clean_text(dirty) == "Hello World !"


def test_split_dataset(tmp_path):
    text = "Sentence one. Sentence two? Sentence three!"
    train, val = dp.split_dataset(text, train_ratio=0.5)
    assert len(train) + len(val) == 3
    assert abs(len(train) - len(val)) <= 1


def test_export_jsonl(tmp_path):
    lines = ["a", "b"]
    p = tmp_path / "out.jsonl"
    dp.export_jsonl(lines, p)
    data = p.read_text().strip().splitlines()
    assert data[0] == '{"text": "a"}'
