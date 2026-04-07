from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from src.config import PROCESSED_DIR, RANDOM_SEED, SPLIT_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def encode_windows(windowed_dataset: List[Dict], vocab: Dict[str, int]) -> List[Dict]:
    encoded = []

    for item in windowed_dataset:
        token_ids = [vocab[token] for token in item["tokens"] if token in vocab]

        encoded.append({
            "file_path": item["file_path"],
            "window_index": item["window_index"],
            "token_ids": token_ids,
        })

    return encoded


def split_dataset(encoded_dataset: List[Dict]) -> tuple[List[Dict], List[Dict], List[Dict]]:
    random.seed(RANDOM_SEED)
    items = encoded_dataset[:]
    random.shuffle(items)

    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_data = items[:n_train]
    val_data = items[n_train:n_train + n_val]
    test_data = items[n_train + n_val:]

    return train_data, val_data, test_data


if __name__ == "__main__":
    assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-8, "Split ratios must sum to 1."

    windowed_path = PROCESSED_DIR / "windowed_dataset_debug.json"
    vocab_path = PROCESSED_DIR / "vocab_debug.json"

    windowed_dataset = load_json(windowed_path)
    vocab = load_json(vocab_path)

    encoded_dataset = encode_windows(windowed_dataset, vocab)
    train_data, val_data, test_data = split_dataset(encoded_dataset)

    save_json(encoded_dataset, SPLIT_DIR / "encoded_dataset_debug.json")
    save_json(train_data, SPLIT_DIR / "train_debug.json")
    save_json(val_data, SPLIT_DIR / "val_debug.json")
    save_json(test_data, SPLIT_DIR / "test_debug.json")

    print(f"Loaded windows: {len(windowed_dataset)}")
    print(f"Encoded windows: {len(encoded_dataset)}")
    print(f"Vocab size: {len(vocab)}")
    print()
    print(f"Train size: {len(train_data)}")
    print(f"Val size:   {len(val_data)}")
    print(f"Test size:  {len(test_data)}")

    if train_data:
        print("\nExample encoded sample:")
        print(f"File: {train_data[0]['file_path']}")
        print(f"Window index: {train_data[0]['window_index']}")
        print(f"First 20 token ids: {train_data[0]['token_ids'][:20]}")
        print(f"Sequence length: {len(train_data[0]['token_ids'])}")