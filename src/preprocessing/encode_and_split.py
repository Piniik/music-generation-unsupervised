from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import (
    PROCESSED_DIR,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    SPLIT_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    UNK_TOKEN,
    VAL_RATIO,
)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def encode_windows(windowed_dataset: List[Dict], vocab: Dict[str, int]) -> List[Dict]:
    encoded = []
    unk_id = vocab[UNK_TOKEN]

    for item in windowed_dataset:
        token_ids = [vocab.get(token, unk_id) for token in item["tokens"]]

        if len(token_ids) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Expected sequence length {SEQUENCE_LENGTH}, got {len(token_ids)} "
                f"for {item['file_path']} window {item['window_index']}"
            )

        encoded.append({
            "file_path": item["file_path"],
            "window_index": item["window_index"],
            "start_token_index": item.get("start_token_index", 0),
            "token_ids": token_ids,
        })

    return encoded


def split_file_paths(file_paths: List[str]) -> Tuple[set, set, set]:
    random.seed(RANDOM_SEED)

    unique_files = sorted(set(file_paths))
    random.shuffle(unique_files)

    n = len(unique_files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_files = set(unique_files[:n_train])
    val_files = set(unique_files[n_train:n_train + n_val])
    test_files = set(unique_files[n_train + n_val:])

    return train_files, val_files, test_files


def split_dataset_by_file(encoded_dataset: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    all_file_paths = [item["file_path"] for item in encoded_dataset]
    train_files, val_files, test_files = split_file_paths(all_file_paths)

    train_data = []
    val_data = []
    test_data = []

    for item in encoded_dataset:
        file_path = item["file_path"]

        if file_path in train_files:
            train_data.append(item)
        elif file_path in val_files:
            val_data.append(item)
        elif file_path in test_files:
            test_data.append(item)
        else:
            raise ValueError(f"File path not assigned to any split: {file_path}")

    return train_data, val_data, test_data


def summarize_split(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> None:
    train_files = {item["file_path"] for item in train_data}
    val_files = {item["file_path"] for item in val_data}
    test_files = {item["file_path"] for item in test_data}

    print("Unique files per split:")
    print(f"  Train files: {len(train_files)}")
    print(f"  Val files:   {len(val_files)}")
    print(f"  Test files:  {len(test_files)}")

    print("\nWindow counts per split:")
    print(f"  Train windows: {len(train_data)}")
    print(f"  Val windows:   {len(val_data)}")
    print(f"  Test windows:  {len(test_data)}")

    overlap_train_val = train_files.intersection(val_files)
    overlap_train_test = train_files.intersection(test_files)
    overlap_val_test = val_files.intersection(test_files)

    print("\nFile overlap check:")
    print(f"  Train ∩ Val:  {len(overlap_train_val)}")
    print(f"  Train ∩ Test: {len(overlap_train_test)}")
    print(f"  Val ∩ Test:   {len(overlap_val_test)}")


if __name__ == "__main__":
    assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-8, "Split ratios must sum to 1."

    windowed_path = PROCESSED_DIR / "windowed_dataset_debug.json"
    vocab_path = PROCESSED_DIR / "vocab_debug.json"

    windowed_dataset = load_json(windowed_path)
    vocab = load_json(vocab_path)

    encoded_dataset = encode_windows(windowed_dataset, vocab)
    train_data, val_data, test_data = split_dataset_by_file(encoded_dataset)

    save_json(encoded_dataset, SPLIT_DIR / "encoded_dataset_debug.json")
    save_json(train_data, SPLIT_DIR / "train_debug.json")
    save_json(val_data, SPLIT_DIR / "val_debug.json")
    save_json(test_data, SPLIT_DIR / "test_debug.json")

    print(f"Loaded windows: {len(windowed_dataset)}")
    print(f"Encoded windows: {len(encoded_dataset)}")
    print(f"Vocab size: {len(vocab)}\n")

    summarize_split(train_data, val_data, test_data)

    if train_data:
        print("\nExample encoded sample:")
        print(f"File: {train_data[0]['file_path']}")
        print(f"Window index: {train_data[0]['window_index']}")
        print(f"First 20 token ids: {train_data[0]['token_ids'][:20]}")
        print(f"Sequence length: {len(train_data[0]['token_ids'])}")