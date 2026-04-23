from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import (
    PROCESSED_DIR,
    SEQUENCE_LENGTH,
    SPLIT_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    UNK_TOKEN,
    VAL_RATIO,
    VOCAB_PATH,
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
            "metadata": item.get("metadata", {}),
        })

    return encoded


def stable_unit_float_from_string(text: str) -> float:
    """
    Deterministically map a string to a float in [0, 1).
    This lets us assign train/val/test splits independently per shard
    without loading the entire dataset at once.
    """
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16)
    return value / 0xFFFFFFFF


def assign_split(file_path: str) -> str:
    """
    Deterministically assign a file to train/val/test using configured ratios.
    """
    p = stable_unit_float_from_string(file_path)

    if p < TRAIN_RATIO:
        return "train"
    elif p < TRAIN_RATIO + VAL_RATIO:
        return "val"
    else:
        return "test"


def split_dataset_by_file(encoded_dataset: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    train_data = []
    val_data = []
    test_data = []

    for item in encoded_dataset:
        split_name = assign_split(item["file_path"])

        if split_name == "train":
            train_data.append(item)
        elif split_name == "val":
            val_data.append(item)
        else:
            test_data.append(item)

    return train_data, val_data, test_data


def summarize_split(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> None:
    train_files = {item["file_path"] for item in train_data}
    val_files = {item["file_path"] for item in val_data}
    test_files = {item["file_path"] for item in test_data}

    print("Unique files per split in this shard:")
    print(f"  Train files: {len(train_files)}")
    print(f"  Val files:   {len(val_files)}")
    print(f"  Test files:  {len(test_files)}")

    print("\nWindow counts per split in this shard:")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Encode token windows and split them into train/val/test shards.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input windowed dataset JSON filename inside data/processed/",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=VOCAB_PATH.name,
        help="Vocab JSON filename inside data/processed/.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional custom prefix for output shard names.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-8, "Split ratios must sum to 1."

    args = parse_args()

    windowed_path = PROCESSED_DIR / args.input
    vocab_path = PROCESSED_DIR / args.vocab

    if not windowed_path.exists():
        raise FileNotFoundError(f"Windowed dataset not found: {windowed_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")

    windowed_dataset = load_json(windowed_path)
    vocab = load_json(vocab_path)

    encoded_dataset = encode_windows(windowed_dataset, vocab)
    train_data, val_data, test_data = split_dataset_by_file(encoded_dataset)

    if args.prefix:
        prefix = args.prefix
    else:
        stem = Path(args.input).stem
        if stem.startswith("windowed_"):
            stem = stem[len("windowed_"):]
        prefix = stem

    save_json(encoded_dataset, SPLIT_DIR / f"encoded_{prefix}.json")
    save_json(train_data, SPLIT_DIR / f"train_{prefix}.json")
    save_json(val_data, SPLIT_DIR / f"val_{prefix}.json")
    save_json(test_data, SPLIT_DIR / f"test_{prefix}.json")

    print(f"Loaded windows: {len(windowed_dataset)}")
    print(f"Encoded windows: {len(encoded_dataset)}")
    print(f"Vocab size: {len(vocab)}\n")

    summarize_split(train_data, val_data, test_data)

    if train_data:
        print("\nExample encoded sample:")
        print(f"File: {train_data[0]['file_path']}")
        print(f"Window index: {train_data[0]['window_index']}")
        print(f"Start token index: {train_data[0]['start_token_index']}")
        print(f"First 20 token ids: {train_data[0]['token_ids'][:20]}")
        print(f"Sequence length: {len(train_data[0]['token_ids'])}")
        