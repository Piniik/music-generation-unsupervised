from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.config import PROCESSED_DIR, SEQUENCE_LENGTH


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def tokens_to_windows(tokens: List[str], window_size: int) -> List[List[str]]:
    """
    Split a token sequence into fixed-length non-overlapping windows.
    """
    windows = []

    for i in range(0, len(tokens) - window_size + 1, window_size):
        window = tokens[i:i + window_size]
        if len(window) == window_size:
            windows.append(window)

    return windows


def build_windowed_dataset(dataset: List[Dict], window_size: int) -> List[Dict]:
    """
    For each tokenized MIDI file, split tokens into fixed-length windows.
    """
    all_windows: List[Dict] = []

    for item in dataset:
        file_path = item["file_path"]
        tokens = item["tokens"]

        windows = tokens_to_windows(tokens, window_size)

        for window_idx, window in enumerate(windows):
            all_windows.append({
                "file_path": file_path,
                "window_index": window_idx,
                "tokens": window,
            })

    return all_windows


if __name__ == "__main__":
    dataset_path = PROCESSED_DIR / "tokenized_dataset_debug.json"
    output_path = PROCESSED_DIR / "windowed_dataset_debug.json"

    dataset = load_json(dataset_path)
    windowed_dataset = build_windowed_dataset(dataset, window_size=SEQUENCE_LENGTH)

    save_json(windowed_dataset, output_path)

    print(f"Loaded tokenized dataset from: {dataset_path}")
    print(f"Window size: {SEQUENCE_LENGTH}")
    print(f"Total windows created: {len(windowed_dataset)}")
    print(f"Saved windowed dataset to: {output_path}")

    if windowed_dataset:
        print("\nExample window:")
        print(f"File: {windowed_dataset[0]['file_path']}")
        print(f"Window index: {windowed_dataset[0]['window_index']}")
        print(f"First 20 tokens: {windowed_dataset[0]['tokens'][:20]}")