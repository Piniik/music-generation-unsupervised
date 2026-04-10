from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.config import PROCESSED_DIR, SEQUENCE_LENGTH, WINDOW_STRIDE


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def tokens_to_windows(tokens: List[str], window_size: int, stride: int) -> List[List[str]]:
    """
    Split a token sequence into fixed-length overlapping windows.
    window_size and stride should both be multiples of 4 for the current event grammar.
    """
    if window_size % 4 != 0 or stride % 4 != 0:
        raise ValueError("window_size and stride must be multiples of 4.")

    windows = []

    for i in range(0, len(tokens) - window_size + 1, stride):
        window = tokens[i:i + window_size]
        if len(window) == window_size:
            windows.append(window)

    return windows


def build_windowed_dataset(dataset: List[Dict], window_size: int, stride: int) -> List[Dict]:
    all_windows: List[Dict] = []

    for item in dataset:
        file_path = item["file_path"]
        tokens = item["tokens"]

        windows = tokens_to_windows(tokens, window_size, stride)

        for window_idx, window in enumerate(windows):
            all_windows.append({
                "file_path": file_path,
                "window_index": window_idx,
                "start_token_index": window_idx * stride,
                "tokens": window,
            })

    return all_windows


if __name__ == "__main__":
    dataset_path = PROCESSED_DIR / "tokenized_dataset_debug.json"
    output_path = PROCESSED_DIR / "windowed_dataset_debug.json"

    dataset = load_json(dataset_path)
    windowed_dataset = build_windowed_dataset(
    dataset,
    window_size=SEQUENCE_LENGTH,
    stride=WINDOW_STRIDE,
)

    save_json(windowed_dataset, output_path)

    print(f"Loaded tokenized dataset from: {dataset_path}")
    print(f"Window size: {SEQUENCE_LENGTH}")
    print(f"Window stride: {WINDOW_STRIDE}")
    print(f"Total windows created: {len(windowed_dataset)}")
    print(f"Saved windowed dataset to: {output_path}")

    if windowed_dataset:
        print("\nExample window:")
        print(f"File: {windowed_dataset[0]['file_path']}")
        print(f"Window index: {windowed_dataset[0]['window_index']}")
        print(f"First 20 tokens: {windowed_dataset[0]['tokens'][:20]}")