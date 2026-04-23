from __future__ import annotations

import argparse
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

    For the current event grammar, both window_size and stride should be multiples of 4:
    [TIME_SHIFT, NOTE_ON, DURATION, VELOCITY]
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if window_size % 4 != 0 or stride % 4 != 0:
        raise ValueError("window_size and stride must be multiples of 4.")

    windows = []

    for i in range(0, len(tokens) - window_size + 1, stride):
        window = tokens[i:i + window_size]
        if len(window) == window_size:
            windows.append(window)

    return windows


def build_windowed_dataset(dataset: List[Dict], window_size: int, stride: int) -> List[Dict]:
    """
    For each tokenized MIDI file, split tokens into fixed-length overlapping windows.
    """
    all_windows: List[Dict] = []

    for item in dataset:
        file_path = item["file_path"]
        tokens = item["tokens"]
        metadata = item.get("metadata", {})

        windows = tokens_to_windows(tokens, window_size, stride)

        for window_idx, window in enumerate(windows):
            all_windows.append({
                "file_path": file_path,
                "window_index": window_idx,
                "start_token_index": window_idx * stride,
                "tokens": window,
                "metadata": metadata,
            })

    return all_windows


def parse_args():
    parser = argparse.ArgumentParser(description="Create overlapping token windows from a tokenized MIDI shard.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input tokenized dataset JSON filename inside data/processed/",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional custom output filename. Defaults to windowed_<input filename>.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Window size in tokens.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=WINDOW_STRIDE,
        help="Stride between consecutive windows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset_path = PROCESSED_DIR / args.input
    if not dataset_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {dataset_path}")

    dataset = load_json(dataset_path)
    windowed_dataset = build_windowed_dataset(
        dataset=dataset,
        window_size=args.window_size,
        stride=args.stride,
    )

    if args.output_name:
        output_path = PROCESSED_DIR / args.output_name
    else:
        output_path = PROCESSED_DIR / f"windowed_{dataset_path.name}"

    save_json(windowed_dataset, output_path)

    print(f"Loaded tokenized dataset from: {dataset_path}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Total windows created: {len(windowed_dataset)}")
    print(f"Saved windowed dataset to: {output_path}")

    if windowed_dataset:
        print("\nExample window:")
        print(f"File: {windowed_dataset[0]['file_path']}")
        print(f"Window index: {windowed_dataset[0]['window_index']}")
        print(f"Start token index: {windowed_dataset[0]['start_token_index']}")
        print(f"First 20 tokens: {windowed_dataset[0]['tokens'][:20]}")