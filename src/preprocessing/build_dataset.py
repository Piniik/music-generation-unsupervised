from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from src.config import MAX_FILES_DEBUG, PROCESSED_DIR, SPECIAL_TOKENS
from src.preprocessing.midi_parser import find_midi_files
from src.preprocessing.tokenizer import tokenize_midi_file


def build_tokenized_dataset(max_files: int = MAX_FILES_DEBUG) -> List[Dict]:
    midi_files = find_midi_files()
    selected_files = midi_files[:max_files]

    dataset: List[Dict] = []
    skipped = 0

    for file_path in tqdm(selected_files, desc="Tokenizing MIDI files"):
        try:
            tokens = tokenize_midi_file(file_path)
            if not tokens:
                skipped += 1
                continue

            dataset.append({
                "file_path": str(file_path),
                "tokens": tokens,
                "num_tokens": len(tokens),
            })
        except Exception as e:
            skipped += 1
            print(f"[WARN] Skipping {file_path} because of error: {e}")

    print(f"\nProcessed files: {len(selected_files)}")
    print(f"Usable tokenized files: {len(dataset)}")
    print(f"Skipped files: {skipped}")

    return dataset


def build_vocab(dataset: List[Dict]) -> Dict[str, int]:
    counter = Counter()

    for item in dataset:
        counter.update(item["tokens"])

    vocab = {}
    idx = 0

    for token in SPECIAL_TOKENS:
        vocab[token] = idx
        idx += 1

    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = idx
            idx += 1

    return vocab


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset = build_tokenized_dataset(max_files=1000)
    vocab = build_vocab(dataset)

    dataset_path = PROCESSED_DIR / "tokenized_dataset_debug.json"
    vocab_path = PROCESSED_DIR / "vocab_debug.json"

    save_json(dataset, dataset_path)
    save_json(vocab, vocab_path)

    print(f"\nSaved dataset to: {dataset_path}")
    print(f"Saved vocab to:   {vocab_path}")
    print(f"Vocab size: {len(vocab)}")

    if dataset:
        print("\nExample item:")
        print(f"File: {dataset[0]['file_path']}")
        print(f"Num tokens: {dataset[0]['num_tokens']}")
        print(f"First 20 tokens: {dataset[0]['tokens'][:20]}")