from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from src.config import MAX_FILES_DEBUG, MIN_NOTES_PER_FILE, PROCESSED_DIR, RANDOM_SEED
from src.preprocessing.midi_parser import find_midi_files, load_midi_file, summarize_midi
from src.preprocessing.tokenizer import build_fixed_vocab, tokenize_midi_file


def build_tokenized_dataset(max_files: int = MAX_FILES_DEBUG) -> List[Dict]:
    midi_files = find_midi_files()

    rng = random.Random(RANDOM_SEED)
    if max_files and len(midi_files) > max_files:
        midi_files = sorted(rng.sample(midi_files, max_files))

    dataset: List[Dict] = []
    skipped = 0

    for file_path in tqdm(midi_files, desc="Tokenizing MIDI files"):
        try:
            midi_obj = load_midi_file(file_path)
            if midi_obj is None:
                skipped += 1
                continue

            summary = summarize_midi(midi_obj, file_path)
            if summary["total_notes"] < MIN_NOTES_PER_FILE:
                skipped += 1
                continue

            tokens = tokenize_midi_file(file_path)
            if not tokens:
                skipped += 1
                continue

            dataset.append({
                "file_path": str(file_path),
                "tokens": tokens,
                "num_tokens": len(tokens),
                "metadata": summary,
            })
        except Exception as e:
            skipped += 1
            print(f"[WARN] Skipping {file_path} because of error: {e}")

    print(f"\nScanned files: {len(midi_files)}")
    print(f"Usable tokenized files: {len(dataset)}")
    print(f"Skipped files: {skipped}")

    return dataset


def build_vocab(dataset: List[Dict]) -> Dict[str, int]:
    return build_fixed_vocab()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset = build_tokenized_dataset(max_files=10000)
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