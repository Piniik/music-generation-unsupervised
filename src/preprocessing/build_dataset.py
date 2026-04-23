from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.config import MAX_FILES_DEBUG, MIN_NOTES_PER_FILE, PROCESSED_DIR, RANDOM_SEED
from src.preprocessing.midi_parser import find_midi_files, load_midi_file, summarize_midi
from src.preprocessing.tokenizer import build_fixed_vocab, tokenize_midi_file


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_tokenized_dataset(
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    max_files: Optional[int] = None,
    random_subset: bool = False,
) -> List[Dict]:
    midi_files = find_midi_files()
    total_files = len(midi_files)

    if total_files == 0:
        print("No MIDI files found.")
        return []

    if end_idx is None or end_idx > total_files:
        end_idx = total_files

    if start_idx < 0 or start_idx >= total_files:
        raise ValueError(f"start_idx out of range: {start_idx} (total files: {total_files})")

    if end_idx <= start_idx:
        raise ValueError(f"end_idx must be greater than start_idx. Got start={start_idx}, end={end_idx}")

    selected_files = midi_files[start_idx:end_idx]

    if max_files is not None and max_files > 0:
        if random_subset:
            rng = random.Random(RANDOM_SEED)
            if len(selected_files) > max_files:
                selected_files = sorted(rng.sample(selected_files, max_files))
        else:
            selected_files = selected_files[:max_files]

    dataset: List[Dict] = []
    skipped = 0

    for file_path in tqdm(selected_files, desc="Tokenizing MIDI files"):
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

    print(f"\nTotal MIDI files found: {total_files}")
    print(f"Requested slice: [{start_idx}:{end_idx}]")
    print(f"Files selected for this run: {len(selected_files)}")
    print(f"Usable tokenized files: {len(dataset)}")
    print(f"Skipped files: {skipped}")

    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize MIDI files into shardable JSON datasets.")
    parser.add_argument("--start", type=int, default=0, help="Start index in the global MIDI file list.")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive) in the global MIDI file list.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap after slicing. Use for small debug runs.",
    )
    parser.add_argument(
        "--random-subset",
        action="store_true",
        help="If set with --max-files, randomly sample from the selected slice.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional custom output filename. Defaults to tokenized_dataset_part_START_END.json",
    )
    parser.add_argument(
        "--save-vocab",
        action="store_true",
        help="Save deterministic vocab JSON alongside the dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()

    dataset = build_tokenized_dataset(
        start_idx=args.start,
        end_idx=args.end,
        max_files=args.max_files,
        random_subset=args.random_subset,
    )

    if args.output_name:
        dataset_path = PROCESSED_DIR / args.output_name
    else:
        end_str = "END" if args.end is None else str(args.end - 1).zfill(5)
        dataset_path = PROCESSED_DIR / f"tokenized_dataset_part_{str(args.start).zfill(5)}_{end_str}.json"

    save_json(dataset, dataset_path)
    print(f"\nSaved dataset to: {dataset_path}")

    if args.save_vocab:
        vocab = build_fixed_vocab()
        vocab_path = PROCESSED_DIR / "vocab_debug.json"
        save_json(vocab, vocab_path)
        print(f"Saved vocab to:   {vocab_path}")
        print(f"Vocab size: {len(vocab)}")

    if dataset:
        print("\nExample item:")
        print(f"File: {dataset[0]['file_path']}")
        print(f"Num tokens: {dataset[0]['num_tokens']}")
        print(f"First 20 tokens: {dataset[0]['tokens'][:20]}")