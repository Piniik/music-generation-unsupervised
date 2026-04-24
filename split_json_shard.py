import json
import math
from pathlib import Path

INPUT_FILES = [
    Path("data/train_test_split/train_tokenized_dataset_part_00000_19999.json"),
    Path("data/train_test_split/val_tokenized_dataset_part_00000_19999.json"),
    Path("data/train_test_split/train_tokenized_dataset_part_20000_39999.json"),
    Path("data/train_test_split/val_tokenized_dataset_part_20000_39999.json"),
    Path("data/train_test_split/train_tokenized_dataset_part_40000_59999.json"),
    Path("data/train_test_split/val_tokenized_dataset_part_40000_59999.json"),
    Path("data/train_test_split/train_tokenized_dataset_part_60000_79999.json"),
    Path("data/train_test_split/val_tokenized_dataset_part_60000_79999.json"),
    Path("data/train_test_split/train_tokenized_dataset_part_80000_99999.json"),
    Path("data/train_test_split/val_tokenized_dataset_part_80000_99999.json"),
    Path("data/train_test_split/train_tokenized_dataset_part_100000_116188.json"),
    Path("data/train_test_split/val_tokenized_dataset_part_100000_116188.json"),
]

CHUNK_SIZE = 10000 # number of samples per smaller file


def split_file(path: Path, chunk_size: int):
    if not path.exists():
        print(f"[SKIP] Missing: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    if total == 0:
        print(f"[SKIP] Empty: {path}")
        return

    num_chunks = math.ceil(total / chunk_size)
    stem = path.stem

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        chunk = data[start:end]

        out_path = path.parent / f"{stem}_chunk_{i+1:03d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f)

        print(f"[OK] {out_path.name}  ({start}:{end})")

    print(f"[DONE] Split {path.name} into {num_chunks} chunks")


if __name__ == "__main__":
    for p in INPUT_FILES:
        split_file(p, CHUNK_SIZE)