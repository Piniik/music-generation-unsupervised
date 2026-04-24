import argparse
import glob
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
    RANDOM_SEED,
    SPLIT_DIR,
    USE_AMP,
    VOCAB_PATH,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_FF_DIM,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_MAX_SEQ_LEN,
)
from src.models.transformer import MusicTransformer


TRANSFORMER_CHECKPOINT_PATH = CHECKPOINT_DIR / "transformer_best.pt"
TRANSFORMER_HISTORY_PATH = CHECKPOINT_DIR / "transformer_history.json"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultiFileMusicTokenDataset(IterableDataset):
    def __init__(self, json_paths, shuffle_files: bool = False, shuffle_within_file: bool = False):
        self.json_paths = list(json_paths)
        self.shuffle_files = shuffle_files
        self.shuffle_within_file = shuffle_within_file

        self.total_items = 0
        for path in self.json_paths:
            with open(path, "r", encoding="utf-8") as f:
                shard_data = json.load(f)

            for item in shard_data:
                if "token_ids" not in item:
                    raise ValueError(f"Missing token_ids in dataset item from {path}")

            self.total_items += len(shard_data)

        if self.total_items == 0:
            raise ValueError("No training samples loaded.")

    def __len__(self):
        return self.total_items

    def __iter__(self):
        file_list = self.json_paths[:]

        if self.shuffle_files:
            random.shuffle(file_list)

        for path in file_list:
            with open(path, "r", encoding="utf-8") as f:
                shard_data = json.load(f)

            if self.shuffle_within_file:
                random.shuffle(shard_data)

            for item in shard_data:
                yield torch.tensor(item["token_ids"], dtype=torch.long)


def resolve_split_files(patterns):
    matched = []

    for pattern in patterns:
        full_pattern = str(SPLIT_DIR / pattern)
        matched.extend(glob.glob(full_pattern))

    matched = sorted(set(matched))
    if not matched:
        raise FileNotFoundError(f"No files matched patterns: {patterns}")

    return [Path(p) for p in matched]


def make_autoregressive_io(batch: torch.Tensor):
    """
    For a sequence [t1, t2, t3, ..., tT]:
      input  = [t1, t2, ..., t(T-1)]
      target = [t2, t3, ..., tT]
    """
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    return inputs, targets


def transformer_loss_function(logits, targets):
    """
    logits:  [B, T, V]
    targets: [B, T]
    """
    return nn.CrossEntropyLoss()(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


def compute_perplexity(loss_value: float) -> float:
    try:
        return math.exp(loss_value)
    except OverflowError:
        return float("inf")


def train_one_epoch(model, dataloader, optimizer, device, scaler, amp_enabled: bool):
    model.train()

    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        batch = batch.to(device)
        inputs, targets = make_autoregressive_io(batch)

        optimizer.zero_grad()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(amp_enabled and device == "cuda"),
        ):
            logits = model(inputs)
            loss = transformer_loss_function(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            ppl=f"{compute_perplexity(loss.item()):.2f}",
        )

    avg_loss = total_loss / len(dataloader)
    avg_ppl = compute_perplexity(avg_loss)
    return avg_loss, avg_ppl


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    for batch in progress_bar:
        batch = batch.to(device)
        inputs, targets = make_autoregressive_io(batch)

        logits = model(inputs)
        loss = transformer_loss_function(logits, targets)

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            ppl=f"{compute_perplexity(loss.item()):.2f}",
        )

    avg_loss = total_loss / len(dataloader)
    avg_ppl = compute_perplexity(avg_loss)
    return avg_loss, avg_ppl


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer from sharded train/val JSON files.")
    parser.add_argument(
        "--train-patterns",
        nargs="+",
        default=["train_*.json"],
        help="Glob patterns inside data/train_test_split/ for train shards.",
    )
    parser.add_argument(
        "--val-patterns",
        nargs="+",
        default=["val_*.json"],
        help="Glob patterns inside data/train_test_split/ for val shards.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=TRANSFORMER_CHECKPOINT_PATH.name,
        help="Output checkpoint filename inside checkpoints/.",
    )
    parser.add_argument(
        "--history-name",
        type=str,
        default=TRANSFORMER_HISTORY_PATH.name,
        help="Output history filename inside checkpoints/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)

    train_files = resolve_split_files(args.train_patterns)
    val_files = resolve_split_files(args.val_patterns)

    print("Train shard files:")
    for p in train_files:
        print(f"  {p.name}")

    print("\nVal shard files:")
    for p in val_files:
        print(f"  {p.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    train_dataset = MultiFileMusicTokenDataset(
        train_files,
        shuffle_files=True,
        shuffle_within_file=True,
    )
    val_dataset = MultiFileMusicTokenDataset(
        val_files,
        shuffle_files=False,
        shuffle_within_file=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = MusicTransformer(
        vocab_size=vocab_size,
        max_seq_len=TRANSFORMER_MAX_SEQ_LEN,
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD,
        num_layers=TRANSFORMER_NUM_LAYERS,
        dim_feedforward=TRANSFORMER_FF_DIM,
        dropout=TRANSFORMER_DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    checkpoint_path = TRANSFORMER_CHECKPOINT_PATH.parent / args.checkpoint_name
    history_path = TRANSFORMER_HISTORY_PATH.parent / args.history_name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    amp_enabled = USE_AMP and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_loss = float("inf")

    history = {
        "epoch": [],
        "train_loss": [],
        "train_perplexity": [],
        "val_loss": [],
        "val_perplexity": [],
        "train_files": [p.name for p in train_files],
        "val_files": [p.name for p in val_files],
    }

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Vocab size:    {vocab_size}")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_ppl = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )

        val_loss, val_ppl = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_perplexity"].append(train_ppl)
        history["val_loss"].append(val_loss)
        history["val_perplexity"].append(val_ppl)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} | "
            f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "vocab_size": vocab_size,
                },
                checkpoint_path,
            )
            print(f"Saved best model to {checkpoint_path}")

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()