import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    BEST_CHECKPOINT_PATH,
    BETA_ANNEAL_EPOCHS,
    BETA_END,
    BETA_START,
    DROPOUT,
    EMBED_DIM,
    HIDDEN_DIM,
    HISTORY_PATH,
    LATENT_DIM,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_LAYERS,
    RANDOM_SEED,
    SPLIT_DIR,
    USE_AMP,
    VOCAB_PATH,
    WORD_DROPOUT,
)
from src.models.vae import MusicVAE


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


def get_beta(epoch: int) -> float:
    if BETA_ANNEAL_EPOCHS <= 1:
        return BETA_END

    progress = min((epoch - 1) / (BETA_ANNEAL_EPOCHS - 1), 1.0)
    return BETA_START + progress * (BETA_END - BETA_START)


def vae_loss_function(logits, targets, mu, logvar, beta: float):
    recon_loss = nn.CrossEntropyLoss()(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )

    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def make_decoder_io(batch: torch.Tensor, bos_token_id: int):
    decoder_input = batch.clone()
    decoder_input[:, 1:] = batch[:, :-1]
    decoder_input[:, 0] = bos_token_id
    targets = batch
    return decoder_input, targets


def apply_word_dropout(decoder_input: torch.Tensor, bos_token_id: int, unk_token_id: int, drop_prob: float):
    if drop_prob <= 0:
        return decoder_input

    dropped = decoder_input.clone()
    mask = (torch.rand_like(decoder_input.float()) < drop_prob)
    mask[:, 0] = False
    dropped[mask] = unk_token_id
    return dropped


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    bos_token_id: int,
    unk_token_id: int,
    beta: float,
    scaler,
    amp_enabled: bool,
):
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        batch = batch.to(device)

        decoder_input, targets = make_decoder_io(batch, bos_token_id)
        decoder_input = apply_word_dropout(
            decoder_input=decoder_input,
            bos_token_id=bos_token_id,
            unk_token_id=unk_token_id,
            drop_prob=WORD_DROPOUT,
        )

        optimizer.zero_grad()

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(amp_enabled and device == "cuda"),
        ):
            logits, mu, logvar = model(batch, decoder_input)
            loss, recon_loss, kl_loss = vae_loss_function(logits, targets, mu, logvar, beta)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            recon=f"{recon_loss.item():.4f}",
            kl=f"{kl_loss.item():.4f}",
        )

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def evaluate(model, dataloader, device, bos_token_id: int, beta: float):
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    for batch in progress_bar:
        batch = batch.to(device)

        decoder_input, targets = make_decoder_io(batch, bos_token_id)

        logits, mu, logvar = model(batch, decoder_input)
        loss, recon_loss, kl_loss = vae_loss_function(logits, targets, mu, logvar, beta)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            recon=f"{recon_loss.item():.4f}",
            kl=f"{kl_loss.item():.4f}",
        )

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_kl / n


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE from sharded train/val JSON files.")
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
        default=BEST_CHECKPOINT_PATH.name,
        help="Output checkpoint filename inside checkpoints/.",
    )
    parser.add_argument(
        "--history-name",
        type=str,
        default=HISTORY_PATH.name,
        help="Output history filename inside checkpoints/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)

    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    bos_token_id = vocab["<BOS>"]
    unk_token_id = vocab["<UNK>"]

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

    model = MusicVAE(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_path = BEST_CHECKPOINT_PATH.parent / args.checkpoint_name
    history_path = HISTORY_PATH.parent / args.history_name

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    amp_enabled = USE_AMP and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_loss = float("inf")

    history = {
        "epoch": [],
        "beta": [],
        "train_loss": [],
        "train_recon": [],
        "train_kl": [],
        "val_loss": [],
        "val_recon": [],
        "val_kl": [],
        "train_files": [p.name for p in train_files],
        "val_files": [p.name for p in val_files],
    }

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Vocab size:    {vocab_size}")

    for epoch in range(1, NUM_EPOCHS + 1):
        beta = get_beta(epoch)

        train_loss, train_recon, train_kl = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            bos_token_id=bos_token_id,
            unk_token_id=unk_token_id,
            beta=beta,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )

        val_loss, val_recon, val_kl = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            bos_token_id=bos_token_id,
            beta=beta,
        )

        history["epoch"].append(epoch)
        history["beta"].append(beta)
        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)

        print(
            f"Epoch {epoch:02d} | "
            f"Beta: {beta:.4f} | "
            f"Train Loss: {train_loss:.4f} | Train Recon: {train_recon:.4f} | Train KL: {train_kl:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Recon: {val_recon:.4f} | Val KL: {val_kl:.4f}"
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