import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    BEST_CHECKPOINT_PATH,
    CHECKPOINT_DIR,
    DROPOUT,
    EMBED_DIM,
    HIDDEN_DIM,
    HISTORY_PATH,
    LATENT_DIM,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_LAYERS,
    BETA_START,
    BETA_END,
    BETA_ANNEAL_EPOCHS,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
    TRAIN_PATH,
    USE_AMP,
    VAL_PATH,
    VOCAB_PATH,
    WORD_DROPOUT,
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_word_dropout(decoder_input: torch.Tensor, bos_token_id: int, unk_token_id: int, drop_prob: float):
    if drop_prob <= 0:
        return decoder_input

    dropped = decoder_input.clone()
    mask = (torch.rand_like(decoder_input.float()) < drop_prob)
    mask[:, 0] = False
    dropped[mask] = unk_token_id
    return dropped

class MusicTokenDataset(Dataset):
    def __init__(self, json_path: Path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        for item in self.data:
            if "token_ids" not in item:
                raise ValueError("Missing token_ids in dataset item.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids = self.data[idx]["token_ids"]
        return torch.tensor(token_ids, dtype=torch.long)


def get_beta(epoch: int) -> float:
    """
    Linearly anneal beta from BETA_START to BETA_END and actually hit BETA_END.
    """
    if BETA_ANNEAL_EPOCHS <= 1:
        return BETA_END

    progress = min((epoch - 1) / (BETA_ANNEAL_EPOCHS - 1), 1.0)
    return BETA_START + progress * (BETA_END - BETA_START)


def vae_loss_function(logits, targets, mu, logvar, beta: float):
    """
    logits:   [B, T, V]
    targets:  [B, T]
    mu/logvar: [B, latent_dim]
    """
    recon_loss = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def make_decoder_io(batch: torch.Tensor, bos_token_id: int):
    """
    For a sequence:
      target = [t1, t2, t3, ..., tT]
      input  = [BOS, t1, t2, ..., t(T-1)]

    This prevents trivial full-sequence copying.
    """
    decoder_input = batch.clone()
    decoder_input[:, 1:] = batch[:, :-1]
    decoder_input[:, 0] = bos_token_id

    targets = batch
    return decoder_input, targets


def train_one_epoch(model, dataloader, optimizer, device, bos_token_id: int, unk_token_id: int, beta: float, scaler, amp_enabled: bool):
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        batch = batch.to(device)

        decoder_input, targets = make_decoder_io(batch, bos_token_id)
        decoder_input = apply_word_dropout(decoder_input, bos_token_id, unk_token_id, WORD_DROPOUT)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp_enabled and device == "cuda")):
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


def main():
    from src.models.vae import MusicVAE
    set_seed(RANDOM_SEED)

    train_path = TRAIN_PATH
    val_path = VAL_PATH
    vocab_path = VOCAB_PATH

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    unk_token_id = vocab["<UNK>"]
    vocab_size = len(vocab)
    bos_token_id = vocab["<BOS>"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset = MusicTokenDataset(train_path)
    val_dataset = MusicTokenDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
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

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
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
    }

    amp_enabled = USE_AMP and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    for epoch in range(1, NUM_EPOCHS + 1):
        beta = get_beta(epoch)

        train_loss, train_recon, train_kl = train_one_epoch(
        model, train_loader, optimizer, device, bos_token_id, unk_token_id, beta, scaler, amp_enabled
        )
        
        val_loss, val_recon, val_kl = evaluate(
            model, val_loader, device, bos_token_id, beta
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
                BEST_CHECKPOINT_PATH,
            )
            print(f"Saved best model to {BEST_CHECKPOINT_PATH}")

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved training history to {HISTORY_PATH}")

if __name__ == "__main__":
    main()
    