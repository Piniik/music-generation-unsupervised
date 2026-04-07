import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EMBED_DIM,
    HIDDEN_DIM,
    LATENT_DIM,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_LAYERS,
    DROPOUT,
    SPLIT_DIR,
    BETA_START,
    BETA_END,
    BETA_ANNEAL_EPOCHS,
)


class MusicTokenDataset(Dataset):
    def __init__(self, json_path: Path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids = self.data[idx]["token_ids"]
        return torch.tensor(token_ids, dtype=torch.long)


def get_beta(epoch: int) -> float:
    """
    Linearly anneal beta from BETA_START to BETA_END over BETA_ANNEAL_EPOCHS.
    """
    if epoch >= BETA_ANNEAL_EPOCHS:
        return BETA_END

    progress = epoch / max(1, BETA_ANNEAL_EPOCHS)
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


def train_one_epoch(model, dataloader, optimizer, device, bos_token_id: int, beta: float):
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        batch = batch.to(device)

        decoder_input, targets = make_decoder_io(batch, bos_token_id)

        optimizer.zero_grad()

        logits, mu, logvar = model(batch, decoder_input)
        loss, recon_loss, kl_loss = vae_loss_function(logits, targets, mu, logvar, beta)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

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


def main():
    from src.models.vae import MusicVAE

    train_path = SPLIT_DIR / "train_debug.json"
    val_path = SPLIT_DIR / "val_debug.json"
    vocab_path = Path("data/processed/vocab_debug.json")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab_size = len(vocab)
    bos_token_id = vocab["<BOS>"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset = MusicTokenDataset(train_path)
    val_dataset = MusicTokenDataset(val_path)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

    for epoch in range(1, NUM_EPOCHS + 1):
        beta = get_beta(epoch)

        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, device, bos_token_id, beta
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
            checkpoint_path = CHECKPOINT_DIR / "vae_debug_best.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

    history_path = CHECKPOINT_DIR / "vae_debug_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()