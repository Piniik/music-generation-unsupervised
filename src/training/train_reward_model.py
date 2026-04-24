import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import (
    CHECKPOINT_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    REWARD_BATCH_SIZE,
    REWARD_EMBED_DIM,
    REWARD_HIDDEN_DIM,
    REWARD_LEARNING_RATE,
    REWARD_MODEL_CHECKPOINT,
    REWARD_NUM_EPOCHS,
    RLHF_SCORE_CSV,
    VOCAB_PATH,
)
from src.models.reward_model import MusicRewardModel


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_vocab(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class HumanScoreDataset(Dataset):
    def __init__(self, csv_path: Path):
        self.items = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if not row.get("score"):
                    continue

                score = float(row["score"])
                score = max(1.0, min(5.0, score))
                normalized_score = (score - 1.0) / 4.0

                token_path = Path(row["tokens_path"])
                with open(token_path, "r", encoding="utf-8") as tf:
                    token_obj = json.load(tf)

                self.items.append({
                    "token_ids": token_obj["token_ids"],
                    "reward": normalized_score,
                })

        if len(self.items) == 0:
            raise ValueError("No scored samples found. Fill the score column first.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return item["token_ids"], item["reward"]


def collate_batch(batch):
    token_lists, rewards = zip(*batch)

    max_len = max(len(x) for x in token_lists)
    padded = []

    for x in token_lists:
        padded.append(x + [0] * (max_len - len(x)))

    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(rewards, dtype=torch.float32),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train reward model from human scores.")
    parser.add_argument("--scores-csv", type=str, default=str(RLHF_SCORE_CSV))
    parser.add_argument("--checkpoint-name", type=str, default=REWARD_MODEL_CHECKPOINT.name)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)

    vocab = load_vocab(PROCESSED_DIR / VOCAB_PATH.name)
    vocab_size = len(vocab)

    dataset = HumanScoreDataset(Path(args.scores_csv))

    loader = DataLoader(
        dataset,
        batch_size=REWARD_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    print(f"Scored samples: {len(dataset)}")

    model = MusicRewardModel(
        vocab_size=vocab_size,
        embed_dim=REWARD_EMBED_DIM,
        hidden_dim=REWARD_HIDDEN_DIM,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=REWARD_LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint_name

    for epoch in range(1, REWARD_NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for token_ids, rewards in loader:
            token_ids = token_ids.to(device)
            rewards = rewards.to(device)

            pred = model(token_ids)
            loss = loss_fn(pred, rewards)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:02d} | Reward model loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "vocab_size": vocab_size,
                },
                checkpoint_path,
            )
            print(f"Saved reward model to {checkpoint_path}")


if __name__ == "__main__":
    main()