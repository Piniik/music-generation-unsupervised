import json
from pathlib import Path

import torch

from src.config import (
    CHECKPOINT_DIR,
    EMBED_DIM,
    HIDDEN_DIM,
    LATENT_DIM,
    NUM_LAYERS,
    DROPOUT,
    MAX_GENERATION_LENGTH,
)
from src.models.vae import MusicVAE


def load_vocab(vocab_path: Path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {idx: token for token, idx in vocab.items()}
    return vocab, id_to_token


def get_valid_token_ids(vocab, step_type: str):
    valid_ids = []

    for token, idx in vocab.items():
        if step_type == "TIME_SHIFT" and token.startswith("TIME_SHIFT_"):
            valid_ids.append(idx)
        elif step_type == "NOTE_ON" and token.startswith("NOTE_ON_"):
            valid_ids.append(idx)
        elif step_type == "DURATION" and token.startswith("DURATION_"):
            valid_ids.append(idx)
        elif step_type == "VELOCITY" and token.startswith("VELOCITY_"):
            valid_ids.append(idx)

    return valid_ids


@torch.no_grad()
def sample_next_token_constrained(
    model,
    current_token,
    hidden,
    cell,
    valid_token_ids,
    temperature=1.0,
):
    logits, hidden, cell = model.decode_step(current_token, hidden, cell)
    logits = logits / temperature

    mask = torch.full_like(logits, float("-inf"))
    mask[:, valid_token_ids] = 0.0
    masked_logits = logits + mask

    probs = torch.softmax(masked_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

    return next_token, hidden, cell


@torch.no_grad()
def sample_from_latent_constrained(
    model,
    vocab,
    id_to_token,
    device,
    max_length=128,
    temperature=1.0,
):
    bos_id = vocab["<BOS>"]

    valid_time_shift_ids = get_valid_token_ids(vocab, "TIME_SHIFT")
    valid_note_on_ids = get_valid_token_ids(vocab, "NOTE_ON")
    valid_duration_ids = get_valid_token_ids(vocab, "DURATION")
    valid_velocity_ids = get_valid_token_ids(vocab, "VELOCITY")

    z = torch.randn(1, model.latent_dim, device=device)
    hidden, cell = model.init_decoder_state(z)

    current_token = torch.tensor([bos_id], dtype=torch.long, device=device)

    generated_ids = []

    pattern = [
        ("TIME_SHIFT", valid_time_shift_ids),
        ("NOTE_ON", valid_note_on_ids),
        ("DURATION", valid_duration_ids),
        ("VELOCITY", valid_velocity_ids),
    ]

    for step in range(max_length):
        _, valid_ids = pattern[step % 4]

        next_token, hidden, cell = sample_next_token_constrained(
            model=model,
            current_token=current_token,
            hidden=hidden,
            cell=cell,
            valid_token_ids=valid_ids,
            temperature=temperature,
        )

        token_id = int(next_token.item())
        generated_ids.append(token_id)
        current_token = next_token

    generated_tokens = [id_to_token[idx] for idx in generated_ids]
    return generated_tokens


def main():
    vocab_path = Path("data/processed/vocab_debug.json")
    checkpoint_path = CHECKPOINT_DIR / "vae_debug_best.pt"

    vocab, id_to_token = load_vocab(vocab_path)
    vocab_size = len(vocab)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = MusicVAE(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}\n")

    for i in range(3):
        tokens = sample_from_latent_constrained(
            model=model,
            vocab=vocab,
            id_to_token=id_to_token,
            device=device,
            max_length=MAX_GENERATION_LENGTH,
            temperature=1.0,
        )

        print(f"Sample {i + 1}:")
        print(tokens[:50])
        print()


if __name__ == "__main__":
    main()