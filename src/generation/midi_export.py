import json
from pathlib import Path
from typing import List
from xml.parsers.expat import model

import pretty_midi
import torch

from src.config import (
    BEST_CHECKPOINT_PATH,
    DEFAULT_BPM,
    DROPOUT,
    EMBED_DIM,
    GENERATED_MIDI_DIR,
    HIDDEN_DIM,
    LATENT_DIM,
    MAX_GENERATION_LENGTH,
    NUM_GENERATED_SAMPLES,
    NUM_LAYERS,
    SAMPLING_TEMPERATURE,
    STEPS_PER_BEAT,
    TOP_K,
    VOCAB_PATH,
    VELOCITY_BINS,
)
from src.models.vae import MusicVAE


def load_vocab(vocab_path: Path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {idx: token for token, idx in vocab.items()}
    return vocab, id_to_token


def get_valid_tokens_and_ids(vocab, step_type: str):
    valid = []

    for token, idx in vocab.items():
        if step_type == "TIME_SHIFT" and token.startswith("TIME_SHIFT_"):
            valid.append((token, idx))
        elif step_type == "NOTE_ON" and token.startswith("NOTE_ON_"):
            valid.append((token, idx))
        elif step_type == "DURATION" and token.startswith("DURATION_"):
            valid.append((token, idx))
        elif step_type == "VELOCITY" and token.startswith("VELOCITY_"):
            valid.append((token, idx))

    return valid


@torch.no_grad()
def sample_next_token_constrained(
    model,
    current_token,
    hidden,
    cell,
    z,
    valid_token_pairs,
    temperature=1.0,
    top_k=16,
):
    logits, hidden, cell = model.decode_step(current_token, hidden, cell, z)
    logits = logits / temperature

    valid_ids = [idx for _, idx in valid_token_pairs]

    mask = torch.full_like(logits, float("-inf"))
    mask[:, valid_ids] = 0.0
    masked_logits = logits + mask

    if top_k is not None and top_k > 0:
        top_values, _ = torch.topk(masked_logits, k=min(top_k, masked_logits.size(-1)), dim=-1)
        kth = top_values[:, -1].unsqueeze(1)
        masked_logits = torch.where(masked_logits < kth, torch.full_like(masked_logits, float("-inf")), masked_logits)

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

    valid_time_shift = get_valid_tokens_and_ids(vocab, "TIME_SHIFT")
    valid_note_on = get_valid_tokens_and_ids(vocab, "NOTE_ON")
    valid_duration = get_valid_tokens_and_ids(vocab, "DURATION")
    valid_velocity = get_valid_tokens_and_ids(vocab, "VELOCITY")

    z = torch.randn(1, model.latent_dim, device=device)
    hidden, cell = model.init_decoder_state(z)

    current_token = torch.tensor([bos_id], dtype=torch.long, device=device)
    generated_ids = []

    pattern = [
        valid_time_shift,
        valid_note_on,
        valid_duration,
        valid_velocity,
    ]

    for step in range(max_length):
        valid_pairs = pattern[step % 4]

        next_token, hidden, cell = sample_next_token_constrained(
            model=model,
            current_token=current_token,
            hidden=hidden,
            cell=cell,
            z=z,
            valid_token_pairs=valid_pairs,
            temperature=temperature,
            top_k=TOP_K,
        )

        token_id = int(next_token.item())
        generated_ids.append(token_id)
        current_token = next_token

    generated_tokens = [id_to_token[idx] for idx in generated_ids]
    return generated_tokens


def velocity_bin_to_value(bin_idx: int) -> int:
    bin_idx = max(0, min(bin_idx, len(VELOCITY_BINS) - 1))
    return max(1, min(127, int(VELOCITY_BINS[bin_idx])))


def parse_note_groups(tokens):
    """
    Convert token groups into note tuples:
    (start_time_seconds, pitch, duration_seconds, velocity)
    """
    notes = []
    current_time = 0.0
    seconds_per_step = (60.0 / DEFAULT_BPM) / STEPS_PER_BEAT
    seen = set()

    usable_length = len(tokens) - (len(tokens) % 4)

    for i in range(0, usable_length, 4):
        t_tok = tokens[i]
        n_tok = tokens[i + 1]
        d_tok = tokens[i + 2]
        v_tok = tokens[i + 3]

        if not (
            t_tok.startswith("TIME_SHIFT_")
            and n_tok.startswith("NOTE_ON_")
            and d_tok.startswith("DURATION_")
            and v_tok.startswith("VELOCITY_")
        ):
            continue

        time_shift = int(t_tok.split("_")[-1])
        pitch = int(n_tok.split("_")[-1])
        duration_steps = int(d_tok.split("_")[-1])
        velocity_bin = int(v_tok.split("_")[-1])

        current_time += time_shift * seconds_per_step
        duration = max(seconds_per_step, duration_steps * seconds_per_step)
        velocity = velocity_bin_to_value(velocity_bin)

        pitch = max(0, min(127, int(pitch)))
        velocity = max(1, min(127, int(velocity)))

        start_time = float(current_time)
        end_time = float(start_time + duration)

        key = (round(start_time, 4), pitch, round(end_time, 4))
        if key in seen:
            continue
        seen.add(key)

        notes.append((start_time, pitch, end_time - start_time, velocity))

    return notes


def tokens_to_pretty_midi(tokens):
    midi = pretty_midi.PrettyMIDI(initial_tempo=DEFAULT_BPM)
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    note_tuples = parse_note_groups(tokens)
    note_tuples = sorted(note_tuples, key=lambda x: (x[0], x[1]))

    for start_time, pitch, duration, velocity in note_tuples:
        end_time = start_time + duration
        note = pretty_midi.Note(
            velocity=int(velocity),
            pitch=int(pitch),
            start=float(start_time),
            end=float(end_time),
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi


def save_midi(midi_obj: pretty_midi.PrettyMIDI, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_obj.write(str(output_path))


def main():
    vocab_path = VOCAB_PATH
    checkpoint_path = BEST_CHECKPOINT_PATH

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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    GENERATED_MIDI_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(NUM_GENERATED_SAMPLES):
        tokens = sample_from_latent_constrained(
            model=model,
            vocab=vocab,
            id_to_token=id_to_token,
            device=device,
            max_length=MAX_GENERATION_LENGTH,
            temperature=SAMPLING_TEMPERATURE,
        )

        midi_obj = tokens_to_pretty_midi(tokens)
        output_path = GENERATED_MIDI_DIR / f"debug_sample_{i+1}.mid"
        save_midi(midi_obj, output_path)

        print(f"Saved: {output_path}")
        print(f"First 20 tokens: {tokens[:20]}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()