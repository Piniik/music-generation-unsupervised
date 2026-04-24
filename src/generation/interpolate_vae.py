import argparse
import json
from pathlib import Path

import pretty_midi
import torch

from src.config import (
    CHECKPOINT_DIR,
    DEFAULT_BPM,
    DROPOUT,
    EMBED_DIM,
    GENERATED_MIDI_DIR,
    HIDDEN_DIM,
    LATENT_DIM,
    NUM_LAYERS,
    PROCESSED_DIR,
    SAMPLING_TEMPERATURE,
    SPLIT_DIR,
    STEPS_PER_BEAT,
    TOP_K,
    TRANSFORMER_MAX_SEQ_LEN,
    VOCAB_PATH,
    VELOCITY_BINS,
)
from src.models.vae import MusicVAE


def load_vocab(vocab_path: Path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {idx: token for token, idx in vocab.items()}
    return vocab, id_to_token


def load_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_valid_token_ids(vocab, step_type: str):
    valid = []
    for token, idx in vocab.items():
        if step_type == "TIME_SHIFT" and token.startswith("TIME_SHIFT_"):
            valid.append(idx)
        elif step_type == "NOTE_ON" and token.startswith("NOTE_ON_"):
            valid.append(idx)
        elif step_type == "DURATION" and token.startswith("DURATION_"):
            valid.append(idx)
        elif step_type == "VELOCITY" and token.startswith("VELOCITY_"):
            valid.append(idx)
    return valid


@torch.no_grad()
def sample_next_token_constrained(
    model,
    current_token,
    hidden,
    cell,
    z,
    valid_token_ids,
    temperature=1.0,
    top_k=16,
):
    logits, hidden, cell = model.decode_step(current_token, hidden, cell, z)
    logits = logits / temperature

    mask = torch.full_like(logits, float("-inf"))
    mask[:, valid_token_ids] = 0.0
    logits = logits + mask

    if top_k is not None and top_k > 0:
        top_values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        kth = top_values[:, -1].unsqueeze(1)
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

    return next_token, hidden, cell


@torch.no_grad()
def decode_from_z(model, z, vocab, id_to_token, device, max_length, temperature, top_k):
    bos_id = vocab["<BOS>"]

    valid_time_shift_ids = get_valid_token_ids(vocab, "TIME_SHIFT")
    valid_note_on_ids = get_valid_token_ids(vocab, "NOTE_ON")
    valid_duration_ids = get_valid_token_ids(vocab, "DURATION")
    valid_velocity_ids = get_valid_token_ids(vocab, "VELOCITY")

    pattern = [
        valid_time_shift_ids,
        valid_note_on_ids,
        valid_duration_ids,
        valid_velocity_ids,
    ]

    hidden, cell = model.init_decoder_state(z)
    current_token = torch.tensor([bos_id], dtype=torch.long, device=device)

    generated_ids = []

    for step in range(max_length):
        valid_ids = pattern[step % 4]

        next_token, hidden, cell = sample_next_token_constrained(
            model=model,
            current_token=current_token,
            hidden=hidden,
            cell=cell,
            z=z,
            valid_token_ids=valid_ids,
            temperature=temperature,
            top_k=top_k,
        )

        token_id = int(next_token.item())
        generated_ids.append(token_id)
        current_token = next_token

    return [id_to_token[idx] for idx in generated_ids]


def velocity_bin_to_value(bin_idx: int) -> int:
    bin_idx = max(0, min(bin_idx, len(VELOCITY_BINS) - 1))
    return max(1, min(127, int(VELOCITY_BINS[bin_idx])))


def parse_note_groups(tokens):
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
    instrument = pretty_midi.Instrument(program=0)

    for start_time, pitch, duration, velocity in parse_note_groups(tokens):
        note = pretty_midi.Note(
            velocity=int(velocity),
            pitch=int(pitch),
            start=float(start_time),
            end=float(start_time + duration),
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    return midi


def parse_args():
    parser = argparse.ArgumentParser(description="VAE latent interpolation experiment.")
    parser.add_argument("--checkpoint-name", type=str, required=True)
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--idx-a", type=int, default=0)
    parser.add_argument("--idx-b", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=SAMPLING_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--output-prefix", type=str, default="vae_interp")
    return parser.parse_args()


def main():
    args = parse_args()

    vocab_path = PROCESSED_DIR / VOCAB_PATH.name
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint_name
    input_path = SPLIT_DIR / args.input_json

    vocab, id_to_token = load_vocab(vocab_path)
    data = load_dataset(input_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"

    print(f"Using device: {device}")

    model = MusicVAE(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    ids_a = torch.tensor(data[args.idx_a]["token_ids"], dtype=torch.long, device=device).unsqueeze(0)
    ids_b = torch.tensor(data[args.idx_b]["token_ids"], dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        mu_a, _ = model.encode(ids_a)
        mu_b, _ = model.encode(ids_b)

    GENERATED_MIDI_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(args.num_steps):
        alpha = i / max(1, args.num_steps - 1)
        z = (1.0 - alpha) * mu_a + alpha * mu_b

        tokens = decode_from_z(
            model=model,
            z=z,
            vocab=vocab,
            id_to_token=id_to_token,
            device=device,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        midi = tokens_to_pretty_midi(tokens)
        out_path = GENERATED_MIDI_DIR / f"{args.output_prefix}_{i+1}_alpha_{alpha:.2f}.mid"
        midi.write(str(out_path))

        print(f"Saved interpolation step {i+1}/{args.num_steps}: {out_path}")


if __name__ == "__main__":
    main()