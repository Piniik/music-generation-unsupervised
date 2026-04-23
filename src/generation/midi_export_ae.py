import argparse
import json
from pathlib import Path
from typing import List

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
    MAX_GENERATION_LENGTH,
    NUM_GENERATED_SAMPLES,
    NUM_LAYERS,
    PROCESSED_DIR,
    SAMPLING_TEMPERATURE,
    SPLIT_DIR,
    STEPS_PER_BEAT,
    TOP_K,
    VOCAB_PATH,
    VELOCITY_BINS,
)
from src.models.autoencoder import MusicAutoencoder


def load_vocab(vocab_path: Path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    id_to_token = {idx: token for token, idx in vocab.items()}
    return vocab, id_to_token


def load_dataset(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        masked_logits = torch.where(
            masked_logits < kth,
            torch.full_like(masked_logits, float("-inf")),
            masked_logits,
        )

    probs = torch.softmax(masked_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

    return next_token, hidden, cell


@torch.no_grad()
def sample_from_encoded_input(
    model,
    source_token_ids,
    vocab,
    id_to_token,
    device,
    max_length=128,
    temperature=1.0,
    top_k=16,
):
    bos_id = vocab["<BOS>"]

    valid_time_shift = get_valid_tokens_and_ids(vocab, "TIME_SHIFT")
    valid_note_on = get_valid_tokens_and_ids(vocab, "NOTE_ON")
    valid_duration = get_valid_tokens_and_ids(vocab, "DURATION")
    valid_velocity = get_valid_tokens_and_ids(vocab, "VELOCITY")

    source = torch.tensor(source_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    z = model.encode(source)

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
            top_k=top_k,
        )

        token_id = int(next_token.item())
        generated_ids.append(token_id)
        current_token = next_token

    generated_tokens = [id_to_token[idx] for idx in generated_ids]
    return generated_tokens


def velocity_bin_to_value(bin_idx: int) -> int:
    bin_idx = max(0, min(bin_idx, len(VELOCITY_BINS) - 1))
    return max(1, min(127, int(VELOCITY_BINS[bin_idx])))


def parse_note_groups(tokens: List[str]):
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


def tokens_to_pretty_midi(tokens: List[str]):
    midi = pretty_midi.PrettyMIDI(initial_tempo=DEFAULT_BPM)
    instrument = pretty_midi.Instrument(program=0)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Export AE-generated MIDI files.")
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="ae_debug_best.pt",
        help="Checkpoint filename inside checkpoints/.",
    )
    parser.add_argument(
        "--vocab-name",
        type=str,
        default=VOCAB_PATH.name,
        help="Vocab filename inside data/processed/.",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default="train_debug.json",
        help="Input encoded dataset filename inside data/train_test_split/.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_GENERATED_SAMPLES,
        help="Number of MIDI files to generate.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_GENERATION_LENGTH,
        help="Maximum generated token length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=SAMPLING_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Top-k filtering.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="ae_sample",
        help="Filename prefix for generated MIDIs.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index into the input dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    vocab_path = PROCESSED_DIR / args.vocab_name
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint_name
    input_path = SPLIT_DIR / args.input_json

    vocab, id_to_token = load_vocab(vocab_path)
    input_data = load_dataset(input_path)

    vocab_size = len(vocab)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = MusicAutoencoder(
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
    model.eval()

    GENERATED_MIDI_DIR.mkdir(parents=True, exist_ok=True)

    start = max(0, args.start_index)
    end = min(start + args.num_samples, len(input_data))

    if start >= len(input_data):
        raise ValueError(f"start-index {start} is out of range for dataset of size {len(input_data)}")

    for out_idx, item_idx in enumerate(range(start, end), start=1):
        source_token_ids = input_data[item_idx]["token_ids"]

        tokens = sample_from_encoded_input(
            model=model,
            source_token_ids=source_token_ids,
            vocab=vocab,
            id_to_token=id_to_token,
            device=device,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        midi_obj = tokens_to_pretty_midi(tokens)
        output_path = GENERATED_MIDI_DIR / f"{args.output_prefix}_{out_idx}.mid"
        save_midi(midi_obj, output_path)

        print(f"Saved: {output_path} (source idx {item_idx})")
        print(f"First 20 tokens: {tokens[:20]}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()