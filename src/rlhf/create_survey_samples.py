import argparse
import csv
import json
from pathlib import Path

import torch

from src.config import (
    CHECKPOINT_DIR,
    GENERATED_MIDI_DIR,
    PROCESSED_DIR,
    RLHF_MAX_NEW_TOKENS,
    RLHF_NUM_SURVEY_SAMPLES,
    RLHF_PRIMER_LENGTH,
    RLHF_SURVEY_MIDI_DIR,
    RLHF_TOKEN_DIR,
    SAMPLING_TEMPERATURE,
    SPLIT_DIR,
    TOP_K,
    TRANSFORMER_D_MODEL,
    TRANSFORMER_DROPOUT,
    TRANSFORMER_FF_DIM,
    TRANSFORMER_MAX_SEQ_LEN,
    TRANSFORMER_NHEAD,
    TRANSFORMER_NUM_LAYERS,
    VOCAB_PATH,
)
from src.generation.generate_transformer import (
    generate_from_primer,
    load_vocab,
    tokens_to_pretty_midi,
    save_midi,
)
from src.models.transformer import MusicTransformer


def load_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokens_to_ids(tokens, vocab):
    return [vocab[t] for t in tokens if t in vocab]


def parse_args():
    parser = argparse.ArgumentParser(description="Create Task 4 human survey MIDI samples.")
    parser.add_argument("--checkpoint-name", type=str, required=True)
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=RLHF_NUM_SURVEY_SAMPLES)
    parser.add_argument("--primer-length", type=int, default=RLHF_PRIMER_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=RLHF_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=SAMPLING_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args()


def main():
    args = parse_args()

    RLHF_SURVEY_MIDI_DIR.mkdir(parents=True, exist_ok=True)
    RLHF_TOKEN_DIR.mkdir(parents=True, exist_ok=True)

    vocab_path = PROCESSED_DIR / VOCAB_PATH.name
    input_path = SPLIT_DIR / args.input_json
    checkpoint_path = CHECKPOINT_DIR / args.checkpoint_name

    vocab, id_to_token = load_vocab(vocab_path)
    data = load_dataset(input_path)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    model = MusicTransformer(
        vocab_size=len(vocab),
        max_seq_len=TRANSFORMER_MAX_SEQ_LEN,
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD,
        num_layers=TRANSFORMER_NUM_LAYERS,
        dim_feedforward=TRANSFORMER_FF_DIM,
        dropout=TRANSFORMER_DROPOUT,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    manifest_path = RLHF_SURVEY_MIDI_DIR / "survey_manifest.csv"

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "midi_path", "tokens_path", "score"],
        )
        writer.writeheader()

        for i in range(args.num_samples):
            source_token_ids = data[i]["token_ids"]
            primer = source_token_ids[: args.primer_length]

            tokens = generate_from_primer(
                model=model,
                primer_token_ids=primer,
                vocab=vocab,
                id_to_token=id_to_token,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            sample_id = f"sample_{i+1:03d}"

            midi = tokens_to_pretty_midi(tokens)
            midi_path = RLHF_SURVEY_MIDI_DIR / f"{sample_id}.mid"
            save_midi(midi, midi_path)

            token_ids = tokens_to_ids(tokens, vocab)
            token_path = RLHF_TOKEN_DIR / f"{sample_id}.json"
            with open(token_path, "w", encoding="utf-8") as tf:
                json.dump({"sample_id": sample_id, "token_ids": token_ids}, tf)

            writer.writerow({
                "sample_id": sample_id,
                "midi_path": str(midi_path),
                "tokens_path": str(token_path),
                "score": "",
            })

            print(f"Saved {sample_id}: {midi_path}")

    print(f"\nSurvey manifest saved to: {manifest_path}")
    print("Fill the score column with human ratings from 1 to 5.")


if __name__ == "__main__":
    main()