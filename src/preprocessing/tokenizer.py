from pathlib import Path
from typing import Dict, List

from src.config import (
    MAX_DURATION_STEPS,
    MAX_TIME_SHIFT_STEPS,
    SPECIAL_TOKENS,
    STEPS_PER_BEAT,
    VELOCITY_BINS,
)
from src.preprocessing.midi_parser import (
    extract_note_events,
    find_midi_files,
    load_midi_file,
)


TIME_STEP = 0.125
MAX_TIME_SHIFT = 32
MAX_DURATION = 32
VELOCITY_BINS = [16, 32, 48, 64, 80, 96, 112, 127]


def quantize_beats(value_beats: float) -> int:
    return max(0, int(round(value_beats * STEPS_PER_BEAT)))


def quantize_velocity(velocity: int) -> int:
    for i, threshold in enumerate(VELOCITY_BINS):
        if velocity <= threshold:
            return i
    return len(VELOCITY_BINS) - 1


def clamp(value: int, max_value: int) -> int:
    return min(value, max_value)


def note_events_to_tokens(events: List[Dict]) -> List[str]:
    tokens: List[str] = []
    prev_start_beat = 0.0

    for event in events:
        time_shift_steps = quantize_beats(event["start_beat"] - prev_start_beat)
        duration_steps = quantize_beats(event["duration_beats"])
        velocity_bin = quantize_velocity(event["velocity"])

        time_shift_steps = clamp(time_shift_steps, MAX_TIME_SHIFT_STEPS)
        duration_steps = max(1, clamp(duration_steps, MAX_DURATION_STEPS))

        tokens.append(f"TIME_SHIFT_{time_shift_steps}")
        tokens.append(f"NOTE_ON_{event['pitch']}")
        tokens.append(f"DURATION_{duration_steps}")
        tokens.append(f"VELOCITY_{velocity_bin}")

        prev_start_beat = event["start_beat"]

    return tokens

def build_fixed_vocab() -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    idx = 0

    for token in SPECIAL_TOKENS:
        vocab[token] = idx
        idx += 1

    for step in range(MAX_TIME_SHIFT_STEPS + 1):
        vocab[f"TIME_SHIFT_{step}"] = idx
        idx += 1

    for pitch in range(128):
        vocab[f"NOTE_ON_{pitch}"] = idx
        idx += 1

    for duration in range(1, MAX_DURATION_STEPS + 1):
        vocab[f"DURATION_{duration}"] = idx
        idx += 1

    for vel_bin in range(len(VELOCITY_BINS)):
        vocab[f"VELOCITY_{vel_bin}"] = idx
        idx += 1

    return vocab


def tokenize_midi_file(file_path: Path) -> List[str]:
    midi_obj = load_midi_file(file_path)
    if midi_obj is None:
        return []

    events = extract_note_events(midi_obj)
    if not events:
        return []

    return note_events_to_tokens(events)


if __name__ == "__main__":
    files = find_midi_files()

    if not files:
        print("No MIDI files found.")
        raise SystemExit

    sample_file = files[0]
    print(f"Testing tokenizer on:\n{sample_file}\n")

    tokens = tokenize_midi_file(sample_file)

    print(f"Total tokens: {len(tokens)}")
    print("\nFirst 40 tokens:")
    for i, token in enumerate(tokens[:40], start=1):
        print(f"{i:02d}: {token}")