from pathlib import Path
from typing import Dict, List

from src.preprocessing.midi_parser import (
    extract_note_events,
    find_midi_files,
    load_midi_file,
)


TIME_STEP = 0.125
MAX_TIME_SHIFT = 32
MAX_DURATION = 32
VELOCITY_BINS = [16, 32, 48, 64, 80, 96, 112, 127]


def quantize_value(value: float, step: float) -> int:
    return max(0, int(round(value / step)))


def quantize_velocity(velocity: int) -> int:
    for i, threshold in enumerate(VELOCITY_BINS):
        if velocity <= threshold:
            return i
    return len(VELOCITY_BINS) - 1


def clamp(value: int, max_value: int) -> int:
    return min(value, max_value)


def note_events_to_tokens(events: List[Dict]) -> List[str]:
    tokens: List[str] = []
    prev_start = 0.0

    for event in events:
        time_shift = quantize_value(event["start"] - prev_start, TIME_STEP)
        duration = quantize_value(event["duration"], TIME_STEP)
        velocity_bin = quantize_velocity(event["velocity"])

        time_shift = clamp(time_shift, MAX_TIME_SHIFT)
        duration = max(1, clamp(duration, MAX_DURATION))

        tokens.append(f"TIME_SHIFT_{time_shift}")
        tokens.append(f"NOTE_ON_{event['pitch']}")
        tokens.append(f"DURATION_{duration}")
        tokens.append(f"VELOCITY_{velocity_bin}")

        prev_start = event["start"]

    return tokens


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