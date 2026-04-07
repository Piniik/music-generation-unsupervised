from pathlib import Path
from typing import Dict, List, Optional

import pretty_midi

from src.config import RAW_MIDI_DIR


def find_midi_files(root_dir: Path = RAW_MIDI_DIR) -> List[Path]:
    midi_files = list(root_dir.rglob("*.mid")) + list(root_dir.rglob("*.midi"))
    return sorted(midi_files)


def load_midi_file(file_path: Path) -> Optional[pretty_midi.PrettyMIDI]:
    try:
        return pretty_midi.PrettyMIDI(str(file_path))
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}")
        print(f"        Reason: {e}")
        return None


def summarize_midi(midi_obj: pretty_midi.PrettyMIDI, file_path: Path) -> Dict:
    instruments = midi_obj.instruments
    total_notes = sum(len(instr.notes) for instr in instruments)
    non_drum_instruments = [instr for instr in instruments if not instr.is_drum]
    drum_instruments = [instr for instr in instruments if instr.is_drum]
    duration = float(midi_obj.get_end_time())

    return {
        "file_path": str(file_path),
        "num_instruments": len(instruments),
        "num_non_drum_instruments": len(non_drum_instruments),
        "num_drum_instruments": len(drum_instruments),
        "total_notes": total_notes,
        "duration_seconds": round(duration, 2),
    }


def extract_note_events(midi_obj: pretty_midi.PrettyMIDI) -> List[Dict]:
    """
    Extract all note events from non-drum instruments and sort by start time.
    """
    events: List[Dict] = []

    for instrument_idx, instrument in enumerate(midi_obj.instruments):
        if instrument.is_drum:
            continue

        program_name = pretty_midi.program_to_instrument_name(instrument.program)

        for note in instrument.notes:
            events.append({
                "instrument_idx": instrument_idx,
                "instrument_program": int(instrument.program),
                "instrument_name": program_name,
                "pitch": int(note.pitch),
                "start": float(note.start),
                "end": float(note.end),
                "duration": float(note.end - note.start),
                "velocity": int(note.velocity),
            })

    events.sort(key=lambda x: (x["start"], x["pitch"], x["duration"]))
    return events


if __name__ == "__main__":
    files = find_midi_files()
    print(f"Found {len(files)} MIDI files.\n")

    if not files:
        print("No MIDI files found.")
        raise SystemExit

    sample_file = files[0]
    print(f"Testing sample file:\n{sample_file}\n")

    midi_obj = load_midi_file(sample_file)
    if midi_obj is None:
        raise SystemExit

    summary = summarize_midi(midi_obj, sample_file)
    print("MIDI summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    events = extract_note_events(midi_obj)

    print(f"\nExtracted non-drum note events: {len(events)}")

    print("\nFirst 10 note events:")
    for i, event in enumerate(events[:10], start=1):
        print(f"  Event {i}: {event}")