from pathlib import Path
import pretty_midi

from src.config import GENERATED_MIDI_DIR
from collections import Counter

def inspect_midi(path: Path, max_notes: int = 20):
    midi = pretty_midi.PrettyMIDI(str(path))
    print(f"File: {path}")
    print(f"End time: {midi.get_end_time():.3f}s")
    print(f"Num instruments: {len(midi.instruments)}")

    all_notes = []
    for inst in midi.instruments:
        all_notes.extend(inst.notes)

    total_notes = len(all_notes)
    print(f"Total notes: {total_notes}")

    if total_notes > 0:
        pitches = [n.pitch for n in all_notes]
        durations = [n.end - n.start for n in all_notes]

        print(f"Pitch range: {min(pitches)} - {max(pitches)}")
        print(f"Avg duration: {sum(durations) / len(durations):.3f}s")
        print(f"Min duration: {min(durations):.3f}s")
        print(f"Max duration: {max(durations):.3f}s")

        event_counter = Counter(
            (round(n.start, 4), n.pitch, round(n.end, 4)) for n in all_notes
        )
        duplicates = sum(c - 1 for c in event_counter.values() if c > 1)
        print(f"Duplicate note events: {duplicates}")

    for i, inst in enumerate(midi.instruments):
        print(f"\nInstrument {i}: program={inst.program}, is_drum={inst.is_drum}, notes={len(inst.notes)}")
        for j, note in enumerate(inst.notes[:max_notes], start=1):
            print(
                f"  {j:02d}: pitch={note.pitch}, vel={note.velocity}, "
                f"start={note.start:.3f}, end={note.end:.3f}, dur={note.end-note.start:.3f}"
            )

if __name__ == "__main__":
    for i in range(1, 4):
        path = GENERATED_MIDI_DIR / f"debug_sample_{i}.mid"
        if path.exists():
            inspect_midi(path)
            print("\n" + "=" * 60 + "\n")
        else:
            print(f"Missing: {path}")