from pathlib import Path
import pretty_midi

from src.config import GENERATED_MIDI_DIR


def inspect_midi(path: Path, max_notes: int = 20):
    midi = pretty_midi.PrettyMIDI(str(path))
    print(f"File: {path}")
    print(f"End time: {midi.get_end_time():.3f}s")
    print(f"Num instruments: {len(midi.instruments)}")

    total_notes = sum(len(inst.notes) for inst in midi.instruments)
    print(f"Total notes: {total_notes}")

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