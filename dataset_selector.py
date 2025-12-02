# dataset_selector.py
import os
import h5py
import random
from note_seq import NoteSequence  # Assuming you have note_seq imported in main.py
from config import MAX_FILES, YEARS, DATASET_DIR

# dataset_selector.py
import os
import h5py
import random
import numpy as np
from note_seq import NoteSequence
from config import SAMPLE_RATE, FRAME_SIZE, HOP_SIZE

# How many random frames to sample per file to estimate polyphony
NUM_SAMPLES_FOR_POLYPHONY_CHECK = 20
_MIN_POLYPHONY_REQUIRED = 2  # at least 2 simultaneous notes in at least one sampled frame


def _estimate_max_polyphony_fast(ns: NoteSequence) -> int:
    """
    Quickly estimate max polyphony by sampling random time windows.
    Avoids building full event timeline.
    """
    if not ns.notes:
        return 0

    duration = ns.total_time
    if duration < FRAME_SIZE / SAMPLE_RATE:
        return 1  # too short

    max_poly = 1
    frame_dur = FRAME_SIZE / SAMPLE_RATE

    for _ in range(NUM_SAMPLES_FOR_POLYPHONY_CHECK):
        # Pick random center time, ensure frame fits
        center_time = random.uniform(frame_dur / 2, duration - frame_dur / 2)
        start_time = center_time - frame_dur / 2
        end_time = center_time + frame_dur / 2

        active = 0
        for note in ns.notes:
            if note.start_time < end_time and note.end_time > start_time:
                active += 1
                if active >= _MIN_POLYPHONY_REQUIRED:
                    return active  # early exit if we already found polyphony
        max_poly = max(max_poly, active)

        # Early exit if we already found clear polyphony
        if max_poly >= 4:
            return max_poly

    return max_poly


def select_small_polyphonic_subset(dataset_dir=DATASET_DIR, years=YEARS, max_files=MAX_FILES):
    """
    Fast polyphonic file selector using sampled frame checks.
    Returns list of (year, h5_full_path, group_name)
    """
    selected = []
    total_selected = 0

    for year in years:
        if total_selected >= max_files:
            break

        year_folder = os.path.join(dataset_dir, year)
        h5_file_path = os.path.join(year_folder, f"{year}.h5")

        if not os.path.isfile(h5_file_path):
            print(f"Warning: File not found: {h5_file_path}")
            continue

        print(f"Scanning {year}.h5 for polyphonic tracks...")

        try:
            with h5py.File(h5_file_path, "r") as f:
                candidates = []

                for group_name in f.keys():
                    group = f[group_name]
                    if "midi" not in group:
                        continue

                    midi_bytes = group["midi"][()]
                    if not midi_bytes:
                        continue

                    try:
                        ns = NoteSequence.FromString(midi_bytes)
                        if len(ns.notes) == 0:
                            continue

                        # Fast polyphony estimation
                        est_poly = _estimate_max_polyphony_fast(ns)
                        if est_poly >= _MIN_POLYPHONY_REQUIRED:
                            candidates.append((group_name, est_poly))

                    except Exception as e:
                        print(f"  Error reading MIDI in {group_name}: {e}")
                        continue

                # Sort by estimated polyphony (prefer highly polyphonic)
                candidates.sort(key=lambda x: x[1], reverse=True)

                # Take top ones
                for group_name, poly in candidates:
                    if total_selected >= max_files:
                        break
                    if len([c for c in selected if c[2] == group_name]) == 0:  # avoid dupes
                        selected.append((year, h5_file_path, group_name))
                        total_selected += 1
                        print(f"  Selected: {year}/{group_name} (est. poly ~{poly})")
                        if total_selected >= max_files:
                            break

                print(f"  Found {len(candidates)} polyphonic tracks in {year}")

        except Exception as e:
            print(f"Error opening {h5_file_path}: {e}")

    print(f"\nSelected {len(selected)} polyphonic files for MVP:")
    for y, p, g in selected:
        print(f"  - {y}/{g}")
    return selected
