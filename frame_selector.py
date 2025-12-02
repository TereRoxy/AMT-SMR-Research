# frame_selector.py — run once
import h5py
import numpy as np
from config import HOP_SIZE, FRAME_SIZE, SAMPLE_RATE
from preprocessing import int16_to_float32

FILENAME = "MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--2"
H5_PATH = "./maestro_h5/2017/2017.h5"

with h5py.File(H5_PATH, "r") as f:
    audio = int16_to_float32(f[f"{FILENAME}/audio"][:])

energies = []
for i in range(0, len(audio)-FRAME_SIZE, HOP_SIZE*4):  # every ~100ms
    frame = audio[i:i+FRAME_SIZE]
    energy = np.sum(frame**2)
    energies.append((i // HOP_SIZE, energy))

# Top 40 loudest frames
best_frames = sorted(energies, key=lambda x: x[1], reverse=True)[:40]
frame_indices = [idx for idx, energy in best_frames if energy > 0.5]  # filter near-silence

print("SELECTED 30+ HIGH-ENERGY FRAMES:")
print(frame_indices[:35])
# → Save this list — use it forever