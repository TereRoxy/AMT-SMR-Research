# preprocessing.py
import os
import h5py
import numpy as np
from config import FRAME_SIZE, HOP_SIZE, SAMPLE_RATE, FFT_CACHE_DIR

def int16_to_float32(x):
    return x / 32767.0

def cache_fft_for_file(h5_path, filename, cache_dir=FFT_CACHE_DIR):
    print(f"Caching FFT for file: {filename}")
    os.makedirs(cache_dir, exist_ok=True)
    fft_path = os.path.join(cache_dir, f"{filename}_fft.npy")

    if os.path.exists(fft_path):
        print(f"FFT cache exists for {filename}, skipping.")
        return fft_path

    with h5py.File(h5_path, "r") as f:
        audio = int16_to_float32(f[f"{filename}/audio"][...])

    n_frames = (len(audio) - FRAME_SIZE) // HOP_SIZE + 1
    fft_frames = []
    window = np.hanning(FRAME_SIZE)   # ← CRITICAL: ADD THIS

    for i in range(n_frames):
        start = i * HOP_SIZE
        end = start + FRAME_SIZE
        frame = audio[start:end]
        if len(frame) < FRAME_SIZE:
            frame = np.pad(frame, (0, FRAME_SIZE - len(frame)))

        frame = frame * window        # ← APPLY WINDOW HERE
        fft_frame = np.fft.rfft(frame).astype(np.complex64)
        fft_frames.append(fft_frame)

    fft_frames = np.array(fft_frames)
    np.save(fft_path, fft_frames)
    print(f"Cached {n_frames} WINDOWED FFTs → {fft_path}")
    return fft_path