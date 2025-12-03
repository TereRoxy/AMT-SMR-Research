"""
synthetic_data.py
Utilities to generate and save small synthetic datasets (frames + ground truth)
so you can reuse across experiments or include as a "dataset" for case study.
"""
import os
import numpy as np
from synthesizer import synthesize_pitch_set, FRAME_SIZE, NormalizationMode, SAMPLE_RATE

def generate_synthetic_frames(n_frames=200, out_dir="synthetic_dataset", min_pitch=36, max_pitch=96, notes_range=(2,6), snr_db=30):
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    gts = []
    for i in range(n_frames):
        n = np.random.randint(notes_range[0], notes_range[1]+1)
        gt = set(np.random.randint(min_pitch, max_pitch+1, size=n))
        wave = synthesize_pitch_set(gt, norm_mode=NormalizationMode.PEAK)
        # add noise
        sig_power = np.mean(wave**2)
        snr_lin = 10 ** (snr_db / 10.0)
        noise_power = sig_power / (snr_lin + 1e-12)
        noise = np.random.normal(0, np.sqrt(noise_power), size=wave.shape)
        wave_noisy = wave + noise
        frames.append(wave_noisy * np.hanning(len(wave_noisy)))
        gts.append(gt)
    np.save(os.path.join(out_dir, "frames.npy"), np.array(frames))
    np.save(os.path.join(out_dir, "gts.npy"), np.array(gts, dtype=object))
    print(f"Saved {n_frames} frames to {out_dir}")
    return out_dir
