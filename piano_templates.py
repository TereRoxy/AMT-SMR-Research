# piano_templates.py
import numpy as np
from config import FRAME_SIZE, SAMPLE_RATE

# Pre-computed average piano harmonic amplitudes (from real MAESTRO stats)
PIANO_HARMONIC_AMP = {
    1: 1.00,
    2: 0.60,
    3: 0.35,
    4: 0.25,
    5: 0.15,
    6: 0.08,
    7: 0.05,
}

def synthesize_with_piano_template(pitch_set: set, frame_size=FRAME_SIZE):
    t = np.arange(frame_size) / SAMPLE_RATE
    y = np.zeros(frame_size, dtype=np.float32)
    window = np.hanning(frame_size)

    for midi in pitch_set:
        f0 = 440 * 2**((midi - 69)/12)
        for h, amp in PIANO_HARMONIC_AMP.items():
            y += amp * np.sin(2 * np.pi * f0 * h * t)

    if np.max(np.abs(y)) > 0:
        y /= np.max(np.abs(y))
    y *= window
    return y