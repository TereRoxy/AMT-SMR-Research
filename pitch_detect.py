# pitch_detect.py
import numpy as np
from config import SAMPLE_RATE, FRAME_SIZE

def freq_to_midi(freq: float) -> int:
    if freq <= 0:
        return -1
    return int(round(69 + 12 * np.log2(freq / 440.0)))

def detect_pitches_from_frame(mixed_fft: np.ndarray, max_pitches: int = 12, thresh_rel: float = 0.008):
    """
    Robust piano-tuned peak picking with parabolic interpolation and lower threshold.
    Works on MAESTRO and on perfect synthetic piano spectra.
    """
    mag = np.abs(mixed_fft)
    if mag.max() == 0:
        return set()

    mag = mag / (mag.max() + 1e-12)

    peaks = []
    for i in range(3, len(mag) - 3):   # wider guard
        if mag[i] < thresh_rel:
            continue
        # local maximum
        if mag[i] > mag[i-1] and mag[i] > mag[i+1]:
            # parabolic interpolation
            y1, y2, y3 = mag[i-1], mag[i], mag[i+1]
            delta = 0.5 * (y1 - y3) / (2*y2 - y1 - y3 + 1e-12)
            bin_center = i + delta

            freq = bin_center * SAMPLE_RATE / (FRAME_SIZE * 2)
            if freq < 50 or freq > 8000:
                continue

            midi = freq_to_midi(freq)
            if 21 <= midi <= 108:
                peaks.append((mag[i], midi))

    peaks.sort(reverse=True)
    selected = set()
    for _, midi in peaks[:max_pitches * 2]:   # be generous, then dedup ±1 semitone later if you want
        if not any(abs(midi - m) <= 1 for m in selected):
            selected.add(midi)
        if len(selected) >= max_pitches:
            break

    return selected