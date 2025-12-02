# pitch_detect.py
import numpy as np
from config import SAMPLE_RATE, FRAME_SIZE

def freq_to_midi(freq: float) -> int:
    if freq <= 0:
        return -1
    return int(round(69 + 12 * np.log2(freq / 440.0)))

def detect_pitches_from_frame(mixed_fft: np.ndarray, max_pitches: int = 12, thresh_rel: float = 0.02):
    """
    Robust peak picking with parabolic interpolation.
    Used in SMR crossover.
    """
    mag = np.abs(mixed_fft)
    if np.max(mag) == 0:
        return set()

    mag = mag / (np.max(mag) + 1e-8)

    peaks = []
    for i in range(2, len(mag) - 1):
        if mag[i] > mag[i-1] and mag[i] > mag[i+1] and mag[i] > thresh_rel:
            # Parabolic interpolation
            y1, y2, y3 = mag[i-1], mag[i], mag[i+1]
            if y2 <= 0:
                continue
            p = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3 + 1e-8)
            bin_center = i + p
            freq = bin_center * SAMPLE_RATE / (FRAME_SIZE * 2)

            if 50 < freq < 8000:
                midi = freq_to_midi(freq)
                if 21 <= midi <= 108:
                    peaks.append((mag[i], midi))

    peaks.sort(reverse=True)
    return {midi for _, midi in peaks[:max_pitches]}