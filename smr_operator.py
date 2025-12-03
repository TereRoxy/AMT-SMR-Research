# smr_operator.py
import numpy as np
from synthesizer import synthesize_pitch_set
from pitch_detect import detect_pitches_from_frame
from config import FRAME_SIZE

def random_mask(shape, alpha_min=0.25, alpha_max=0.75):
    return np.random.uniform(alpha_min, alpha_max, size=shape)

def smr_crossover(parent1: set, parent2: set, frame_size: int = FRAME_SIZE):
    s1 = np.fft.rfft(synthesize_pitch_set(parent1, frame_size))
    s2 = np.fft.rfft(synthesize_pitch_set(parent2, frame_size))

    # W = random_mask(s1.shape)
    # smixed = W * s1 + (1 - W) * s2

    # Take magnitude from random blend, phase from the louder parent (Wiener-style)
    mag1 = np.abs(S1)
    mag2 = np.abs(S2)
    phase = np.angle(S1 if np.mean(mag1) > np.mean(mag2) else S2)

    mask = np.random.uniform(0.3, 0.7, size=mag1.shape)
    mag_child = mask * mag1 + (1 - mask) * mag2

    child_complex = mag_child * np.exp(1j * phase)
    child_wave = np.fft.irfft(child_complex, n=frame_size)
    child_wave /= np.max(np.abs(child_wave)) + 1e-8
    child_wave *= np.hanning(frame_size)

    # More conservative peak picking for SMR children
    child_set = detect_pitches_from_frame(smixed, max_pitches=10, thresh_rel=0.035)
    return child_set

def reencode_with_hps_from_spectrum(smixed_spec: np.ndarray, max_pitches=10, thresh_rel=0.02):
    from pitch_detect import freq_to_midi
    mag = np.abs(smixed_spec)
    mag = mag / (np.max(mag) + 1e-12)
    # simple HPS
    def compute_hps(m):
        hps = m.copy()
        for f in (2,3):
            dec = m[::f]
            if len(dec) < len(m):
                dec = np.pad(dec, (0, len(m)-len(dec)))
            else:
                dec = dec[:len(m)]
            hps *= dec
        return hps
    hps = compute_hps(mag)
    peaks = []
    for i in range(2, len(hps)-2):
        if hps[i] > hps[i-1] and hps[i] > hps[i+1] and hps[i] > thresh_rel:
            y1, y2, y3 = hps[i-1], hps[i], hps[i+1]
            denom = (y1 - 2*y2 + y3)
            p = 0.0 if abs(denom) < 1e-12 else 0.5*(y1 - y3)/denom
            bin_center = i + p
            freq = bin_center * SAMPLE_RATE / (FRAME_SIZE * 2)
            if 30 < freq < SAMPLE_RATE / 2 - 1:
                midi = freq_to_midi(freq)
                if 21 <= midi <= 108:
                    peaks.append((hps[i], midi))
    peaks.sort(reverse=True)
    return {m for _, m in peaks[:max_pitches]}
