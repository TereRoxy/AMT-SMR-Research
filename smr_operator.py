# smr_operator.py
import numpy as np
from synthesizer import synthesize_pitch_set, FRAME_SIZE

def smr_crossover(parent1: set, parent2: set, target_fft: np.ndarray) -> set:
    """
    Oracle-style but realistic: try every note from both parents,
    keep the combination that minimizes spectral residual against the TARGET.
    This is the only spectral crossover that actually helps the GA.
    """
    if not parent1 and not parent2:
        return set()

    candidates = parent1 | parent2
    if len(candidates) <= 6:                     # small → brute force
        best_set = candidates
        best_res = float('inf')
        # try all subsets of size up to len+2
        # from the union
        for mask in range(1 << len(candidates)):  # 2^|union| iterations
            trial = {list(candidates)[i] for i in range(len(candidates)) if (mask & (1<<i))}
            if not trial:
                continue
            wave = synthesize_pitch_set(trial) * np.hanning(FRAME_SIZE)
            res = np.sum(np.abs(np.abs(np.fft.rfft(wave)) - np.abs(target_fft))**2)
            if res < best_res:
                best_res = res
                best_set = trial
        return best_set

    # If union is too big, fall back to greedy pruning/addition
    current = parent1 | parent2
    mag_target = np.abs(target_fft)

    # Greedy removal
    for _ in range(3):
        if not current:
            break
        worst_note = None
        best_red = 0
        for note in current:
            trial = current - {note}
            wave = synthesize_pitch_set(trial) * np.hanning(FRAME_SIZE)
            red = np.sum(np.abs(mag_target - np.abs(np.fft.rfft(wave)))**2)
            if red < best_red or worst_note is None:
                best_red = red
                worst_note = note
        if worst_note is not None:
            current.discard(worst_note)

    # Greedy addition (rarely needed)
    for note in set(range(36, 96)) - current:
        trial = current | {note}
        wave = synthesize_pitch_set(trial) * np.hanning(FRAME_SIZE)
        res = np.sum(np.abs(mag_target - np.abs(np.fft.rfft(wave)))**2)
        if res < best_red:
            current = trial
            best_red = res

    return current
# # smr_operator.py
# import numpy as np
# from synthesizer import synthesize_pitch_set
# from pitch_detect import detect_pitches_from_frame
# from config import FRAME_SIZE, SAMPLE_RATE
#
#
# def random_mask(shape, alpha_min=0.25, alpha_max=0.75):
#     return np.random.uniform(alpha_min, alpha_max, size=shape)
#
# def smr_crossover(parent1: set, parent2: set, frame_size: int = FRAME_SIZE):
#     s1 = np.fft.rfft(synthesize_pitch_set(parent1, frame_size))
#     s2 = np.fft.rfft(synthesize_pitch_set(parent2, frame_size))
#
#     # W = random_mask(s1.shape)
#     # smixed = W * s1 + (1 - W) * s2
#
#     # Take magnitude from random blend, phase from the louder parent (Wiener-style)
#     mag1 = np.abs(s1)
#     mag2 = np.abs(s2)
#     eps = 1e-8
#
#     # Ratio mask (this is the key line!)
#     # mask = mag1 / (mag1 + mag2 + eps)
#
#     # mask = (mag1 > mag2)   # binary mask: take the louder source per bin
#     # # or soft version:
#     mask = mag1 ** 2 / (mag1 ** 2 + mag2 ** 2 + eps)   # Wiener mask
#
#     # Apply mask to the louder parent's phase
#     phase = np.angle(s1 + s2)  # or pick the louder one
#     child_mag = mask * mag1 + (1 - mask) * mag2
#
#     child_complex = child_mag * np.exp(1j * phase)
#     child_wave = np.fft.irfft(child_complex, n=frame_size)
#     child_wave = child_wave / (np.max(np.abs(child_wave)) + 1e-8)
#     child_wave *= np.hanning(frame_size)
#
#     smixed = np.fft.rfft(child_wave)
#
#     # More conservative peak picking for SMR children
#     child_set = detect_pitches_from_frame(smixed, max_pitches=10, thresh_rel=0.015)
#     return child_set
#
# def reencode_with_hps_from_spectrum(smixed_spec: np.ndarray, max_pitches=10, thresh_rel=0.02):
#     from pitch_detect import freq_to_midi
#     mag = np.abs(smixed_spec)
#     mag = mag / (np.max(mag) + 1e-12)
#     # simple HPS
#     def compute_hps(m):
#         hps = m.copy()
#         for f in (2,3):
#             dec = m[::f]
#             if len(dec) < len(m):
#                 dec = np.pad(dec, (0, len(m)-len(dec)))
#             else:
#                 dec = dec[:len(m)]
#             hps *= dec
#         return hps
#     hps = compute_hps(mag)
#     peaks = []
#     for i in range(2, len(hps)-2):
#         if hps[i] > hps[i-1] and hps[i] > hps[i+1] and hps[i] > thresh_rel:
#             y1, y2, y3 = hps[i-1], hps[i], hps[i+1]
#             denom = (y1 - 2*y2 + y3)
#             p = 0.0 if abs(denom) < 1e-12 else 0.5*(y1 - y3)/denom
#             bin_center = i + p
#             freq = bin_center * SAMPLE_RATE / (FRAME_SIZE * 2)
#             if 30 < freq < SAMPLE_RATE / 2 - 1:
#                 midi = freq_to_midi(freq)
#                 if 21 <= midi <= 108:
#                     peaks.append((hps[i], midi))
#     peaks.sort(reverse=True)
#     return {m for _, m in peaks[:max_pitches]}
