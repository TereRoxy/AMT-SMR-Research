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

    W = random_mask(s1.shape)
    smixed = W * s1 + (1 - W) * s2

    # More conservative peak picking for SMR children
    child_set = detect_pitches_from_frame(smixed, max_pitches=10, thresh_rel=0.035)
    return child_set