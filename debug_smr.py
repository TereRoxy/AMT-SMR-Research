from synthesizer import synthesize_pitch_set, FRAME_SIZE
from smr_operator import smr_crossover
import numpy as np

p1 = {48, 60, 64, 67}   # C major chord spread
p2 = {48, 60, 63, 67}   # C minor chord spread

child = smr_crossover(p1, p2, target_fft= np.fft.rfft(
    synthesize_pitch_set({48, 60, 65, 69}, frame_size=FRAME_SIZE)
))

print("Parent 1 :", sorted(p1))
print("Parent 2 :", sorted(p2))
print("Child    :", sorted(child))
print("Union    :", sorted(p1 | p2))