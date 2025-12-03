# config.py
import os
import numpy as np

# Dataset parameters
DATASET_DIR = "./maestro_h5"    # Consistent name
YEARS = ["2017", "2018"]        # Reduce to two years for MVP and smaller subset / download
SAMPLE_RATE = 16000             # Actual dataset is downsampled to 16kHz, see details: https://www.kaggle.com/datasets/stareven233/maestrov300-hdf5?resource=download&select=README
MAX_POLYPHONY = 10              # Max number of simultaneous notes
HARMONIC_LIMIT_HZ = 8000        # Upper limit for harmonics in synthesis -->
# the nyquist at 16kHz is 8kHz (accurate digital representation with 16kHz sample rate, extracted with fft - librosa)

# GA parameters (MVP: simplified)
POPULATION_SIZE = 80
NUM_GENERATIONS = 100
TOURNAMENT_SIZE = 5
ELITISM_RATE = 0.05
MUTATION_RATE = 0.35
CROSSOVER_RATE = 0.5 # Crossover rate for SMR

# Critical: stronger consonance + polyphony control
CONSONANCE_BONUS = -1.2    # reward octaves/fifths
DISSONANCE_PENALTY = 2.8   # punish minor 2nds/tritones
POLYPHONY_PENALTY = 0.6    # per extra note beyond 5

# FFT seeding strength
SEEDING_PROB = 0.85
MAX_SEED_NOTES = 8
ENRICH_OCTAVES = True

# Frame segmentation (MVP: smaller for speed; paper uses 2048/441 hop @44.1kHz, dataset is downsampled to 16kHz)
FRAME_SIZE = 1024   # Frame size for FFT - smaller for speed, approx 64ms
HOP_SIZE = 256     # Hop size between frames - smaller for more frames, approx 16ms
WINDOW = "hann"     # Window type for FFT -- hanning window is common

# Number of frames to transcribe per audio file (MVP: up to 10)
NUM_FRAMES_PER_FILE = 5

MAX_FILES = 1

# Fitness parameters
LAMBDA_POLYPHONY = 0.3  # Penalty weight for number of pitches > 6

# Pitch parameters
MIDI_MIN = 21   # A0
MIDI_MAX = 108  # C8

# Cache directories
FFT_CACHE_DIR = "./fft_cache"
SYNTH_CACHE_DIR = "./synth_cache"

# Random seed for reproducibility
RANDOM_SEED = 57
np.random.seed(RANDOM_SEED)