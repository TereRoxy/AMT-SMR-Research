# ga_transcription.py
import numpy as np
from synthesizer import *
from smr_operator import smr_crossover
from pitch_detect import detect_pitches_from_frame
from config import *
from piano_templates import synthesize_with_piano_template

# utils.py or inside ga_transcription.py
import numpy as np
from config import SAMPLE_RATE

def is_silent_frame(fft_frame: np.ndarray,
                    energy_db_threshold: float = -45.0,
                    spectral_centroid_threshold: float = 800.0) -> bool:
    """
    Very fast silence detector using only the FFT magnitude.
    Works great on piano recordings.
    """
    mag = np.abs(fft_frame)
    if len(mag) == 0:
        return True

    # 1. Total energy in dB
    energy = np.sum(mag**2)
    energy_db = 10 * np.log10(energy + 1e-12)

    # 2. Spectral centroid (bright = notes, low = silence/reverb tail)
    freqs = np.linspace(0, SAMPLE_RATE/2, len(mag))
    centroid = np.sum(freqs * mag) / (np.sum(mag) + 1e-12)

    # 3. Optional: peakiness (high if tonal, low if noise)
    peakiness = np.max(mag) / (np.mean(mag) + 1e-12)

    # Combine — tuned on MAESTRO
    if energy_db < energy_db_threshold and centroid < spectral_centroid_threshold:
        return True
    if energy_db < -50:
        return True
    return False


# def fitness(individual, target_fft):
#     if not individual:
#         return 200.0
#
#     synth = synthesize_pitch_set(individual, norm_mode=NormalizationMode.PEAK)
#     synth *= np.hanning(FRAME_SIZE)
#     sfft = np.fft.rfft(synth)
#
#     # Log-spectral error
#     target_db = 20 * np.log10(np.abs(target_fft) + 1e-8)
#     synth_db  = 20 * np.log10(np.abs(sfft)      + 1e-8)
#     log_error = np.mean(np.abs(target_db - synth_db))
#
#     # Consonance scoring
#     pitches = sorted(individual)
#     cons = 0.0
#     for i in range(len(pitches)):
#         for j in range(i+1, len(pitches)):
#             interval = abs(pitches[i] - pitches[j]) % 12
#             if interval in {0, 5, 7}:      # octave, fifth, fourth
#                 cons -= CONSONANCE_BONUS
#             if interval in {1, 2, 6, 11}:  # ugly intervals
#                 cons += DISSONANCE_PENALTY
#
#     poly_penalty = POLYPHONY_PENALTY * max(0, len(individual) - 5)**2
#
#     return log_error + cons + poly_penalty

def fitness(individual, target_fft):
    if not individual:
        return 200.0
    synth = synthesize_pitch_set(individual, norm_mode=NormalizationMode.PEAK)
    synth *= np.hanning(FRAME_SIZE)
    sfft = np.fft.rfft(synth)
    # Complex distance (magnitude + phase)
    complex_error = np.mean(np.abs(target_fft - sfft)**2)
    # Add penalties as before
    pitches = sorted(individual)
    cons = 0.0
    for i in range(len(pitches)):
        for j in range(i+1, len(pitches)):
            interval = abs(pitches[i] - pitches[j]) % 12
            if interval in {0, 5, 7}:
                cons -= CONSONANCE_BONUS
            if interval in {1, 2, 6, 11}:
                cons += DISSONANCE_PENALTY
    poly_penalty = POLYPHONY_PENALTY * max(0, len(individual) - 5)**2
    return complex_error + cons + poly_penalty

# 2. Allow mutation to remove ALL notes
def mutate(individual: set, mutation_rate=MUTATION_RATE) -> set:
    new_ind = individual.copy()
    if np.random.rand() < mutation_rate:
        if len(new_ind) > 0 and np.random.rand() < 0.6:  # ↑ increased removal prob
            new_ind.remove(np.random.choice(list(new_ind)))
            if len(new_ind) == 0:
                return set()  # explicitly allow empty
        else:
            new_ind.add(np.random.randint(21, 109))
    return new_ind

# --- Tournament selection ---
def tournament_select(population, fitnesses, k=TOURNAMENT_SIZE):
    # print("Performing tournament selection.")
    selected_idx = np.random.choice(len(population), k)
    best_idx = selected_idx[np.argmin(fitnesses[selected_idx])]
    return population[best_idx]

# --- GA main loop for one frame ---
def run_ga_for_frame(target_fft: np.ndarray, frame_size: int = 1024) -> set:
    print("Starting GA for frame transcription.")
    # --- Initialize population ---
    # seed_pitches = detect_pitches_from_frame(target_fft, max_pitches=8, thresh_rel=0.05)
    # In run_ga_for_frame() — replace population init
    strong_peaks = detect_pitches_from_frame(target_fft, max_pitches=12, thresh_rel=0.03)
    strong_peaks = list(strong_peaks)

    population = []
    for _ in range(POPULATION_SIZE):
        if strong_peaks and np.random.rand() < 0.8:
            base = set(np.random.choice(strong_peaks, size=min(4, len(strong_peaks)), replace=False))
        else:
            base = set()

        # Add octaves and close neighbors
        enriched = base.copy()
        for p in list(base):
            if np.random.rand() < 0.7:
                enriched.add(p + 12)
            if np.random.rand() < 0.3:
                enriched.add(p + 1)
            if np.random.rand() < 0.3:
                enriched.add(p - 1)
        population.append(enriched if enriched else set(np.random.randint(36, 96, 3)))
    print(f"Initialized population with {POPULATION_SIZE} individuals.")

    for gen in range(NUM_GENERATIONS):
        # print(f"Generation {gen+1}/{NUM_GENERATIONS}.")
        # Compute fitnesses
        fitnesses = np.array([fitness(ind, target_fft) for ind in population])

        # Elitism
        elite_count = int(ELITISM_RATE * POPULATION_SIZE)
        elite_idx = fitnesses.argsort()[:elite_count]
        new_population = [population[i] for i in elite_idx]

        # Generate offspring
        while len(new_population) < POPULATION_SIZE:
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            child = smr_crossover(p1, p2, frame_size)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Return best individual
    fitnesses = np.array([fitness(ind, target_fft) for ind in population])
    best_idx = np.argmin(fitnesses)
    print(f"GA completed, best individual: {population[best_idx]}")
    return population[best_idx]
