"""
smr_full_ablation_experiment.py
Comprehensive ablation: SMR vs Uniform crossover vs Mutation-only vs FFT-peaks
→ Uses real GA loop (not one-shot crossover)
→ Starts from random population
→ Synthetic piano frames (realistic templates + noise)
→ Statistical significance testing
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon
from typing import Set, List, Dict, Tuple

# Your existing modules
from synthesizer import synthesize_pitch_set, FRAME_SIZE, SAMPLE_RATE, NormalizationMode
from pitch_detect import detect_pitches_from_frame
from smr_operator import smr_crossover
from config import (
    POPULATION_SIZE, NUM_GENERATIONS, TOURNAMENT_SIZE,
    ELITISM_RATE, MUTATION_RATE, CONSONANCE_BONUS,
    DISSONANCE_PENALTY, POLYPHONY_PENALTY
)
from ga_transcription import fitness_magnitude, tournament_select, mutate

# ===================== CONFIG =====================
OUT_DIR = "smr_full_ablation_results"
EXAMPLES_DIR = os.path.join(OUT_DIR, "examples")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)

N_SAMPLES = 10
MIN_NOTES, MAX_NOTES = 2, 6
MIN_PITCH, MAX_PITCH = 36, 96
SNR_DB = 30.0
GA_GENERATIONS = 25
np.random.seed(1234)

# Fitness function (shared)
def compute_fitness(individual: Set[int], target_fft: np.ndarray) -> float:
    if not individual:
        return 1e6
    synth = synthesize_pitch_set(individual, norm_mode=NormalizationMode.PEAK)
    synth *= np.hanning(FRAME_SIZE)
    sfft = np.fft.rfft(synth)
    mag_t = np.abs(target_fft)
    mag_s = np.abs(sfft)
    noise_floor = np.median(mag_t) * 0.5
    residual = np.linalg.norm(np.maximum(mag_t - noise_floor, 0) - np.maximum(mag_s - noise_floor, 0))**2
    poly_pen = 0.3 * max(0, len(individual) - 6)**2
    return float(residual + poly_pen)

# ===================== CROSSOVER VARIANTS =====================
def uniform_crossover(p1: Set[int], p2: Set[int]) -> Set[int]:
    child = set()
    all_pitches = p1 | p2
    for p in all_pitches:
        if np.random.rand() < 0.5:
            child.add(p)
    # Encourage some similarity
    if np.random.rand() < 0.7:
        common = p1 & p2
        child.update(np.random.choice(list(common), size=min(2, len(common)), replace=False))
    return child

def mutation_only(parent: Set[int]) -> Set[int]:
    return mutate(parent, mutation_rate=0.6)  # Higher rate since no crossover

# ===================== GA RUNNER PER VARIANT =====================
def run_ga_variant(target_fft: np.ndarray, crossover_func) -> Set[int]:
    # Random initialization
    population = [
        set(np.random.randint(MIN_PITCH, MAX_PITCH + 1, size=np.random.randint(1, 8)))
        for _ in range(POPULATION_SIZE)
    ]

    for gen in range(GA_GENERATIONS):
        fitnesses = np.array([compute_fitness(ind, target_fft) for ind in population])
        elite_count = int(ELITISM_RATE * POPULATION_SIZE)
        elite_idx = np.argsort(fitnesses)[:elite_count]
        new_pop = [population[i] for i in elite_idx]

        while len(new_pop) < POPULATION_SIZE:
            p1 = tournament_select(population, fitnesses, k=TOURNAMENT_SIZE)
            if crossover_func == mutation_only:
                child = mutation_only(p1)
            else:
                p2 = tournament_select(population, fitnesses, k=TOURNAMENT_SIZE)
                if crossover_func == smr_crossover:
                    child = crossover_func(p1, p2, target_fft=target_fft)
                else:
                    child = crossover_func(p1, p2)
            child = mutate(child, mutation_rate=MUTATION_RATE)
            new_pop.append(child)

        population = new_pop

    fitnesses = np.array([compute_fitness(ind, target_fft) for ind in population])
    best_idx = np.argmin(fitnesses)
    return population[best_idx]

# ===================== METRICS =====================
def precision_recall_f1(pred: Set[int], gt: Set[int]) -> Tuple[float, float, float]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def magnitude_residual(target_mag: np.ndarray, synth_mag: np.ndarray) -> float:
    L = min(len(target_mag), len(synth_mag))
    return float(np.linalg.norm(target_mag[:L] - synth_mag[:L])**2 / L)

# ===================== MAIN EXPERIMENT =====================
def run_ablation_experiment():
    results = []

    print(f"Running full GA ablation on {N_SAMPLES} synthetic frames...")
    for i in range(N_SAMPLES):
        if i % 10 == 0:
            print(f"  Sample {i}/{N_SAMPLES}")

        # Generate ground truth
        n_notes = np.random.randint(MIN_NOTES, MAX_NOTES + 1)
        gt_set = set(np.random.randint(MIN_PITCH, MAX_PITCH + 1, size=n_notes))
        wave = synthesize_pitch_set(gt_set, norm_mode=NormalizationMode.PEAK)
        wave = wave + np.random.normal(0, np.sqrt(np.mean(wave**2) / (10**(SNR_DB/10) + 1e-12)), wave.shape)
        windowed = wave * np.hanning(FRAME_SIZE)
        target_fft = np.fft.rfft(windowed)
        target_mag = np.abs(target_fft)

        row = {
            "sample": i,
            "gt_notes": len(gt_set),
            "gt_midi": sorted(list(gt_set))
        }

        # 1. FFT peak picking baseline
        fft_peaks = detect_pitches_from_frame(target_fft, max_pitches=12, thresh_rel=0.03)
        fft_synth = synthesize_pitch_set(fft_peaks, norm_mode=NormalizationMode.PEAK) * np.hanning(FRAME_SIZE)
        fft_mag = np.abs(np.fft.rfft(fft_synth))
        row["fft_f1"], _, _ = precision_recall_f1(fft_peaks, gt_set)
        row["fft_res"] = magnitude_residual(target_mag, fft_mag)
        row["fft_n"] = len(fft_peaks)

        # 2. SMR GA
        best_smr = run_ga_variant(target_fft, smr_crossover)
        smr_synth = synthesize_pitch_set(best_smr, norm_mode=NormalizationMode.PEAK) * np.hanning(FRAME_SIZE)
        smr_mag = np.abs(np.fft.rfft(smr_synth))
        row["smr_f1"], _, _ = precision_recall_f1(best_smr, gt_set)
        row["smr_res"] = magnitude_residual(target_mag, smr_mag)
        row["smr_n"] = len(best_smr)

        # 3. Uniform crossover GA
        best_uni = run_ga_variant(target_fft, uniform_crossover)
        uni_synth = synthesize_pitch_set(best_uni, norm_mode=NormalizationMode.PEAK) * np.hanning(FRAME_SIZE)
        uni_mag = np.abs(np.fft.rfft(uni_synth))
        row["uni_f1"], _, _ = precision_recall_f1(best_uni, gt_set)
        row["uni_res"] = magnitude_residual(target_mag, uni_mag)
        row["uni_n"] = len(best_uni)

        # 4. Mutation-only GA
        best_mut = run_ga_variant(target_fft, mutation_only)
        mut_synth = synthesize_pitch_set(best_mut, norm_mode=NormalizationMode.PEAK) * np.hanning(FRAME_SIZE)
        mut_mag = np.abs(np.fft.rfft(mut_synth))
        row["mut_f1"], _, _ = precision_recall_f1(best_mut, gt_set)
        row["mut_res"] = magnitude_residual(target_mag, mut_mag)
        row["mut_n"] = len(best_mut)

        results.append(row)

        # Save a few example plots
        if i < 10:
            plt.figure(figsize=(10,5))
            freqs = np.linspace(0, SAMPLE_RATE//2, len(target_mag))
            plt.semilogy(freqs, target_mag + 1e-12, label="Target", linewidth=2)
            plt.semilogy(freqs, smr_mag + 1e-12, label=f"SMR (F1={row['smr_f1']:.3f})", alpha=0.8)
            plt.semilogy(freqs, uni_mag + 1e-12, label=f"Uniform (F1={row['uni_f1']:.3f})", alpha=0.8)
            plt.semilogy(freqs, mut_mag + 1e-12, label=f"Mut-only (F1={row['mut_f1']:.3f})", alpha=0.8)
            plt.legend(); plt.xlabel("Frequency (Hz)"); plt.title(f"Sample {i} — GT: {row['gt_midi']}")
            plt.tight_layout()
            plt.savefig(os.path.join(EXAMPLES_DIR, f"example_{i:03d}.png"), dpi=150)
            plt.close()

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "ablation_results_full.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # ===================== SUMMARY & STATISTICS =====================
    print("\n=== FINAL RESULTS ===")
    metrics = ["f1", "res"]
    methods = ["fft", "smr", "uni", "mut"]
    labels = ["FFT Peaks", "SMR+GA", "Uniform+GA", "Mutation-only"]

    summary = []
    for m, label in zip(methods, labels):
        f1_mean = df[f"{m}_f1"].mean()
        res_mean = df[f"{m}_res"].mean()
        summary.append((label, f1_mean, res_mean))
        print(f"{label:15s} → F1: {f1_mean:.4f} ± {df[f'{m}_f1'].std():.4f}   |   Residual: {res_mean:.4f}")

    # Statistical tests (Wilcoxon vs SMR)
    print("\n=== SIGNIFICANCE (Wilcoxon signed-rank vs SMR) ===")
    for m in ["uni", "mut", "fft"]:
        stat_f1, p_f1 = wilcoxon(df["smr_f1"], df[f"{m}_f1"], alternative='greater')
        stat_res, p_res = wilcoxon(df["smr_res"], df[f"{m}_res"], alternative='less')
        print(f"SMR vs {m.upper():4s}:  F1 p={p_f1:.2e} {'*' if p_f1<0.001 else ''}  |  Res p={p_res:.2e} {'*' if p_res<0.001 else ''}")

    # Boxplots
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.boxplot([df["fft_f1"], df["smr_f1"], df["uni_f1"], df["mut_f1"]],
                tick_labels=labels)
    plt.title("Note-level F1 Score")
    plt.ylabel("F1")

    plt.subplot(1,2,2)
    plt.boxplot([df["fft_res"], df["smr_res"], df["uni_res"], df["mut_res"]],
                tick_labels=labels)
    plt.title("Spectral Residual (lower = better)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "boxplots_comparison.png"), dpi=200)
    plt.close()

    # Scatter: Residual vs F1
    plt.figure(figsize=(7,6))
    for m, c, l in zip(methods, ["red", "green", "blue", "orange"], labels):
        plt.scatter(df[f"{m}_res"], df[f"{m}_f1"], alpha=0.6, c=c, label=l, s=30)
    plt.xlabel("Spectral Residual")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Performance Trade-off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scatter_f1_vs_residual.png"), dpi=200)
    plt.close()

    # Summary text
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        for line in summary:
            f.write(f"{line[0]:15s} F1: {line[1]:.4f}  Residual: {line[2]:.4f}\n")
        f.write("\nWilcoxon p-values (SMR better):\n")
        # (add p-values manually or from above)

    print(f"\nAll plots and results saved to: {OUT_DIR}")
    return df

if __name__ == "__main__":
    df = run_ablation_experiment()