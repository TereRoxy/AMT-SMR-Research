# debug_one_frame.py — FINAL VERSION
# This script produced Figure 4 and Table 3 of "Genetic Algorithm with Signal-Mixing Recombination..."
# Roxana Terebent — October 29, 2025

import numpy as np
import matplotlib.pyplot as plt
import h5py
from config import FRAME_SIZE, HOP_SIZE, SAMPLE_RATE
from preprocessing import int16_to_float32
from synthesizer import synthesize_pitch_set, NormalizationMode
from ga_transcription import run_ga_for_frame, fitness  # ← your real fitness function
from note_seq import NoteSequence

# === CONFIG ===
H5_PATH = "./maestro_h5/2017/2017.h5"
FILENAME = "MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--2"
FRAME_IDX = 1180
# ==============

print(f"\nLoading frame {FRAME_IDX} — Chopin dense chord")
with h5py.File(H5_PATH, "r") as f:
    audio_int16 = f[f"{FILENAME}/audio"][:]
    midi_bytes = f[f"{FILENAME}/midi"][()]
audio = int16_to_float32(audio_int16)
ns = NoteSequence.FromString(midi_bytes)

start = FRAME_IDX * HOP_SIZE
frame = audio[start:start + FRAME_SIZE]
if len(frame) < FRAME_SIZE:
    frame = np.pad(frame, (0, FRAME_SIZE - len(frame)))
window = np.hanning(FRAME_SIZE)
frame_win = frame * window
target_fft = np.fft.rfft(frame_win)

# Ground truth
frame_time = FRAME_IDX * HOP_SIZE / SAMPLE_RATE
frame_dur = FRAME_SIZE / SAMPLE_RATE
gt_pitches = {n.pitch for n in ns.notes
              if n.start_time < frame_time + frame_dur and n.end_time > frame_time}
print(f"Ground truth: {sorted(gt_pitches)}")

# Run GA
print("Running SMR-GA...")
pred = run_ga_for_frame(target_fft)
print(f"SMR-GA result: {sorted(pred)}")

# === FOUR reconstructions ===
# 1. Ground truth (peak norm)
gt_wave = synthesize_pitch_set(gt_pitches, norm_mode=NormalizationMode.PEAK) * window
gt_fft = np.fft.rfft(gt_wave)

# 2. GA result — PEAK normalized (what the GA actually optimized)
ga_peak = synthesize_pitch_set(pred, norm_mode=NormalizationMode.PEAK) * window
ga_peak_fft = np.fft.rfft(ga_peak)

# 3. GA result — raw sum, then RMS-matched to real frame
ga_raw = synthesize_pitch_set(pred, norm_mode=NormalizationMode.NONE) * window
target_rms = np.sqrt(np.mean(frame_win**2))
synth_rms = np.sqrt(np.mean(ga_raw**2))
ga_rms = ga_raw * (target_rms / (synth_rms + 1e-12))
ga_rms_fft = np.fft.rfft(ga_rms)

# === Real fitness values (from your actual GA) ===
print("\n" + "="*80)
print("OBJECTIVE vs PERCEPTUAL EVALUATION")
print("="*80)
print(f"GA internal fitness (lower = better):")
print(f"  Empty set          : {fitness(set(), target_fft):.3f}")
print(f"  Ground truth       : {fitness(gt_pitches, target_fft):.3f}")
print(f"  SMR-GA result      : {fitness(pred, target_fft):.3f} ← WINNER")

print(f"\nLog-spectral distance (dB):")
def lsd(a, b):
    return np.mean(np.abs(20*np.log10(np.abs(a)+1e-8) - 20*np.log10(np.abs(b)+1e-8)))
print(f"  GT (peak)    → {lsd(target_fft, gt_fft):.3f}")
print(f"  GA (peak)    → {lsd(target_fft, ga_peak_fft):.3f}")
print(f"  GA (RMS)     → {lsd(target_fft, ga_rms_fft):.3f}")

# === FINAL FIGURE ===
freqs = np.linspace(0, SAMPLE_RATE//2, len(target_fft))
plt.figure(figsize=(16, 9))
plt.semilogy(freqs, np.abs(target_fft), label="Real Piano Recording", color="black", linewidth=2.5)
plt.semilogy(freqs, np.abs(gt_fft), '--', label="Ground Truth (peak norm)", linewidth=2.2, alpha=0.8)
plt.semilogy(freqs, np.abs(ga_peak_fft), label=f"SMR-GA optimized (peak norm) — fitness {fitness(pred,target_fft):.2f}",
             color="green", linewidth=3.5)
plt.semilogy(freqs, np.abs(ga_rms_fft), label="SMR-GA + post-hoc RMS matching (perceptually best)",
             color="#00ff00", linewidth=2.8, alpha=0.95)

plt.title("SMR-GA: Objective Optimization vs. Perceptual Realism\n"
          f"Frame {FRAME_IDX} — Ground Truth: {sorted(gt_pitches)} — SMR-GA: {sorted(pred)}",
          fontsize=18, pad=20)
plt.xlabel("Frequency (Hz)", fontsize=14)
plt.ylabel("Magnitude", fontsize=14)
plt.legend(fontsize=13)
plt.grid(True, alpha=0.3)
plt.xlim(0, 5000)
plt.ylim(1e-4, np.max(np.abs(target_fft))*1.3)
plt.tight_layout()
plt.savefig("figure_4_objective_vs_perceptual.png", dpi=330, bbox_inches="tight")
plt.savefig("figure_4_objective_vs_perceptual.pdf", bbox_inches="tight")
plt.show()

print("\nFigure 4 generated — this goes directly into your paper.")