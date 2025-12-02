# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
from config import *
from ga_transcription import run_ga_for_frame
from dataset_selector import select_small_polyphonic_subset
from preprocessing import cache_fft_for_file
from note_seq import NoteSequence
from synthesizer import synthesize_pitch_set, NormalizationMode
import h5py

H5_PATH = "./maestro_h5/2017/2017.h5"
FILENAME = "MIDI-Unprocessed_041_PIANO041_MID--AUDIO-split_07-06-17_Piano-e_1-01_wav--2"
STATISTICAL_FRAMES = [
    29380, 29376, 29572, 29384, 29200
]

def plot_frame_comparison(frame_idx, target_fft, gt_pitches, pred_pitches, save_dir="figures"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    print(f"Debug: Frame {frame_idx}, GT Pitches: {sorted(gt_pitches)}, Pred Pitches: {sorted(pred_pitches)}")

    # Reconstruct
    gt_wave = synthesize_pitch_set(gt_pitches)
    pred_wave = synthesize_pitch_set(pred_pitches)
    window = np.hanning(len(gt_wave))
    gt_fft = np.fft.rfft(gt_wave * window)
    pred_fft = np.fft.rfft(pred_wave * window)

    freqs = np.linspace(0, SAMPLE_RATE//2, len(target_fft))

    plt.figure(figsize=(12, 7))
    plt.semilogy(freqs, np.abs(target_fft), label="Real Piano", linewidth=2.5, color="black")
    plt.semilogy(freqs, np.abs(gt_fft), '--', label=f"Ground Truth ({len(gt_pitches)} notes)", linewidth=2)
    plt.semilogy(freqs, np.abs(pred_fft), label=f"SMR-GA ({len(pred_pitches)} notes)", linewidth=3, color="green")

    # F1 score
    tp = len(set(pred_pitches) & gt_pitches)
    fp = len(set(pred_pitches) - gt_pitches)
    fn = len(gt_pitches - set(pred_pitches))
    f1 = 2*tp/(2*tp + fp + fn + 1e-8) if (tp+fp+fn)>0 else 0

    plt.title(f"Frame {frame_idx} • GT: {sorted(gt_pitches)} • SMR-GA: {sorted(pred_pitches)}\nF1 = {f1:.3f}", fontsize=14)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5000)
    plt.ylim(1e-4, None)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/frame_{frame_idx:04d}_f1_{f1:.3f}.png", dpi=300)
    plt.close()
    print(f"   → Figure saved: frame_{frame_idx:04d}_f1_{f1:.3f}.png")

def load_midi_notes_from_hdf5(h5_path: str, filename: str):
    with h5py.File(h5_path, "r") as f:
        midi_bytes = f[f'{filename}/midi'][()]
    return NoteSequence.FromString(midi_bytes)

def transcribe_frames(fft_path: str, num_frames: int = NUM_FRAMES_PER_FILE):
    print(f"Loading precomputed FFTs from: {fft_path}")
    fft_frames = np.load(fft_path)
    n = min(num_frames, len(fft_frames))
    results = []
    for i in range(n):
        pred = run_ga_for_frame(fft_frames[i])
        results.append(pred)
        print(f"  Frame {i:2d} → {sorted(pred)}")
    return results


def get_gt_pitches_for_frame(frame_idx: int, ns: NoteSequence):
    t_center = (frame_idx + 0.5) * HOP_SIZE / SAMPLE_RATE   # exact center of frame
    t_start  = frame_idx * HOP_SIZE / SAMPLE_RATE
    t_end    = (frame_idx + 1) * HOP_SIZE / SAMPLE_RATE

    active = set()
    for note in ns.notes:
        # Use generous overlap: any note that overlaps the frame at all
        if note.start_time < t_end and note.end_time > t_start:
            active.add(note.pitch)
    return active

def evaluate_transcription(predictions, ns: NoteSequence, fft_frames, num_plots: int = 8, valid_indices=None):
    frame_dur = FRAME_SIZE / SAMPLE_RATE
    total_frames = int(np.ceil(ns.total_time * SAMPLE_RATE / HOP_SIZE))
    gt_frames = [get_gt_pitches_for_frame(i, ns) for i in range(len(fft_frames))]

    f1s = []
    for i, (pred, gt) in enumerate(zip(predictions, gt_frames)):
        pset = set(pred)
        tp = len(pset & gt)
        fp = len(pset - gt)
        fn = len(gt - pset)
        f1 = 2*tp / (2*tp + fp + fn + 1e-8)
        f1s.append(f1)

        # GENERATE PLOT ONLY FOR TRANSCRIBED FRAMES UP TO num_plots
        if valid_indices and i in valid_indices[:num_plots]:
            print(f" → Plotting frame {i} (F1 = {f1:.3f})")
            plot_frame_comparison(frame_idx=i,
                                  target_fft=fft_frames[i],
                                  gt_pitches=gt,
                                  pred_pitches=pred)

    return f1s

def evaluate_with_waveform_mse(predictions, fft_frames, ns):
    mses = []
    for i, pred in enumerate(predictions):
        gt_pitches = get_gt_pitches_for_frame(i, ns)
        real_wave = np.fft.irfft(fft_frames[i])
        synth_wave = synthesize_pitch_set(pred)
        mse = np.mean((real_wave - synth_wave)**2)
        mses.append(mse)
    return np.mean(mses)

# if __name__ == "__main__":
#     os.makedirs(FFT_CACHE_DIR, exist_ok=True)
#
#     subset = select_small_polyphonic_subset()
#
#     results = []
#     for year, h5path, name in subset:
#         print(f"\n{'='*60}")
#         print(f"TRANSCRIBING: {year}/{name}")
#         print(f"{'='*60}")
#
#         fft_path = cache_fft_for_file(h5path, name)
#         fft_frames = np.load(fft_path)
#         predictions = transcribe_frames(fft_path)
#
#         ns = load_midi_notes_from_hdf5(h5path, name)
#         f1s = evaluate_transcription(predictions, ns, fft_frames, num_plots=5)
#         mean_f1 = np.mean(f1s)
#         results.append(mean_f1)
#
#         print(f"→ F1 scores: {[f'{f:.3f}' for f in f1s]}")
#         print(f"→ MEAN F1: {mean_f1:.3f}")
#
#     print(f"\nFINAL MVP RESULT → Average F1: {np.mean(results):.3f}")

# ==============================================================
if __name__ == "__main__":
    os.makedirs(FFT_CACHE_DIR, exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("="*80)
    print("SMR-GA FINAL EVALUATION — STATISTICAL TEST FRAMES ONLY")
    print(f"File: {FILENAME}")
    print(f"Frames: {len(STATISTICAL_FRAMES)} high-energy frames")
    print("="*80)

    # 1. Load data
    fft_path = cache_fft_for_file(H5_PATH, FILENAME)
    fft_frames = np.load(fft_path)
    ns = load_midi_notes_from_hdf5(H5_PATH, FILENAME)

    # 2. Transcribe only your statistical frames
    predictions = []
    valid_indices = []

    for idx in STATISTICAL_FRAMES:
        if idx >= len(fft_frames):
            print(f"  Skipping frame {idx} (out of range)")
            continue
        pred = run_ga_for_frame(fft_frames[idx])
        predictions.append(pred)
        valid_indices.append(idx)
        print(f"  Frame {idx:5d} → {sorted(pred)}")

    # 3. Evaluate + generate 8 plots
    full_predictions = [set() for _ in range(len(fft_frames))]
    for pred, idx in zip(predictions, valid_indices):
        full_predictions[idx] = set(pred)

    f1s_full = evaluate_transcription(full_predictions, ns, fft_frames, num_plots=8, valid_indices=valid_indices)
    selected_f1s = [f1s_full[i] for i in valid_indices]
    #mean_mse = evaluate_with_waveform_mse(full_predictions, fft_frames, ns)
    #print(f"Mean MSE (time-domain): {mean_mse:.6f}")

    mean_f1 = np.mean(selected_f1s)
    std_f1 = np.std(selected_f1s)
    perfect = sum(1 for f in selected_f1s if abs(f - 1.0) < 0.01)

    print("\n" + "="*80)
    print("FINAL RESULT — EXACT SAME DATA AS STATISTICAL TEST")
    print("="*80)
    print(f"Frames evaluated       : {len(selected_f1s)}")
    print(f"Frame-level F1         : {mean_f1:.3f} ± {std_f1:.3f}")
    print(f"Perfect detection (F1=1.000) : {perfect}/{len(selected_f1s)} frames")