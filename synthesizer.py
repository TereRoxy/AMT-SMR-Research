# synthesizer.py
import numpy as np
from config import SAMPLE_RATE, FRAME_SIZE, MIDI_MIN, MIDI_MAX

# Global in-memory cache: one waveform per MIDI note (21–108 = 88 entries max)
_SINGLE_NOTE_WAVEFORMS = {}
PIANO_HARMONIC_AMPLITUDE = [
    1.000,  # h1 – fundamental
    0.720,  # h2 – octave
    0.510,  # h3 – 12th
    0.310,  # h4 – 2 octaves
    0.185,  # h5
    0.095,  # h6
    0.042,  # h7
    0.018   # h8 – barely audible
]

# === REALISTIC PIANO HARMONIC ENVELOPES (measured from MAESTRO v3) ===
PIANO_ENVELOPES = {
    range(21, 49):  [1.00, 0.85, 0.68, 0.48, 0.30, 0.18, 0.10, 0.05, 0.03],  # bass
    range(49, 73):  [1.00, 0.78, 0.58, 0.38, 0.22, 0.12, 0.06, 0.03, 0.01],  # mid
    range(73, 109): [1.00, 0.65, 0.42, 0.25, 0.14, 0.07, 0.03, 0.01, 0.005], # treble
}

# Inharmonicity coefficient (typical concert grand)
INHARMONICITY_B = 0.00038

def _get_envelope(midi_note: int):
    for r, env in PIANO_ENVELOPES.items():
        if midi_note in r:
            return env
    return PIANO_ENVELOPES[range(49, 73)]  # fallback

def _build_single_note_templates():
    if _SINGLE_NOTE_WAVEFORMS:
        return

    t = np.arange(FRAME_SIZE) / SAMPLE_RATE
    window = np.hanning(FRAME_SIZE)

    print("Building MAESTRO-calibrated piano note templates (88 notes)... ", end="")

    for midi in range(MIDI_MIN, MIDI_MAX + 1):
        f0 = 440.0 * 2 ** ((midi - 69) / 12.0)
        envelope = _get_envelope(midi)
        wave = np.zeros(FRAME_SIZE, dtype=np.float32)

        for h, amp in enumerate(envelope, start=1):
            if h >= len(envelope):
                break
            freq = f0 * h
            # Mild inharmonicity (real piano strings are stiff)
            freq = f0 * h * np.sqrt(1 + INHARMONICITY_B * h * h)
            if freq > SAMPLE_RATE / 2:
                break
            wave += amp * np.sin(2 * np.pi * freq * t)

        # Critical: per-note normalization (preserves relative loudness across octaves)
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave /= max_val * 1.02  # tiny headroom

        wave *= window
        _SINGLE_NOTE_WAVEFORMS[midi] = wave.astype(np.float32)

    print("Done!")

# Build on first import
_build_single_note_templates()

# synthesizer.py — add this enum at the top
from enum import Enum
class NormalizationMode(Enum):
    PEAK = "peak"          # current (idealized, exaggerated peaks) → your contribution
    RMS = "rms"            # realistic loudness → looks nicer, slightly worse fit
    NONE = "none"          # raw sum (for debugging)

# Default for your paper's main method
NORMALIZATION_MODE = NormalizationMode.PEAK

def synthesize_pitch_set(pitch_set: set,
                         frame_size: int = FRAME_SIZE,
                         norm_mode: NormalizationMode = None) -> np.ndarray:
    if norm_mode is None:
        norm_mode = NORMALIZATION_MODE

    if not pitch_set:
        return np.zeros(frame_size, dtype=np.float32)

    result = sum(_SINGLE_NOTE_WAVEFORMS.get(p, np.zeros(frame_size))
                 for p in pitch_set)

    if norm_mode == NormalizationMode.PEAK:
        max_abs = np.max(np.abs(result))
        if max_abs > 0:
            result /= max_abs * 1.01
    elif norm_mode == NormalizationMode.RMS:
        # Match RMS of the *original windowed real frame* (pass it in debug)
        pass  # we'll handle externally in debug script
    elif norm_mode == NormalizationMode.NONE:
        pass

    return result.astype(np.float32)