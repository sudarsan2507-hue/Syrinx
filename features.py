"""
features.py — MFCC extraction with Frame Relevance Scoring (FRS).

FRS scores each MFCC frame by its normalised cepstral energy and keeps only
frames above a threshold, so the descriptor reflects the active portion of
the utterance rather than the surrounding silence.

Set use_frs=False for the plain full-average of every frame.
"""

import os
import glob
from typing import Literal, Tuple, Union, overload

import numpy as np
import librosa

SAMPLE_RATE = 22050
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
FRS_THRESHOLD = 0.6
MIN_FRAMES = 5
MAX_DURATION = 2.0

# "cepstral" scores frames by sum-of-squared-MFCCs, which is dominated by c0
# (~log energy, large and negative in silence) and therefore ranks SILENT
# frames highest. "rms" uses real short-time energy. Callers that need the old
# behaviour must ask for it by name; models carry their own setting.
DEFAULT_FRS_MODE = "rms"

COMMANDS = ["shush", "click", "whistle", "pop", "hiss", "hum"]


def _core(y, use_frs, frs_mode):
    """Shared feature computation. `y` is float audio already at SAMPLE_RATE."""
    y_trim, _ = librosa.effects.trim(y, top_db=25)
    if y_trim.size >= HOP_LENGTH:
        y = y_trim

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    else:
        return None

    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )

    if mfcc.shape[1] == 0:
        return None

    if frs_mode == "rms":
        energy = librosa.feature.rms(
            y=y, frame_length=N_FFT, hop_length=HOP_LENGTH, center=True
        )[0][: mfcc.shape[1]]
        if energy.shape[0] < mfcc.shape[1]:
            energy = np.pad(energy, (0, mfcc.shape[1] - energy.shape[0]))
    else:
        energy = np.sum(mfcc ** 2, axis=0)
    emax = np.max(energy)
    energy = energy / emax if emax > 0 else energy

    if not use_frs:
        return np.mean(mfcc, axis=1), float(np.mean(energy))

    mask = energy > FRS_THRESHOLD
    selected = mfcc[:, mask]

    if selected.shape[1] < MIN_FRAMES:
        selected = mfcc
        frs_score = float(np.mean(energy))
    else:
        frs_score = float(np.mean(energy[mask]))

    return np.mean(selected, axis=1), frs_score


def extract_features(audio, use_frs=True, frs_mode=None, sr=None):
    """Return (feature_vector, frs_score) or None if the audio is unusable.

    `audio` is either a path to an audio file (the original signature) or a
    1-D numpy array of samples. Arrays are assumed to already be at
    SAMPLE_RATE unless `sr` says otherwise, in which case they are resampled.
    Both paths run the identical feature chain, so training and live
    inference cannot drift apart.

    frs_mode:
      None / "cepstral"  original scoring: sum of squared MFCCs per frame.
                         NOTE this is dominated by c0 (~log energy, large and
                         negative in silence), so it scores SILENT frames
                         highest. Kept as the default for backward
                         compatibility with models already trained on it.
      "rms"              alternative scoring: true short-time RMS energy.
      use_frs=False      plain full-average of every frame (no selection).
    """
    if frs_mode is None:
        frs_mode = DEFAULT_FRS_MODE

    label = "<array>"
    try:
        if isinstance(audio, np.ndarray):
            y = np.asarray(audio, dtype=np.float32).reshape(-1)
            if sr is not None and sr != SAMPLE_RATE:
                y = librosa.resample(y=y, orig_sr=sr, target_sr=SAMPLE_RATE)
            y = y[: int(SAMPLE_RATE * MAX_DURATION)]
        else:
            label = os.path.basename(str(audio))
            y, _ = librosa.load(audio, sr=SAMPLE_RATE, duration=MAX_DURATION)

        if y.size == 0:
            return None
        return _core(y, use_frs, frs_mode)

    except Exception as e:
        print(f"  error on {label}: {e}")
        return None


Arrays = Tuple[np.ndarray, np.ndarray, np.ndarray]
ArraysWithGroups = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@overload
def create_dataset(data_folder, use_frs: bool = ..., verbose: bool = ...,
                   frs_mode=..., *,
                   return_groups: Literal[True]) -> ArraysWithGroups: ...


@overload
def create_dataset(data_folder, use_frs: bool = ..., verbose: bool = ...,
                   frs_mode=...,
                   return_groups: Literal[False] = ...) -> Arrays: ...


def create_dataset(data_folder, use_frs=True, verbose=False, frs_mode=None,
                   return_groups=False) -> Union[Arrays, ArraysWithGroups]:
    """Walk sounds/<split>/<class>/ and build (features, labels, speakers).

    With return_groups=True a fourth array is returned, naming the source
    segment each clip came from. The overloads above let a type checker see
    which shape comes back, so unpacking four names is not flagged as an
    error against the three-name form.
    """
    features, labels, speakers, groups = [], [], [], []

    for label, command in enumerate(COMMANDS):
        folder = os.path.join(data_folder, command)
        if not os.path.isdir(folder):
            print(f"  missing folder: {folder}")
            continue

        files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        kept = 0

        for path in files:
            result = extract_features(path, use_frs=use_frs, frs_mode=frs_mode)
            if result is None:
                continue
            vec, score = result
            features.append(vec)
            labels.append(label)
            speakers.append(os.path.basename(path).split("_")[0])
            groups.append(os.path.basename(path).split("_")[1])
            kept += 1
            if verbose:
                print(f"    {os.path.basename(path)} FRS={score:.3f}")

        if verbose or kept != len(files):
            print(f"  {command:8} {kept}/{len(files)} usable")

    if return_groups:
        return (np.array(features), np.array(labels),
                np.array(speakers), np.array(groups))
    return np.array(features), np.array(labels), np.array(speakers)