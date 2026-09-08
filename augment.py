"""
augment.py -- expand sounds/train only, to a balanced target per class.

The test set is never touched. Augmentation runs after the split, so no
augmented variant of a test segment can leak into training.
"""

import os, glob
import numpy as np
import librosa
import soundfile as sf

SR = 22050
TARGET_PER_CLASS = 120
COMMANDS = ["shush", "click", "whistle", "pop", "hiss", "hum"]


def augment(y, rng):
    ops = []
    if rng.rand() < 0.7:                                  # pitch
        n = rng.uniform(-2.0, 2.0)
        y = librosa.effects.pitch_shift(y=y, sr=SR, n_steps=n); ops.append(f"pitch{n:+.1f}")
    if rng.rand() < 0.7:                                  # time stretch
        r = rng.uniform(0.85, 1.20)
        y = librosa.effects.time_stretch(y=y, rate=r); ops.append(f"stretch{r:.2f}")
    if rng.rand() < 0.6:                                  # circular time shift
        k = rng.randint(0, len(y))
        y = np.roll(y, k); ops.append("shift")
    if rng.rand() < 0.6:                                  # additive noise
        snr = rng.uniform(15, 35)
        p = np.mean(y ** 2)
        if p > 0:
            y = y + rng.randn(len(y)) * np.sqrt(p / (10 ** (snr / 10)))
        ops.append(f"noise{snr:.0f}dB")
    if rng.rand() < 0.5:                                  # spectral tilt
        b = rng.uniform(-0.4, 0.4)
        y = np.convolve(y, [1.0, b], mode="same"); ops.append("tilt")
    g = rng.uniform(0.4, 1.0)                             # gain
    y = y * g
    peak = np.max(np.abs(y))
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32), ops


def main():
    rng = np.random.RandomState(0)
    print(f"{'class':9}{'base':>6}{'augmented':>11}{'total':>7}")
    print("-" * 34)
    grand = 0
    for c in COMMANDS:
        d = os.path.join("sounds", "train", c)
        base = sorted(p for p in glob.glob(os.path.join(d, "*.wav")) if "_aug" not in p)
        if not base:
            print(f"{c:9}{0:>6}{0:>11}{0:>7}")
            continue
        need = max(0, TARGET_PER_CLASS - len(base))
        made = 0
        for i in range(need):
            src = base[i % len(base)]
            y, _ = librosa.load(src, sr=SR)
            ya, _ = augment(y, rng)
            if ya.size < 512 or not np.isfinite(ya).all() or np.max(np.abs(ya)) < 1e-4:
                continue
            stem = os.path.splitext(os.path.basename(src))[0]
            sf.write(os.path.join(d, f"{stem}_aug{i:04d}.wav"), ya, SR)
            made += 1
        tot = len(base) + made
        grand += tot
        print(f"{c:9}{len(base):>6}{made:>11}{tot:>7}")
    print(f"\n  training set: {grand} clips")
    te = sum(len(glob.glob(os.path.join('sounds','test',c,'*.wav'))) for c in COMMANDS)
    print(f"  test set:     {te} clips (NOT augmented)")


if __name__ == "__main__":
    main()
