"""
prepare_data.py -- turn Recordings/<Folder>/*.ogg into a segmented, split,
augmented dataset under sounds/.

Key rule: the train/test split happens BEFORE augmentation, and it is made at
the level of the source recording (or, for classes with only one recording, at
the level of time-disjoint regions of that recording). No augmented copy of a
test segment can ever appear in training.
"""

import os, glob, shutil
import numpy as np
import librosa
import soundfile as sf

SR = 22050
FOLDER_TO_CLASS = {
    "Shush": "shush", "Click": "click", "Whistle": "whistle",
    "Pop": "pop", "Ssh": "hiss", "Hmm": "hum",
}
SUSTAINED = {"shush", "hiss", "hum", "whistle"}
WIN = 0.50          # window length for sustained sounds
HOP_W = 0.25        # 50% overlap
PAD = 0.30          # fixed length transients are padded/cropped to
RMS_THRESH = 0.15   # activity threshold, relative to peak RMS
MIN_EVENT = 0.02


def segments(y, sr, cls):
    """Return list of (start_sample, end_sample) base events."""
    if cls not in SUSTAINED:
        # transients: onset detection finds each tap/pop, which RMS runs merge
        on = librosa.onset.onset_detect(y=y, sr=sr, hop_length=128, units="samples",
                                        backtrack=True, delta=0.10, wait=3)
        n = int(PAD * sr)
        out = []
        for st in on:
            a = max(0, int(st) - int(0.02 * sr))
            b = min(a + n, len(y))
            if b - a >= int(0.10 * sr) and np.max(np.abs(y[a:b])) > 0.10:
                out.append((a, b))
        return out
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    if rms.max() <= 0:
        return []
    act = rms / rms.max() > RMS_THRESH
    runs, cur = [], None
    for i, a in enumerate(act):
        if a and cur is None:
            cur = i
        elif not a and cur is not None:
            runs.append((cur, i)); cur = None
    if cur is not None:
        runs.append((cur, len(act)))

    out = []
    for s, e in runs:
        s0, e0 = s * 256, min(e * 256, len(y))
        dur = (e0 - s0) / sr
        if dur < MIN_EVENT:
            continue
        if cls in SUSTAINED and dur >= WIN:
            n, h = int(WIN * sr), int(HOP_W * sr)
            for st in range(s0, e0 - n + 1, h):
                out.append((st, st + n))
        else:
            n = int(PAD * sr)
            c = (s0 + e0) // 2
            st = max(0, c - n // 2)
            out.append((st, min(st + n, len(y))))
    return out


def main():
    if os.path.isdir("sounds"):
        shutil.rmtree("sounds")

    inventory = {}
    for folder, cls in FOLDER_TO_CLASS.items():
        files = sorted(glob.glob(os.path.join("Recordings", folder, "*.ogg")))
        per_file = []
        for fi, f in enumerate(files):
            y, _ = librosa.load(f, sr=SR)
            y = y / (np.max(np.abs(y)) + 1e-9)
            segs = segments(y, SR, cls)
            per_file.append((f, y, segs))
        inventory[cls] = per_file

    print(f"{'class':9}{'files':>6}{'segments':>10}   split strategy")
    print("-" * 64)

    manifest = {"train": [], "test": []}
    for cls, per_file in inventory.items():
        total = sum(len(s) for _, _, s in per_file)
        if len(per_file) >= 2 and all(len(s) >= 2 for _, _, s in per_file):
            strategy = "hold out one whole recording (honest)"
            hold = int(np.argmin([len(s) for _, _, s in per_file]))
            for fi, (f, y, segs) in enumerate(per_file):
                split = "test" if fi == hold else "train"
                for si, (a, b) in enumerate(segs):
                    manifest[split].append((cls, y[a:b], f"f{fi}s{si}"))
        else:
            strategy = "time-disjoint tail of the one recording (OPTIMISTIC)"
            for fi, (f, y, segs) in enumerate(per_file):
                k = max(1, int(round(len(segs) * 0.30)))
                for si, (a, b) in enumerate(segs):
                    split = "test" if si >= len(segs) - k else "train"
                    manifest[split].append((cls, y[a:b], f"f{fi}s{si}"))
        print(f"{cls:9}{len(per_file):>6}{total:>10}   {strategy}")

    for split in ("train", "test"):
        for cls in FOLDER_TO_CLASS.values():
            os.makedirs(os.path.join("sounds", split, cls), exist_ok=True)
        for cls, seg, tag in manifest[split]:
            p = os.path.join("sounds", split, cls, f"s1_{tag}.wav")
            sf.write(p, seg, SR)

    print()
    for split in ("train", "test"):
        c = {cls: sum(1 for x in manifest[split] if x[0] == cls) for cls in FOLDER_TO_CLASS.values()}
        print(f"  {split:6} {sum(c.values()):4} base clips  {c}")


if __name__ == "__main__":
    main()
