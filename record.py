"""
record.py — collect non-linguistic vocal sound samples.

Usage:  python record.py
Records 2-second mono clips at 22.05 kHz into sounds/<split>/<class>/.
Filenames carry a speaker tag so cross-speaker splits are possible later.
"""

import os
import sys
import time
import glob

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

SAMPLE_RATE = 22050
DURATION = 2.0
COMMANDS = ["shush", "click", "whistle", "pop", "hiss", "hum"]

ACTION = {
    "shush": "pause",
    "hum": "play / resume",
    "hiss": "previous track",
    "pop": "next track",
    "click": "volume down",
    "whistle": "volume up",
}


def ensure_dirs(split):
    for c in COMMANDS:
        os.makedirs(os.path.join("sounds", split, c), exist_ok=True)


def next_index(split, command, speaker):
    pattern = os.path.join("sounds", split, command, f"{speaker}_*.wav")
    existing = glob.glob(pattern)
    if not existing:
        return 1
    nums = []
    for path in existing:
        stem = os.path.splitext(os.path.basename(path))[0]
        tail = stem.split("_")[-1]
        if tail.isdigit():
            nums.append(int(tail))
    return max(nums) + 1 if nums else 1


def countdown(n=3):
    for i in range(n, 0, -1):
        print(f"  {i}...", end="", flush=True)
        time.sleep(0.7)
    print("  GO", flush=True)


def record_clip():
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def quality_note(y):
    """Quick feedback so bad takes get caught at record time, not at train time."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    if peak < 0.01:
        return f"TOO QUIET (peak {peak:.3f}) — move closer"
    if peak > 0.98:
        return f"CLIPPING (peak {peak:.3f}) — back off a little"
    if rms < 0.005:
        return f"very low energy (rms {rms:.4f})"
    return f"ok (peak {peak:.2f}, rms {rms:.3f})"


def save(y, split, command, speaker, idx):
    path = os.path.join("sounds", split, command, f"{speaker}_{idx:03d}.wav")
    # clip before scaling: a sample above 1.0 would wrap and become a
    # full-scale click of the opposite sign
    safe = np.clip(y, -1.0, 1.0)
    wav.write(path, SAMPLE_RATE, np.round(safe * 32767).astype(np.int16))
    return path


def record_class(split, command, speaker, count):
    print(f"\n{'=' * 58}")
    print(f"  {command.upper()}  ->  {ACTION[command]}")
    print(f"  speaker: {speaker}   set: {split}   target: {count} clips")
    print(f"{'=' * 58}")

    done = 0
    while done < count:
        idx = next_index(split, command, speaker)
        print(f"\n[{done + 1}/{count}]  Enter to record, 's' to skip class, 'q' to quit")
        key = input("> ").strip().lower()
        if key == "q":
            return "quit"
        if key == "s":
            return "skip"

        countdown()
        y = record_clip()
        print(f"  {quality_note(y)}")

        keep = input("  Enter to keep, 'r' to redo: ").strip().lower()
        if keep == "r":
            print("  discarded")
            continue

        path = save(y, split, command, speaker, idx)
        print(f"  saved {path}")
        done += 1

    return "done"


def summary():
    print(f"\n{'=' * 58}")
    print("  CURRENT DATASET")
    print(f"{'=' * 58}")
    for split in ("train", "test"):
        total = 0
        parts = []
        for c in COMMANDS:
            n = len(glob.glob(os.path.join("sounds", split, c, "*.wav")))
            total += n
            parts.append(f"{c}:{n}")
        print(f"  {split:6} total {total:4}   " + "  ".join(parts))
    print()


def main():
    print("\nNON-LINGUISTIC VOCAL SOUND COLLECTION")
    print("Sounds:", ", ".join(COMMANDS))

    try:
        sd.check_input_settings(samplerate=SAMPLE_RATE, channels=1)
    except Exception as e:
        print(f"\nMicrophone problem: {e}")
        print("Check your input device in system settings and try again.")
        sys.exit(1)

    summary()

    split = ""
    while split not in ("train", "test"):
        split = input("Which set are you recording? (train/test): ").strip().lower()
    ensure_dirs(split)

    speaker = ""
    while not speaker.isalnum():
        speaker = input("Speaker tag (short, letters/digits, e.g. s1): ").strip().lower()

    default = 10 if split == "train" else 3
    raw = input(f"Clips per class [{default}]: ").strip()
    count = int(raw) if raw.isdigit() and int(raw) > 0 else default

    for command in COMMANDS:
        result = record_class(split, command, speaker, count)
        if result == "quit":
            print("\nStopped early.")
            break

    summary()
    print("Done. Run train.py when you have both train and test sets.")


if __name__ == "__main__":
    main()