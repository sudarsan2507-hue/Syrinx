"""
calibrate.py -- measure the room and write a sensible onset threshold.

Records ambient noise with nobody making commands, looks at the distribution
of short-time energy, and writes config.json for listen.py to pick up.

  python calibrate.py
  python calibrate.py --seconds 15 --device 2
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from features import SAMPLE_RATE

CONFIG_PATH = "config.json"
BLOCK = 1024
DEFAULT_SECONDS = 10.0

# an onset has to clear the loudest ambient blocks by this much
HEADROOM = 3.0
# ...and never sit below this, so a silent room does not produce a
# threshold so low that every rustle fires
ABSOLUTE_FLOOR = 0.008


def block_rms(y, block=BLOCK):
    n = (len(y) // block) * block
    if n == 0:
        return np.array([])
    return np.sqrt((y[:n].reshape(-1, block) ** 2).mean(axis=1))


def record(seconds, device=None):
    import sounddevice as sd
    print(f"Recording {seconds:.0f}s of room noise.")
    print("Stay quiet -- do not make any of the command sounds.\n")
    for i in (3, 2, 1):
        print(f"  {i}...", end="", flush=True)
        time.sleep(0.7)
    print("  measuring", flush=True)

    frames = int(seconds * SAMPLE_RATE)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1,
                   dtype="float32", device=device)
    sd.wait()
    return audio.reshape(-1)


def summarise(rms, seconds):
    if rms.size == 0:
        raise RuntimeError("no audio captured")

    pct = {p: float(np.percentile(rms, p)) for p in (50, 90, 95, 99)}
    peak = float(rms.max())
    floor = pct[50]
    threshold = max(peak * HEADROOM, ABSOLUTE_FLOOR)

    print(f"\n  blocks analysed   {rms.size} over {seconds:.0f}s")
    print(f"  median RMS        {pct[50]:.5f}")
    print(f"  90th percentile   {pct[90]:.5f}")
    print(f"  95th percentile   {pct[95]:.5f}")
    print(f"  99th percentile   {pct[99]:.5f}")
    print(f"  loudest block     {peak:.5f}")

    if peak > 0.25:
        print("\n  WARNING: the room is loud, or something made a noise during "
              "calibration.\n           Re-run it in the quiet you actually "
              "expect to use this in.")

    multiplier = float(threshold / floor) if floor > 0 else HEADROOM
    return {
        "noise_floor_rms": round(floor, 6),
        "onset_threshold": round(threshold, 6),
        "floor_multiplier": round(min(max(multiplier, 2.0), 12.0), 2),
        "ambient_percentiles": {str(k): round(v, 6) for k, v in pct.items()},
        "ambient_peak": round(peak, 6),
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def merge_and_write(path, measured):
    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(measured)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    return existing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--from-wav", help="analyse a WAV instead of the mic")
    args = ap.parse_args()

    device = args.device
    if device is not None and str(device).isdigit():
        device = int(device)

    try:
        if args.from_wav:
            import librosa
            y, _ = librosa.load(args.from_wav, sr=SAMPLE_RATE, mono=True)
            seconds = len(y) / SAMPLE_RATE
            print(f"Analysing {args.from_wav} ({seconds:.1f}s)")
        else:
            y = record(args.seconds, device=device)
            seconds = args.seconds
    except Exception as e:
        print(f"\nCould not capture audio: {e}")
        print("Check the input device with: python listen.py --list-devices")
        return 1

    measured = summarise(block_rms(y), seconds)
    written = merge_and_write(args.config, measured)

    print(f"\n  -> onset_threshold {measured['onset_threshold']:.5f}")
    print(f"  -> noise_floor_rms {measured['noise_floor_rms']:.5f}")
    print(f"\n  wrote {args.config} ({len(written)} keys)")
    print("  listen.py picks this up automatically at startup.")
    print("\n  Too many false triggers?  raise onset_threshold.")
    print("  Missing quiet sounds?      lower it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
