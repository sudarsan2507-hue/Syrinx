"""
listen.py -- continuous listening loop.

Streams the mic into a rolling ring buffer, watches for energy onsets, and
classifies the window around each onset. Accepted commands are dispatched to
a Transport.

  python listen.py                             debug backend, no hardware
  python listen.py --transport esp32           over USB serial to an ESP32
  python listen.py --replay sounds/test/hum/x.wav

Every detection, accepted or not, is appended to detections.csv.
"""

import argparse
import csv
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, asdict

from typing import Any, cast
import numpy as np
import joblib

from features import extract_features, COMMANDS, SAMPLE_RATE
from transports import COMMAND_KEYS, key_name, make_transport, TransportError

CONFIG_PATH = "config.json"
MODEL_PATH = "vocal_command_model.pkl"
LOG_PATH = "detections.csv"


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
@dataclass
class Config:
    onset_threshold: float = 0.020     # absolute RMS floor for an onset
    noise_floor_rms: float = 0.004     # measured ambient level
    floor_multiplier: float = 4.0      # onset must beat this * noise floor
    min_confidence: float = 0.40       # reject below this posterior
    cooldown_s: float = 1.5            # silence after an accepted command
    refractory_s: float = 0.35         # min gap between onset triggers
    pre_onset_s: float = 1.0           # window starts this far before onset
    post_onset_s: float = 0.70         # window ends this far after onset;
                                       # measured: 0.70 matches 1.0 for
                                       # accuracy and is ~283 ms quicker
    block_size: int = 1024

    @classmethod
    def load(cls, path):
        cfg = cls()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                known = {k: v for k, v in data.items() if k in cls.__annotations__}
                cfg = cls(**{**asdict(cfg), **known})
                print(f"[config] loaded {path}")
            except Exception as e:
                print(f"[config] could not read {path} ({e}), using defaults")
        else:
            print(f"[config] no {path} yet -- using defaults. "
                  f"Run calibrate.py to tune for your room.")
        return cfg

    @property
    def window_s(self):
        return self.pre_onset_s + self.post_onset_s


# ----------------------------------------------------------------------
# audio plumbing
# ----------------------------------------------------------------------
class RingBuffer:
    """Fixed-size rolling buffer addressed by absolute sample index."""

    def __init__(self, capacity):
        self._buf = np.zeros(int(capacity), dtype=np.float32)
        self._cap = int(capacity)
        self._total = 0
        self._lock = threading.Lock()

    @property
    def total(self):
        with self._lock:
            return self._total

    def write(self, block):
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        n = block.size
        if n == 0:
            return
        with self._lock:
            if n >= self._cap:
                # only the last `cap` samples can survive; skip the rest but
                # still advance the counter so `pos` stays modularly aligned
                self._total += n - self._cap
                block = block[-self._cap:]
                n = self._cap
            pos = self._total % self._cap
            end = pos + n
            if end <= self._cap:
                self._buf[pos:end] = block
            else:
                first = self._cap - pos
                self._buf[pos:] = block[:first]
                self._buf[:n - first] = block[first:]
            self._total += n

    def read(self, start, length):
        """Samples [start, start+length). None if that range has been
        overwritten or has not arrived yet."""
        start, length = int(start), int(length)
        if length <= 0:
            return None
        with self._lock:
            if start < 0:
                length += start
                start = 0
                if length <= 0:
                    return None
            if start + length > self._total:
                return None
            if start < self._total - self._cap:
                return None
            pos = start % self._cap
            end = pos + length
            if end <= self._cap:
                return self._buf[pos:end].copy()
            first = self._cap - pos
            return np.concatenate([self._buf[pos:], self._buf[:length - first]])


class OnsetDetector:
    """Energy onset detection against a running noise floor."""

    def __init__(self, cfg, sample_rate):
        self.cfg = cfg
        self.sr = sample_rate
        self.floor = cfg.noise_floor_rms
        self._armed = True
        self._last_onset = -1e9

    @property
    def threshold(self):
        return max(self.cfg.onset_threshold, self.floor * self.cfg.floor_multiplier)

    def push(self, abs_index, rms, now):
        """Feed one block. Returns the onset's absolute sample index, or None."""
        thr = self.threshold
        onset = None

        if rms > thr and self._armed and (now - self._last_onset) > self.cfg.refractory_s:
            onset = abs_index
            self._armed = False
            self._last_onset = now
        elif rms < thr * 0.6:
            self._armed = True
            # only track the floor while things are quiet
            self.floor = 0.95 * self.floor + 0.05 * rms

        return onset


class Classifier:
    """Wraps the saved bundle and keeps the feature settings that were used
    at training time, so the live path matches exactly."""

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- run: python train_model.py")
        obj: Any = joblib.load(path)
        self.model: Any = None
        if isinstance(obj, dict) and "model" in obj:
            self.model = obj["model"]
            self.commands = obj.get("commands", COMMANDS)
            self.use_frs = obj.get("use_frs", True)
            self.frs_mode = obj.get("frs_mode", "cepstral")
        else:
            # a bare estimator saved by an older script
            self.model = obj
            self.commands = COMMANDS
            self.use_frs = True
            self.frs_mode = "cepstral"
        if not hasattr(self.model, "predict_proba"):
            raise ValueError(
                "the model has no predict_proba -- retrain with probability=True")
        self.warm = threading.Event()
        threading.Thread(target=self.warm_up, daemon=True).start()

    def warm_up(self):
        """Run one throwaway prediction so numba and librosa do their JIT and
        lazy imports now, rather than on the user's first command.

        Measured here: the first call through the feature path costs several
        seconds, every call after it about 10 ms. This runs on a background
        thread so the window or server comes up straight away; a command that
        lands before it finishes simply takes the slow path once.
        """
        noise = (np.random.RandomState(0).randn(int(SAMPLE_RATE * 2.0))
                 * 0.05).astype(np.float32)
        started = time.monotonic()
        try:
            self.predict_detail(noise)
        except Exception:
            pass
        self.warm.set()
        return (time.monotonic() - started) * 1000.0

    def describe(self):
        return (f"{type(self.model).__name__} "
                f"use_frs={self.use_frs} frs_mode={self.frs_mode} "
                f"classes={len(self.commands)}")

    def predict(self, audio, sr=None):
        """audio -> (label, confidence) or (None, 0.0) if unusable."""
        label, conf, _ = self.predict_detail(audio, sr=sr)
        return label, conf

    def predict_detail(self, audio, sr=None):
        """audio -> (label, confidence, {class: probability}).

        The full posterior is what the UI draws its bars from.
        """
        result = extract_features(audio, use_frs=self.use_frs,
                                  frs_mode=self.frs_mode, sr=sr)
        if result is None:
            return None, 0.0, {}
        vec = np.asarray(result[0], dtype=np.float64).reshape(1, -1)
        probs = self.model.predict_proba(vec)[0]

        spread = {}
        for position, class_index in enumerate(self.model.classes_):
            idx = int(class_index)
            name = self.commands[idx] if idx < len(self.commands) else str(idx)
            spread[name] = float(probs[position])

        k = int(np.argmax(probs))
        idx = int(self.model.classes_[k])
        label = self.commands[idx] if idx < len(self.commands) else str(idx)
        return label, float(probs[k]), spread


class DetectionLog:
    COLUMNS = ["timestamp", "predicted", "confidence", "decision",
               "latency_ms", "processing_ms"]

    def __init__(self, path):
        self.path = path
        new = not os.path.exists(path)
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        if new:
            self._w.writerow(self.COLUMNS)
            self._fh.flush()

    def write(self, predicted, confidence, decision, latency_ms, processing_ms):
        self._w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            predicted if predicted is not None else "",
            f"{confidence:.4f}",
            decision,
            f"{latency_ms:.1f}",
            f"{processing_ms:.1f}",
        ])
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# the loop itself
# ----------------------------------------------------------------------
class CommandListener:
    def __init__(self, cfg, classifier, transport, log, verbose=False,
                 on_event=None, quiet=False):
        self.cfg = cfg
        self.clf = classifier
        self.transport = transport
        self.log = log
        self.verbose = verbose
        # on_event is called from whichever thread drives drain(); the UI
        # uses it to push onto a queue rather than touching widgets directly.
        self.on_event = on_event
        self.quiet = quiet
        self.level = 0.0             # most recent block RMS, for a live meter
        # Devices differ in what rates they will open at, so the stream may
        # not be at SAMPLE_RATE. Buffer at whatever the device gives us and
        # resample once, per window, at classification time.
        self.sr = SAMPLE_RATE
        self.ring = RingBuffer(int((cfg.window_s + 1.0) * self.sr))
        self.detector = OnsetDetector(cfg, self.sr)
        self.events = queue.Queue()
        self._pending = []           # (onset_abs, ready_abs, onset_time)
        # Cooldown lives on the caller's clock (audio time), not wall-clock.
        # In live use the two match; in replay, wall-clock would suppress an
        # entire file's worth of commands in one go.
        self._cooldown_until = -1e9
        self._last_feed_time = 0.0
        self.stats = {"onsets": 0, "accepted": 0, "rejected": 0, "cooldown": 0}

    def set_stream_rate(self, sr):
        """Point the buffers at a new capture rate. Call before starting."""
        sr = int(sr)
        if sr == self.sr and self.ring is not None:
            return
        self.sr = sr
        self.ring = RingBuffer(int((self.cfg.window_s + 1.0) * sr))
        self.detector = OnsetDetector(self.cfg, sr)
        self._pending = []

    # -- called from the audio callback (keep it cheap) -----------------
    def feed(self, block, now=None):
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        start = self.ring.total
        self.ring.write(block)
        stamp = now if now is not None else time.monotonic()
        self._last_feed_time = stamp
        rms = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0
        self.level = rms
        self.events.put((start, rms, stamp))

    # -- main-thread work ----------------------------------------------
    def drain(self, wait=0.05):
        deadline = time.monotonic() + wait
        while True:
            try:
                timeout = max(0.0, deadline - time.monotonic())
                start, rms, now = self.events.get(timeout=timeout)
            except queue.Empty:
                break
            onset = self.detector.push(start, rms, now)
            if onset is not None:
                self.stats["onsets"] += 1
                ready = onset + int(self.cfg.post_onset_s * self.sr)
                self._pending.append((onset, ready, now))
                if self.verbose:
                    print(f"  onset @ sample {onset} rms={rms:.4f} "
                          f"thr={self.detector.threshold:.4f}")
            if time.monotonic() >= deadline:
                break
        self._process_ready()

    def _process_ready(self, force=False):
        if not self._pending:
            return
        total = self.ring.total
        still = []
        for onset, ready, onset_time in self._pending:
            if total >= ready or force:
                self._classify(onset, onset_time, self._last_feed_time)
            else:
                still.append((onset, ready, onset_time))
        self._pending = still

    def _classify(self, onset, onset_time, ready_time):
        # Two clocks are in play: the caller's (real for the mic, synthetic in
        # replay) measures the deliberate wait for post-onset audio, and
        # time.monotonic() measures actual compute. Keep them separate.
        wait_ms = max(0.0, (ready_time - onset_time) * 1000.0)
        t_ready = time.monotonic()
        start = onset - int(self.cfg.pre_onset_s * self.sr)
        length = int(self.cfg.window_s * self.sr)
        window = self.ring.read(start, length)
        if window is None:
            window = self.ring.read(max(0, start), min(length, self.ring.total - max(0, start)))
        if window is None or window.size < self.sr // 4:
            return

        label, conf, spread = self.clf.predict_detail(window, sr=self.sr)
        processing = (time.monotonic() - t_ready) * 1000.0
        latency = wait_ms + processing

        if label is None:
            self.stats["rejected"] += 1
            if not self.quiet:
                print(f"  unusable audio, ignored")
            self.log.write(None, 0.0, "rejected_unusable", latency, processing)
            self._emit(None, 0.0, {}, "rejected_unusable", None, latency)
            return

        if onset_time < self._cooldown_until:
            self.stats["cooldown"] += 1
            if not self.quiet:
                print(f"  {label:8} {conf:5.1%}  held down by cooldown "
                      f"({self._cooldown_until - onset_time:.2f}s left)")
            self.log.write(label, conf, "rejected_cooldown", latency, processing)
            self._emit(label, conf, spread, "rejected_cooldown", None, latency)
            return

        if conf < self.cfg.min_confidence:
            self.stats["rejected"] += 1
            if not self.quiet:
                print(f"  {label:8} {conf:5.1%}  REJECTED "
                      f"(below {self.cfg.min_confidence:.0%})")
            self.log.write(label, conf, "rejected_low_confidence", latency, processing)
            self._emit(label, conf, spread, "rejected_low_confidence", None, latency)
            return

        key = COMMAND_KEYS.get(label)
        if key is None:
            self.log.write(label, conf, "rejected_unmapped", latency, processing)
            self._emit(label, conf, spread, "rejected_unmapped", None, latency)
            return

        sent = self.transport.send(key)
        processing = (time.monotonic() - t_ready) * 1000.0
        latency = wait_ms + processing
        self.stats["accepted"] += 1
        self._cooldown_until = ready_time + self.cfg.cooldown_s
        status = "" if sent else "  (transport down, not delivered)"
        if not self.quiet:
            print(f"  {label:8} {conf:5.1%}  -> {key_name(key):14} "
                  f"{latency:6.0f} ms{status}")
        decision = "accepted" if sent else "accepted_undelivered"
        self.log.write(label, conf, decision, latency, processing)
        self._emit(label, conf, spread, decision, key_name(key), latency)

    def _emit(self, label, conf, spread, decision, action, latency_ms):
        """Hand a finished detection to whoever is watching (the UI)."""
        if self.on_event is None:
            return
        try:
            self.on_event({
                "label": label,
                "confidence": conf,
                "spread": spread,
                "decision": decision,
                "action": action,
                "latency_ms": latency_ms,
                "stats": dict(self.stats),
            })
        except Exception:
            # a broken listener must never take the audio loop down
            pass

    def flush(self):
        self._process_ready(force=True)

    def summary(self):
        s = self.stats
        print(f"\n  onsets {s['onsets']}   accepted {s['accepted']}   "
              f"rejected {s['rejected']}   suppressed by cooldown {s['cooldown']}")


# ----------------------------------------------------------------------
# live audio engine (used by both front-ends)
# ----------------------------------------------------------------------
class AudioEngine:
    """Owns the input stream and the drain loop. Restartable on a new device."""

    def __init__(self, listener, on_status):
        self.listener = listener
        self.on_status = on_status
        self._thread = None
        self._stop = threading.Event()
        self.device = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, device):
        self.stop()
        self.device = device
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(device, self._stop),
                                        daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        self.on_status("stopped", "idle")

    @staticmethod
    def negotiate_rate(device):
        """Pick a capture rate this device will actually accept.

        Many Windows inputs refuse 22050 outright -- WASAPI does no rate
        conversion, and Bluetooth headsets are commonly fixed at 16 kHz. Ask
        for our preferred rate, and fall back to whatever the device reports
        as its own. The window gets resampled at classification time, so the
        features come out identical either way.
        """
        import sounddevice as sd

        candidates = [SAMPLE_RATE]
        try:
            info = cast(dict, sd.query_devices(device, "input") if device is not None
                        else sd.query_devices(kind="input"))
            native = int(info["default_samplerate"])
            if native not in candidates:
                candidates.append(native)
        except Exception:
            native = None
        for extra in (48000, 44100, 16000):
            if extra not in candidates:
                candidates.append(extra)

        for rate in candidates:
            try:
                sd.check_input_settings(device=device, samplerate=rate, channels=1)
                return rate
            except Exception:
                continue
        return native or SAMPLE_RATE

    def _run(self, device, stop):
        import sounddevice as sd

        backoff = 1.0
        while not stop.is_set():
            try:
                rate = self.negotiate_rate(device)
                self.listener.set_stream_rate(rate)
                # keep blocks ~46 ms whatever the rate, so onset timing and
                # meter refresh feel the same across devices
                block = max(256, int(round(self.listener.cfg.block_size
                                           * rate / SAMPLE_RATE)))

                def callback(indata, frames, time_info, status):
                    self.listener.feed(indata[:, 0])

                with sd.InputStream(samplerate=rate, channels=1,
                                    dtype="float32", blocksize=block,
                                    device=device, callback=callback):
                    backoff = 1.0
                    note = "" if rate == SAMPLE_RATE else f" @ {rate} Hz"
                    self.on_status(f"listening{note}", "ok")
                    while not stop.is_set():
                        self.listener.drain(wait=0.05)
            except Exception as e:
                if stop.is_set():
                    break
                msg = str(e).split("\n")[0][:44]
                self.on_status(f"retrying: {msg}", "error")
                stop.wait(backoff)
                backoff = min(backoff * 2, 10.0)
        self.listener.level = 0.0


# ----------------------------------------------------------------------
# run modes
# ----------------------------------------------------------------------
def run_replay(path, listener, cfg):
    import librosa
    print(f"\n[replay] {path}")
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    print(f"[replay] {len(y) / SAMPLE_RATE:.2f}s of audio, "
          f"feeding in {cfg.block_size}-sample blocks\n")

    t0 = time.monotonic()
    for i in range(0, len(y), cfg.block_size):
        block = y[i:i + cfg.block_size]
        # a synthetic clock so latency figures stay meaningful in replay
        listener.feed(block, now=t0 + (i / SAMPLE_RATE))
        listener.drain(wait=0.0)
    # let the tail be classified even though the file ran out
    listener.ring.write(np.zeros(int(cfg.post_onset_s * SAMPLE_RATE), dtype=np.float32))
    listener.drain(wait=0.0)
    listener.flush()
    listener.summary()


def run_mic(listener, cfg, device=None):
    import sounddevice as sd

    print(f"\n[mic] listening at {SAMPLE_RATE} Hz, block {cfg.block_size}")
    print(f"[mic] onset threshold {listener.detector.threshold:.4f}  "
          f"noise floor {listener.detector.floor:.4f}")
    print("[mic] make a sound. Ctrl+C to stop.\n")

    def callback(indata, frames, time_info, status):
        if status and listener.verbose:
            print(f"  [stream] {status}")
        listener.feed(indata[:, 0])

    backoff = 1.0
    while True:
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                blocksize=cfg.block_size, device=device,
                                callback=callback):
                backoff = 1.0
                while True:
                    listener.drain(wait=0.05)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"\n[mic] stream error: {e}")
            print(f"[mic] retrying in {backoff:.0f}s "
                  f"(unplug/replug the mic if this persists)")
            try:
                time.sleep(backoff)
            except KeyboardInterrupt:
                raise
            backoff = min(backoff * 2, 30.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transport", default="debug", choices=["debug", "esp32", "bluez"])
    ap.add_argument("--replay", metavar="FILE", help="run the pipeline on a WAV instead of the mic")
    ap.add_argument("--port", help="serial port for the esp32 backend, e.g. COM5")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--log", default=LOG_PATH)
    ap.add_argument("--min-confidence", type=float, default=None)
    ap.add_argument("--cooldown", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=None, help="override onset threshold")
    ap.add_argument("--device", default=None, help="input device index or name")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    cfg = Config.load(args.config)
    if args.min_confidence is not None:
        cfg.min_confidence = args.min_confidence
    if args.cooldown is not None:
        cfg.cooldown_s = args.cooldown
    if args.threshold is not None:
        cfg.onset_threshold = args.threshold

    try:
        clf = Classifier(args.model)
    except Exception as e:
        print(f"[model] {e}")
        return 1
    print(f"[model] {clf.describe()}")
    print(f"[gate ] confidence >= {cfg.min_confidence:.0%}, "
          f"cooldown {cfg.cooldown_s:.1f}s")

    device = args.device
    if device is not None and str(device).isdigit():
        device = int(device)

    try:
        transport = make_transport(args.transport, port=args.port, baud=args.baud)
    except TransportError as e:
        print(f"[transport] {e}")
        return 1

    log = DetectionLog(args.log)
    listener = CommandListener(cfg, clf, transport, log, verbose=args.verbose)

    try:
        if args.replay:
            if not os.path.exists(args.replay):
                print(f"[replay] no such file: {args.replay}")
                return 1
            run_replay(args.replay, listener, cfg)
        else:
            run_mic(listener, cfg, device=device)
    except KeyboardInterrupt:
        print("\n\nstopped")
        listener.summary()
    finally:
        transport.close()
        log.close()
        print(f"[log  ] {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
