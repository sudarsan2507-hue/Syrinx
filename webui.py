"""
webui.py -- browser front-end for the vocal media remote.

    python webui.py
    -> open http://127.0.0.1:8000

Standard library only: http.server plus server-sent events. No Flask, no
websocket package, nothing to install beyond what the listener already needs.

The audio work happens on a worker thread and pushes onto per-client queues;
the SSE handlers drain those. Browsers only ever read that stream and POST
control changes, so nothing in the audio path waits on a socket.
"""

import argparse
import glob
import json
import os
import queue
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from typing import Any, Optional, cast
import numpy as np

from features import SAMPLE_RATE, COMMANDS
from listen import (Config, Classifier, CommandListener, DetectionLog,
                    AudioEngine, CONFIG_PATH, MODEL_PATH, LOG_PATH)
from transports import make_transport, TransportError, COMMAND_KEYS, key_name
from page import PAGE

DB_FLOOR = -60.0
LEVEL_HZ = 25


def to_db(rms):
    return DB_FLOOR if rms <= 1e-7 else max(DB_FLOOR, 20.0 * float(np.log10(rms)))


# ----------------------------------------------------------------------
# state, shared by every connected browser
# ----------------------------------------------------------------------
class Hub:
    def __init__(self):
        self.cfg = Config.load(CONFIG_PATH)
        self.clf = Classifier(MODEL_PATH)
        self.listener: Optional[CommandListener] = None
        self.engine: Optional[AudioEngine] = None
        self.transport: Any = None
        self.log: Optional[DetectionLog] = None
        self.status = "stopped"
        self.status_kind = "idle"
        self.transport_name = "debug"
        self.device = None
        self.replaying = False

        self._clients = set()
        self._lock = threading.Lock()
        threading.Thread(target=self._level_pump, daemon=True).start()

    # -- fan-out -------------------------------------------------------
    def subscribe(self):
        q = queue.Queue(maxsize=200)
        with self._lock:
            self._clients.add(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._clients.discard(q)

    def publish(self, msg):
        with self._lock:
            targets = list(self._clients)
        for q in targets:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass          # a slow browser drops frames rather than blocking

    def _level_pump(self):
        period = 1.0 / LEVEL_HZ
        while True:
            level = self.listener.level if self.listener is not None else 0.0
            self.publish({
                "type": "level",
                "db": round(to_db(level), 1),
                "threshold_db": round(to_db(self.cfg.onset_threshold), 1),
                "running": self.engine is not None and self.engine.running,
                "rate": self.listener.sr if self.listener is not None else None,
            })
            time.sleep(period)

    def _set_status(self, text, kind):
        self.status, self.status_kind = text, kind
        self.publish({"type": "status", "text": text, "kind": kind})

    # -- lifecycle -----------------------------------------------------
    def devices(self):
        import sounddevice as sd
        out = []
        try:
            apis = cast(list, sd.query_hostapis())
            try:
                default = sd.default.device[0]
            except Exception:
                default = None
            from listen import AudioEngine
            for i, raw in enumerate(cast(list, sd.query_devices())):
                d = cast(dict, raw)    # the sounddevice stub is loose
                if d["max_input_channels"] < 1:
                    continue
                try:
                    rate = AudioEngine.negotiate_rate(i)
                except Exception:
                    rate = None
                out.append({"index": i,
                            "name": d["name"],
                            "api": cast(dict, apis[d["hostapi"]])["name"],
                            "rate": rate,
                            "narrow": bool(rate and rate <= 16000),
                            "default": i == default})
        except Exception as e:
            self._set_status(f"device query failed: {e}", "error")
        return out

    def _build_listener(self):
        self.log = DetectionLog(LOG_PATH)
        self.listener = CommandListener(
            self.cfg, self.clf, self.transport, self.log,
            on_event=lambda ev: self.publish({"type": "detection", **ev}),
            quiet=True)

    def start(self, device, transport_name):
        self.stop()
        self.transport_name = transport_name
        try:
            self.transport = make_transport(transport_name)
        except TransportError as e:
            self._set_status(f"transport: {e}", "error")
            return False
        self.device = device
        self._build_listener()
        self.engine = AudioEngine(self.listener, self._set_status)
        self.engine.start(device)
        return True

    def stop(self):
        if self.engine is not None:
            self.engine.stop()
            self.engine = None
        if self.transport is not None:
            self.transport.close()
            self.transport = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self._set_status("stopped", "idle")

    def set_device(self, device):
        self.device = device
        if self.engine is not None and self.engine.running:
            self.engine.start(device)

    def update_config(self, changes):
        for key, value in changes.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, float(value))
        self.publish({"type": "config", **self.snapshot_config()})

    def snapshot_config(self):
        return {"onset_threshold": self.cfg.onset_threshold,
                "min_confidence": self.cfg.min_confidence,
                "cooldown_s": self.cfg.cooldown_s,
                "post_onset_s": self.cfg.post_onset_s}

    def save_config(self):
        data = {}
        if os.path.exists(CONFIG_PATH):
            try:
                data = json.load(open(CONFIG_PATH))
            except Exception:
                data = {}
        data.update(self.snapshot_config())
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
        self._set_status(f"saved {CONFIG_PATH}", "ok")

    # -- replay --------------------------------------------------------
    def replay(self, path):
        if self.replaying:
            return False
        if not os.path.exists(path):
            self._set_status(f"no such file: {path}", "error")
            return False
        threading.Thread(target=self._replay_worker, args=(path,),
                         daemon=True).start()
        return True

    def _replay_worker(self, path):
        import librosa
        self.replaying = True
        owns = self.engine is None or not self.engine.running
        try:
            if owns:
                if self.transport is None:
                    self.transport = make_transport(self.transport_name)
                self._build_listener()
            self._set_status(f"replaying {os.path.basename(path)}", "ok")

            y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            block = self.cfg.block_size
            listener = self.listener
            if listener is None:
                raise RuntimeError("no listener to replay through")
            for i in range(0, len(y), block):
                listener.feed(y[i:i + block], now=i / SAMPLE_RATE)
                listener.drain(wait=0.0)
                chunk = y[i:i + block]
                listener.level = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
                time.sleep(block / SAMPLE_RATE)      # play it at real speed
            tail = int(self.cfg.post_onset_s * SAMPLE_RATE)
            listener.ring.write(np.zeros(tail, dtype=np.float32))
            listener.feed(np.zeros(block, dtype=np.float32),
                          now=(len(y) + tail) / SAMPLE_RATE)
            listener.drain(wait=0.0)
            listener.flush()
            listener.level = 0.0
        except Exception as e:
            self._set_status(f"replay failed: {e}", "error")
        finally:
            self.replaying = False
            if owns:
                self._set_status("stopped", "idle")
            else:
                self._set_status("listening", "ok")

    def sample_files(self):
        out = []
        for pattern in ("sounds/test/*/*.wav", "Recordings/*/*.ogg"):
            for p in sorted(glob.glob(pattern)):
                out.append(p.replace("\\", "/"))
        return out


# ----------------------------------------------------------------------
# http
# ----------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    # assigned in main() before the server starts serving
    hub: "Hub"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass                              # keep the console for the audio loop

    # -- helpers -------------------------------------------------------
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        raw = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    # -- routes --------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        hub = self.hub

        if path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if path == "/api/devices":
            return self._send(200, hub.devices())
        if path == "/api/state":
            return self._send(200, {
                "status": hub.status, "kind": hub.status_kind,
                "running": hub.engine is not None and hub.engine.running,
                "transport": hub.transport_name,
                "commands": COMMANDS,
                "model": hub.clf.describe(),
                "files": hub.sample_files(),
                "keymap": {c: key_name(k) for c, k in COMMAND_KEYS.items()},
                **hub.snapshot_config()})
        if path == "/events":
            return self._sse()
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        hub = self.hub
        body = self._body()

        if path == "/api/start":
            ok = hub.start(body.get("device"), body.get("transport", "debug"))
            return self._send(200, {"ok": ok})
        if path == "/api/stop":
            hub.stop()
            return self._send(200, {"ok": True})
        if path == "/api/device":
            hub.set_device(body.get("device"))
            return self._send(200, {"ok": True})
        if path == "/api/config":
            hub.update_config(body)
            return self._send(200, {"ok": True})
        if path == "/api/save":
            hub.save_config()
            return self._send(200, {"ok": True})
        if path == "/api/replay":
            return self._send(200, {"ok": hub.replay(body.get("file", ""))})
        return self._send(404, {"error": "not found"})

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        q = self.hub.subscribe()
        try:
            while True:
                try:
                    msg = q.get(timeout=10)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                payload = json.dumps(msg)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except Exception:
            pass                          # browser navigated away
        finally:
            self.hub.unsubscribe(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH} not found -- run: python train_model.py")
        return 1

    Handler.hub = Hub()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Vocal Remote -> {url}")
    print(f"  model: {Handler.hub.clf.describe()}")
    print("  Ctrl+C to stop\n")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        Handler.hub.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
