"""
ui.py -- desktop window for the vocal media remote.

  python ui.py

Left panel picks the input device and shows its live level, so you can see
straight away whether the device you chose is actually hearing you. Right side
shows what the classifier thinks, and what it did about it.

Threading note: sounddevice calls back on its own thread and the audio loop
runs on a worker thread. Neither touches a widget. They push onto a queue and
the Tk thread drains it on a timer, because Tkinter is not thread-safe.
"""

import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

from typing import Optional, cast
import numpy as np

from features import SAMPLE_RATE, COMMANDS
from listen import (Config, Classifier, CommandListener, DetectionLog,
                    AudioEngine, CONFIG_PATH, MODEL_PATH, LOG_PATH)
from transports import COMMAND_KEYS, key_name, make_transport, TransportError

# ---------------------------------------------------------------- palette
BG = "#12181d"
PANEL = "#19222a"
PANEL_2 = "#212c35"
LINE = "#2a373e"
TEXT = "#e8eef0"
MUTED = "#7e8f97"
ACCENT = "#4fbcd8"
GOOD = "#7fc79a"
WARN = "#d9a95c"
BAD = "#e08a85"

FONT = "Segoe UI" if sys.platform == "win32" else "Helvetica"
MONO = "Consolas" if sys.platform == "win32" else "Menlo"

DB_FLOOR = -60.0          # bottom of the level meter


def to_db(rms):
    return DB_FLOOR if rms <= 1e-7 else max(DB_FLOOR, 20.0 * np.log10(rms))


def db_fraction(db):
    return min(1.0, max(0.0, (db - DB_FLOOR) / (0.0 - DB_FLOOR)))


# ---------------------------------------------------------------- window
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vocal Remote")
        self.configure(bg=BG)
        self.geometry("940x600")
        self.minsize(880, 560)

        self.ui_queue = queue.Queue()
        self.peak_db = DB_FLOOR
        self.transport = None
        self.log = None
        self.engine: Optional[AudioEngine] = None
        self.listener: Optional[CommandListener] = None
        self.devices = []

        self.cfg = Config.load(CONFIG_PATH)
        try:
            self.clf = Classifier(MODEL_PATH)
        except Exception as e:
            self._fatal(f"Could not load {MODEL_PATH}\n\n{e}\n\n"
                        f"Run:  python train_model.py")
            return

        self._style()
        self._build()
        self._load_devices()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(40, self._tick)

    # ------------------------------------------------------------ chrome
    def _fatal(self, message):
        tk.Label(self, text=message, bg=BG, fg=BAD, font=(FONT, 11),
                 justify="left", padx=30, pady=30).pack(expand=True)

    def _style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("TCombobox", fieldbackground=PANEL_2, background=PANEL_2,
                    foreground=TEXT, arrowcolor=TEXT, bordercolor=LINE,
                    lightcolor=PANEL_2, darkcolor=PANEL_2, selectbackground=PANEL_2,
                    selectforeground=TEXT)
        s.configure("Dark.Horizontal.TScale", background=PANEL, troughcolor=PANEL_2,
                    bordercolor=PANEL, lightcolor=ACCENT, darkcolor=ACCENT)
        s.configure("Dark.Treeview", background=PANEL, fieldbackground=PANEL,
                    foreground=TEXT, bordercolor=LINE, rowheight=23,
                    font=(MONO, 9))
        s.configure("Dark.Treeview.Heading", background=PANEL_2, foreground=MUTED,
                    font=(FONT, 8, "bold"), relief="flat")
        s.map("Dark.Treeview", background=[("selected", PANEL_2)],
              foreground=[("selected", TEXT)])

    def _label(self, parent, text, size=9, color=MUTED, bold=False, mono=False):
        return tk.Label(parent, text=text, bg=parent["bg"], fg=color,
                        font=((MONO if mono else FONT), size,
                              "bold" if bold else "normal"))

    def _heading(self, parent, text):
        lbl = tk.Label(parent, text=text, bg=parent["bg"], fg=MUTED,
                       font=(FONT, 8, "bold"))
        return lbl

    # ------------------------------------------------------------ layout
    def _build(self):
        root = tk.Frame(self, bg=BG)
        root.pack(fill="both", expand=True)

        self.side = tk.Frame(root, bg=PANEL, width=300)
        self.side.pack(side="left", fill="y")
        self.side.pack_propagate(False)
        self._build_side(self.side)

        main = tk.Frame(root, bg=BG)
        main.pack(side="left", fill="both", expand=True)
        self._build_main(main)

    # -- side panel: device + live level -------------------------------
    def _build_side(self, p):
        tk.Frame(p, bg=ACCENT, height=3).pack(fill="x")
        head = tk.Frame(p, bg=PANEL)
        head.pack(fill="x", padx=18, pady=(16, 4))
        self._label(head, "VOCAL REMOTE", 12, TEXT, bold=True).pack(anchor="w")

        self.status_dot = tk.Canvas(head, width=9, height=9, bg=PANEL,
                                    highlightthickness=0)
        self.status_dot.pack(side="left", pady=(4, 0))
        self._dot = self.status_dot.create_oval(1, 1, 8, 8, fill=MUTED, outline="")
        self.status_text = self._label(head, "stopped", 9, MUTED)
        self.status_text.pack(side="left", padx=6, pady=(4, 0))

        # -- device chooser
        box = tk.Frame(p, bg=PANEL)
        box.pack(fill="x", padx=18, pady=(18, 0))
        self._heading(box, "INPUT DEVICE").pack(anchor="w", pady=(0, 6))
        self.device_var = tk.StringVar()
        self.device_box = ttk.Combobox(box, textvariable=self.device_var,
                                       state="readonly", font=(FONT, 9), width=28)
        self.device_box.pack(fill="x")
        self.device_box.bind("<<ComboboxSelected>>", self._on_device_change)
        self._label(box, "switches live", 8, MUTED).pack(anchor="w", pady=(4, 0))

        # -- live level meter
        meter = tk.Frame(p, bg=PANEL)
        meter.pack(fill="x", padx=18, pady=(20, 0))
        self._heading(meter, "LIVE INPUT").pack(anchor="w", pady=(0, 6))

        self.meter = tk.Canvas(meter, height=30, bg=PANEL_2, highlightthickness=0)
        self.meter.pack(fill="x")

        row = tk.Frame(p, bg=PANEL)
        row.pack(fill="x", padx=18, pady=(8, 0))
        self.level_text = self._label(row, "-60.0 dB", 9, TEXT, mono=True)
        self.level_text.pack(side="left")
        self.peak_text = self._label(row, "peak -60.0", 9, MUTED, mono=True)
        self.peak_text.pack(side="right")

        thr = tk.Frame(p, bg=PANEL)
        thr.pack(fill="x", padx=18, pady=(2, 0))
        self.thresh_text = self._label(thr, "", 8, WARN, mono=True)
        self.thresh_text.pack(side="left")

        # -- transport
        tbox = tk.Frame(p, bg=PANEL)
        tbox.pack(fill="x", padx=18, pady=(22, 0))
        self._heading(tbox, "SEND VIA").pack(anchor="w", pady=(0, 6))
        self.transport_var = tk.StringVar(value="debug")
        self.transport_box = ttk.Combobox(
            tbox, textvariable=self.transport_var, state="readonly",
            values=["debug", "esp32", "bluez"], font=(FONT, 9))
        self.transport_box.pack(fill="x")
        self.transport_note = self._label(tbox, "prints only, sends nothing",
                                          8, MUTED)
        self.transport_note.pack(anchor="w", pady=(4, 0))
        self.transport_box.bind("<<ComboboxSelected>>", self._on_transport_change)

        # -- start / stop
        btns = tk.Frame(p, bg=PANEL)
        btns.pack(side="bottom", fill="x", padx=18, pady=18)
        self.toggle_btn = tk.Button(btns, text="Start listening",
                                    command=self._toggle,
                                    bg=ACCENT, fg=BG, font=(FONT, 10, "bold"),
                                    relief="flat", bd=0, pady=9,
                                    activebackground=TEXT, cursor="hand2")
        self.toggle_btn.pack(fill="x")

        self.counts = self._label(btns, "no detections yet", 8, MUTED)
        self.counts.pack(anchor="w", pady=(9, 0))

    # -- main area -----------------------------------------------------
    def _build_main(self, p):
        top = tk.Frame(p, bg=BG)
        top.pack(fill="x", padx=26, pady=(24, 0))

        self._heading(top, "LAST DETECTION").pack(anchor="w")
        self.last_label = tk.Label(top, text="—", bg=BG, fg=TEXT,
                                   font=(FONT, 30, "bold"))
        self.last_label.pack(anchor="w", pady=(2, 0))
        self.last_action = self._label(top, "waiting for a sound", 11, MUTED)
        self.last_action.pack(anchor="w")

        # confidence bars
        conf = tk.Frame(p, bg=BG)
        conf.pack(fill="x", padx=26, pady=(22, 0))
        self._heading(conf, "CONFIDENCE").pack(anchor="w", pady=(0, 8))
        self.bars = {}
        for name in COMMANDS:
            row = tk.Frame(conf, bg=BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=name, bg=BG, fg=MUTED, font=(MONO, 9),
                     width=9, anchor="e").pack(side="left")
            canvas = tk.Canvas(row, height=15, bg=PANEL, highlightthickness=0)
            canvas.pack(side="left", fill="x", expand=True, padx=9)
            value = tk.Label(row, text="0%", bg=BG, fg=MUTED, font=(MONO, 9),
                             width=5, anchor="w")
            value.pack(side="left")
            self.bars[name] = (canvas, value)

        # sliders
        ctl = tk.Frame(p, bg=BG)
        ctl.pack(fill="x", padx=26, pady=(20, 0))
        self._heading(ctl, "TUNING").pack(anchor="w", pady=(0, 6))
        grid = tk.Frame(ctl, bg=BG)
        grid.pack(fill="x")
        self.slider_labels = {}
        self._slider(grid, 0, "Onset threshold", 0.002, 0.15,
                     self.cfg.onset_threshold, self._set_threshold, "{:.3f}")
        self._slider(grid, 1, "Min confidence", 0.0, 1.0,
                     self.cfg.min_confidence, self._set_confidence, "{:.0%}")
        self._slider(grid, 2, "Cooldown", 0.0, 5.0,
                     self.cfg.cooldown_s, self._set_cooldown, "{:.1f}s")
        grid.columnconfigure(1, weight=1)

        # history
        hist = tk.Frame(p, bg=BG)
        hist.pack(fill="both", expand=True, padx=26, pady=(20, 20))
        self._heading(hist, "HISTORY").pack(anchor="w", pady=(0, 6))
        cols = ("time", "sound", "conf", "action", "result")
        self.tree = ttk.Treeview(hist, columns=cols, show="headings",
                                 style="Dark.Treeview", height=6)
        columns: list = [("time", 70, "w"), ("sound", 80, "w"),
                        ("conf", 60, "e"), ("action", 130, "w"),
                        ("result", 130, "w")]
        for col, width, anchor in columns:
            self.tree.heading(col, text=col.upper())
            self.tree.column(col, width=width, anchor=anchor)
        self.tree.pack(fill="both", expand=True)
        self.tree.tag_configure("accepted", foreground=GOOD)
        self.tree.tag_configure("rejected", foreground=MUTED)
        self.tree.tag_configure("cooldown", foreground=WARN)
        self.tree.tag_configure("undelivered", foreground=BAD)

    def _slider(self, parent, row, text, lo, hi, initial, setter, fmt):
        tk.Label(parent, text=text, bg=BG, fg=MUTED, font=(FONT, 9),
                 width=15, anchor="w").grid(row=row, column=0, sticky="w", pady=3)
        # register the readout before the scale exists: ttk.Scale can fire its
        # command during construction, and the callback looks this up
        value = tk.Label(parent, text=fmt.format(initial), bg=BG, fg=TEXT,
                         font=(MONO, 9), width=7, anchor="e")
        value.grid(row=row, column=2, sticky="e", pady=3)
        self.slider_labels[text] = (value, fmt)

        var = tk.DoubleVar(value=initial)
        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var,
                          orient="horizontal", style="Dark.Horizontal.TScale",
                          command=lambda v: setter(float(v)))
        scale.grid(row=row, column=1, sticky="ew", padx=12, pady=3)

    # ------------------------------------------------------------ events
    def _load_devices(self):
        try:
            import sounddevice as sd
            hostapis = cast(list, sd.query_hostapis())
            names, default = [], 0
            try:
                default_index = sd.default.device[0]
            except Exception:
                default_index = None
            for i, raw in enumerate(cast(list, sd.query_devices())):
                d = cast(dict, raw)    # the sounddevice stub is loose
                if d["max_input_channels"] < 1:
                    continue
                api = cast(dict, hostapis[d["hostapi"]])["name"]
                label = f"{d['name'][:34]}  ({api})"
                self.devices.append((i, label))
                if i == default_index:
                    default = len(self.devices) - 1
                names.append(label)
            self.device_box["values"] = names
            if names:
                self.device_box.current(default)
            else:
                self.device_var.set("no input devices found")
        except Exception as e:
            self.device_var.set(f"device query failed: {e}")

    def _selected_device(self):
        idx = self.device_box.current()
        if idx < 0 or idx >= len(self.devices):
            return None
        return self.devices[idx][0]

    def _on_device_change(self, _evt=None):
        self.device_box.selection_clear()
        if self.engine is not None and self.engine.running:
            self.engine.start(self._selected_device())   # restart on the new one

    def _on_transport_change(self, _evt=None):
        self.transport_box.selection_clear()
        notes = {"debug": "prints only, sends nothing",
                 "esp32": "USB serial to the ESP32",
                 "bluez": "Linux only — will fall back"}
        self.transport_note.configure(text=notes.get(self.transport_var.get(), ""))
        if self.engine is not None and self.engine.running:
            self._stop()
            self._start()

    def _set_threshold(self, v):
        # the listener and its detector hold this same Config object, so the
        # change takes effect on the next block with no restart
        self.cfg.onset_threshold = v
        lbl, fmt = self.slider_labels["Onset threshold"]
        lbl.configure(text=fmt.format(v))

    def _set_confidence(self, v):
        self.cfg.min_confidence = v
        lbl, fmt = self.slider_labels["Min confidence"]
        lbl.configure(text=fmt.format(v))

    def _set_cooldown(self, v):
        self.cfg.cooldown_s = v
        lbl, fmt = self.slider_labels["Cooldown"]
        lbl.configure(text=fmt.format(v))

    # ------------------------------------------------------------ run
    def _status(self, text, color):
        self.ui_queue.put({"kind": "status", "text": text, "color": color})

    def _toggle(self):
        if self.engine is not None and self.engine.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        device = self._selected_device()
        try:
            self.transport = make_transport(self.transport_var.get())
        except TransportError as e:
            self._status(f"transport: {e}", BAD)
            return

        self.log = DetectionLog(LOG_PATH)
        self.listener = CommandListener(
            self.cfg, self.clf, self.transport, self.log,
            on_event=lambda ev: self.ui_queue.put({"kind": "detection", **ev}),
            quiet=True)
        self.engine = AudioEngine(self.listener, self._status)
        self.engine.start(device)
        self.toggle_btn.configure(text="Stop", bg=PANEL_2, fg=TEXT)

    def _stop(self):
        if self.engine is not None:
            self.engine.stop()
            self.engine = None
        if self.transport is not None:
            self.transport.close()
            self.transport = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self.toggle_btn.configure(text="Start listening", bg=ACCENT, fg=BG)

    # ------------------------------------------------------------ tick
    def _tick(self):
        self._draw_meter()
        while True:
            try:
                ev = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if ev["kind"] == "status":
                self.status_text.configure(text=ev["text"], fg=ev["color"])
                self.status_dot.itemconfigure(self._dot, fill=ev["color"])
            elif ev["kind"] == "detection":
                self._show_detection(ev)
        self.after(40, self._tick)

    def _draw_meter(self):
        c = self.meter
        c.delete("all")
        w = max(c.winfo_width(), 10)
        h = int(c["height"])

        # narrow directly: a bool variable does not narrow Optional
        level = (self.listener.level
                 if self.engine is not None and self.listener is not None
                 else 0.0)
        db = to_db(level)
        self.peak_db = max(db, self.peak_db - 0.6)      # slow decay

        frac = db_fraction(db)
        thr_frac = db_fraction(to_db(self.cfg.onset_threshold))

        over = db >= to_db(self.cfg.onset_threshold)
        colour = GOOD if over else ACCENT
        if frac > 0.01:
            c.create_rectangle(0, 0, w * frac, h, fill=colour, outline="")

        # threshold marker
        x = w * thr_frac
        c.create_line(x, 0, x, h, fill=WARN, width=2)

        # peak hold
        px = w * db_fraction(self.peak_db)
        c.create_line(px, 2, px, h - 2, fill=TEXT, width=1)

        self.level_text.configure(text=f"{db:5.1f} dB",
                                  fg=GOOD if over else TEXT)
        self.peak_text.configure(text=f"peak {self.peak_db:5.1f}")
        self.thresh_text.configure(
            text=f"threshold {to_db(self.cfg.onset_threshold):.1f} dB "
                 f"({self.cfg.onset_threshold:.3f} rms)")

    def _show_detection(self, ev):
        label = ev.get("label")
        conf = ev.get("confidence", 0.0)
        decision = ev.get("decision", "")
        action = ev.get("action")
        spread = ev.get("spread") or {}

        if label:
            self.last_label.configure(
                text=label.upper(),
                fg=GOOD if decision == "accepted" else
                   WARN if decision == "rejected_cooldown" else
                   BAD if decision == "accepted_undelivered" else MUTED)
        if decision == "accepted":
            detail = f"{action}   ·   {conf:.0%}   ·   {ev['latency_ms']:.0f} ms"
        elif decision == "rejected_low_confidence":
            detail = f"ignored — {conf:.0%} is below {self.cfg.min_confidence:.0%}"
        elif decision == "rejected_cooldown":
            detail = f"ignored — still cooling down"
        elif decision == "accepted_undelivered":
            detail = f"{action} — but the transport did not deliver it"
        else:
            detail = decision.replace("_", " ")
        self.last_action.configure(text=detail)

        for name, (canvas, value) in self.bars.items():
            p = spread.get(name, 0.0)
            canvas.delete("all")
            w = max(canvas.winfo_width(), 10)
            h = int(canvas["height"])
            if p > 0.005:
                col = GOOD if (name == label and decision == "accepted") else \
                      ACCENT if name == label else PANEL_2
                canvas.create_rectangle(0, 0, w * p, h, fill=col, outline="")
            value.configure(text=f"{p:.0%}",
                            fg=TEXT if name == label else MUTED)

        tag = ("accepted" if decision == "accepted" else
               "cooldown" if decision == "rejected_cooldown" else
               "undelivered" if decision == "accepted_undelivered" else "rejected")
        result = {"accepted": "sent",
                  "rejected_low_confidence": "low confidence",
                  "rejected_cooldown": "cooldown",
                  "accepted_undelivered": "not delivered",
                  "rejected_unusable": "unusable audio"}.get(decision, decision)
        self.tree.insert("", 0, values=(time.strftime("%H:%M:%S"),
                                        label or "—", f"{conf:.0%}",
                                        action or "—", result), tags=(tag,))
        for extra in self.tree.get_children()[60:]:
            self.tree.delete(extra)

        s = ev.get("stats", {})
        self.counts.configure(
            text=f"{s.get('accepted', 0)} sent   "
                 f"{s.get('rejected', 0)} rejected   "
                 f"{s.get('cooldown', 0)} cooled")

    def _on_close(self):
        self._stop()
        self.destroy()


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH} not found -- run: python train_model.py")
        return 1
    App().mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
