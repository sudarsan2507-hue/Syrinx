# Syrinx: a vocal media remote

Control music on a phone by making noises at a laptop. Shush at it to pause,
hum to resume, hiss for the previous track, pop for the next one, click and
whistle for volume.

```
mic -> ring buffer -> onset detector -> MFCC -> SVM -> media key -> phone
```

The laptop does the listening. Getting the key to the phone is a separate
concern, handled by a swappable transport, so the whole thing runs and can be
tested with no hardware at all.

## Sounds

| Sound   | Action          | HID usage |
|---------|-----------------|-----------|
| shush   | play / pause    | `0x00CD`  |
| hum     | play / pause    | `0x00CD`  |
| hiss    | previous track  | `0x00B6`  |
| pop     | next track      | `0x00B5`  |
| click   | volume down     | `0x00EA`  |
| whistle | volume up       | `0x00E9`  |

## Quick start

```
pip install -r requirements.txt

python calibrate.py          # 10s of room noise -> config.json
python webui.py              # opens http://127.0.0.1:8000
```

or as a desktop window instead:

```
python ui.py
```

or stay in the terminal:

```
python listen.py --verbose   # debug transport, prints instead of sending
```

### The browser UI

`webui.py` serves a page on localhost using nothing but the standard library
-- `http.server` plus server-sent events, no Flask and no websocket package.
It also has a **Replay** control, so a file can be pushed through the whole
pipeline at real speed with no microphone at all.

`ui.py` is the same thing as a desktop window (Tkinter, no extra install).

The **left panel** picks the input device and shows that device's live level
in dB, with the onset threshold drawn on the meter as a marker and a peak-hold
line. Changing the device restarts the stream immediately, so you can tell at a
glance whether the one you picked is actually hearing you. Nothing is
classified until you press **Start listening**.

The **right side** shows the last detection, a confidence bar for all six
sounds, sliders for threshold / confidence / cooldown that take effect on the
next audio block with no restart, and a scrolling history colour-coded by what
happened: sent, low confidence, cooled down, not delivered.

Try it on a file first, no microphone involved:

```
python listen.py --replay Recordings/Shush/*.ogg
```

## Transports

```
python listen.py --transport debug     # prints what it would send (default)
python listen.py --transport esp32     # USB serial -> ESP32 -> Bluetooth HID
python listen.py --transport bluez     # Linux only, see the caveat below
```

- **debug** - no hardware. Everything else works identically.
- **esp32** - the real path. Flash `firmware/esp32_hid/`, pair the board with
  the phone, run with `--transport esp32`. See [firmware/README.md](firmware/README.md).
- **bluez** - *not implemented.* The class detects the platform and fails with
  an explanation. A working version needs `org.bluez.Profile1` registration on
  PSM 17/19, an SDP record, and `bluetoothd --noplugin=input`.

Unavailable backends fall back - `bluez` → `esp32` → `debug` - printing the
reason at each step, so nothing silently does nothing.

## Tuning

`calibrate.py` writes `config.json`, which `listen.py` reads at startup.

| Problem | Knob |
|---|---|
| Fires at nothing | raise `onset_threshold` |
| Misses quiet sounds | lower `onset_threshold` |
| Right sound, rejected | lower `min_confidence` (default 0.40) |
| Feels sluggish | lower `post_onset_s` - but see the measurements below |
| One sound fires twice | raise `cooldown_s` (default 1.5) |

Latency is almost entirely `post_onset_s`, not compute: MFCC plus SVM takes
about **10 ms**. The window has to wait for the audio *after* the onset before
it can classify, and that wait is the cost. Measured across the recordings:

| `post_onset_s` | latency | correct |
|---|---|---|
| 1.00 | 986 ms | 6/9 |
| **0.70** | **703 ms** | **6/9**  ← the default |
| 0.50 | 471 ms | 5/9 |
| 0.35 | 335 ms | 5/9 |
| 0.25 | 236 ms | 4/9 |

0.70 is where accuracy stops being free, so that is the default.

The first call into librosa costs several seconds of numba JIT. `Classifier`
warms that up on a background thread at startup, so the first real command
takes ~12 ms rather than ~3200 ms.

Rejected predictions are printed with their confidence so the threshold can be
set from real numbers rather than guesswork.

## Logging

Every detection lands in `detections.csv`:

```
timestamp, predicted, confidence, decision, latency_ms, processing_ms
```

`decision` is one of `accepted`, `accepted_undelivered`, `rejected_low_confidence`,
`rejected_cooldown`, `rejected_unusable`.

## Files

| | |
|---|---|
| `webui.py` | localhost web UI - stdlib only, SSE, includes replay |
| `ui.py` | desktop window - device panel, live meter, confidence bars, history |
| `listen.py` | the loop - ring buffer, onsets, cooldown, dispatch, logging |
| `transports.py` | `Transport` ABC and the three backends |
| `calibrate.py` | measures the room, writes `config.json` |
| `features.py` | MFCC extraction; takes a path **or** a numpy array |
| `train_model.py` | fits the SVM, saves `vocal_command_model.pkl` |
| `test_inference.py` | 23 tests, no microphone or hardware required |
| `firmware/esp32_hid/` | the Arduino sketch |
| `prepare_data.py`, `augment.py`, `evaluate.py` | dataset build and scoring |
| `record.py`, `train.py` | the original recording and training scripts |

## Tests

```
python -m pytest test_inference.py -v
```

23 tests covering feature extraction, the ring buffer, the full onset →
dispatch path, cooldown behaviour, and the ESP32 serial protocol against a
mock board. One `xfail`: the model does not currently recognise `hum`.

## The model

`vocal_command_model.pkl` is a bundle - the estimator plus the feature
settings used to train it - so the live path reads its configuration from the
model and cannot drift out of sync with training. A bare estimator also loads.

It is trained on 720 augmented clips derived from 40 segments of 9 phone
recordings of one person. Held-out accuracy is 52.6%, and `hum` is recognised
0 times out of 4. It is enough to demonstrate the loop, not enough to rely on.
More recordings - several sessions, more than one voice - then
`python train_model.py`, is what improves it.
