# ESP32 Bluetooth HID bridge

The laptop listens and classifies. This board does one job: read a word off
USB serial and emit the matching Bluetooth media key to the phone.

```
mic -> listen.py -> USB serial -> ESP32 -> Bluetooth HID -> phone
```

## What you need

- An ESP32 with Bluetooth — ESP32 Dev Module, WROOM-32, DevKitC, most cheap
  boards. **Not an ESP32-S2**, it has no Bluetooth.
- A USB cable that carries data, not just power. This trips people up.

## Setup, once

1. **Arduino IDE** → Preferences → Additional Board Manager URLs, add:

   ```
   https://espressif.github.io/arduino-esp32/package_esp32_index.json
   ```

2. Tools → Board → Boards Manager → install **esp32** by Espressif.

3. Install the keyboard library. It is **not** in the Library Manager:
   download the release ZIP from
   <https://github.com/T-vK/ESP32-BLE-Keyboard/releases>, then
   Sketch → Include Library → Add .ZIP Library.

4. Open `esp32_hid/esp32_hid.ino`, select your board under Tools → Board,
   pick the COM port, and upload.

   If upload fails with `Failed to connect`, hold the **BOOT** button while it
   says "Connecting…", then release.

## Pairing

1. Power the board. The onboard LED blinks while it waits.
2. On the phone: Bluetooth → pair with **Vocal Remote**.
3. LED goes solid once connected.

Play something on the phone before testing — most phones ignore media keys
when nothing is loaded.

## Check it works

Open Serial Monitor at **115200 baud**, line ending **Newline**, and type:

| You send    | Board replies       | Effect              |
|-------------|---------------------|---------------------|
| `PING`      | `PONG`              | serial link is fine |
| `STATUS`    | `CONNECTED`         | phone is paired     |
| `PLAYPAUSE` | `OK PLAYPAUSE`      | play/pause          |
| `NEXT`      | `OK NEXT`           | next track          |
| `PREV`      | `OK PREV`           | previous track      |
| `VOLUP`     | `OK VOLUP`          | volume up           |
| `VOLDOWN`   | `OK VOLDOWN`        | volume down         |

`ERR not-connected` means the serial side is fine but no phone is paired.

**Close the Serial Monitor before running `listen.py`** — only one program can
hold the port.

## Run it

```
python listen.py --transport esp32
python listen.py --transport esp32 --port COM5     # if autodetect picks wrong
```

The port is autodetected by looking for CP210x / CH340 / ESP32 in the device
description. `python -m serial.tools.list_ports -v` shows what is attached.

## When it misbehaves

| Symptom | Cause |
|---|---|
| `no serial ports found` | cable is power-only, or driver missing — install the [CP210x](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers) or CH340 driver |
| `could not open serial port: Access is denied` | Serial Monitor still open, or another `listen.py` running |
| Connects, but no `PONG` | sketch not uploaded, or wrong baud |
| `ERR not-connected` on every command | phone unpaired — check the LED is solid |
| Keys do nothing on the phone | no media app is playing; start something first |
| Works, then dies after a while | usually USB power saving — try a different port or a powered hub |

The Python side reconnects on its own: pull the cable and `listen.py` keeps
running, reporting `link lost`, and picks the board up again when you plug it
back in.

## Protocol

One line in, one line out, 115200 8N1. Unknown words get `ERR unknown <word>`.
Lines longer than 32 characters are dropped with `ERR line-too-long` rather
than wedging the parser.

`transports.py` holds the host side of this; `Esp32Transport.send()` treats any
reply starting with `ERR` as a failed send, so a command that never reached the
phone is logged as `accepted_undelivered` rather than silently counted as a
success.
