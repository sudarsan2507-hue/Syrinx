"""
transports.py -- how a recognised sound gets turned into a media key.

One interface, three backends:

  debug   prints what it would have sent. No hardware. The default.
  esp32   sends a short token over USB serial to an ESP32 that is paired to
          the phone as a Bluetooth HID keyboard.
  bluez   the laptop itself acts as a Bluetooth HID peripheral. Linux only.

Every backend takes a HID consumer-control key code in send().
"""

from typing import Any
from abc import ABC, abstractmethod
import sys
import time

# HID consumer control usage IDs
PLAY_PAUSE = 0x00CD
SCAN_NEXT = 0x00B5
SCAN_PREV = 0x00B6
VOLUME_UP = 0x00E9
VOLUME_DOWN = 0x00EA

# sound -> key code
COMMAND_KEYS = {
    "shush": PLAY_PAUSE,
    "hum": PLAY_PAUSE,
    "hiss": SCAN_PREV,
    "pop": SCAN_NEXT,
    "click": VOLUME_DOWN,
    "whistle": VOLUME_UP,
}

# key code -> (human name, serial token understood by the ESP32 sketch)
KEY_INFO = {
    PLAY_PAUSE: ("Play/Pause", "PLAYPAUSE"),
    SCAN_NEXT: ("Next track", "NEXT"),
    SCAN_PREV: ("Previous track", "PREV"),
    VOLUME_UP: ("Volume up", "VOLUP"),
    VOLUME_DOWN: ("Volume down", "VOLDOWN"),
}


def key_name(key_code):
    return KEY_INFO.get(key_code, (f"0x{key_code:04X}", "UNKNOWN"))[0]


def key_token(key_code):
    return KEY_INFO.get(key_code, (None, "UNKNOWN"))[1]


class TransportError(RuntimeError):
    """Raised when a backend cannot be brought up at all."""


class Transport(ABC):
    """A way of delivering a HID consumer-control key to the phone."""

    name = "base"

    @abstractmethod
    def connect(self) -> None:
        """Bring the link up. Raise TransportError if impossible."""

    @abstractmethod
    def send(self, key_code) -> bool:
        """Deliver one key press. Return True if it went out, False if the
        link was down. Must not raise for an ordinary disconnect -- the
        listen loop has to survive the hardware going away."""

    def close(self):
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class DebugTransport(Transport):
    """Prints instead of transmitting. Lets the whole pipeline be exercised
    with no hardware attached."""

    name = "debug"

    def __init__(self, **_):
        self._count = 0

    def connect(self):
        print("[debug] transport ready -- nothing will actually be transmitted")

    def send(self, key_code):
        self._count += 1
        name, token = KEY_INFO.get(key_code, (f"0x{key_code:04X}", "UNKNOWN"))
        print(f"  [debug] -> would send {name:14} "
              f"usage=0x{key_code:04X}  serial={token!r}  (#{self._count})")
        return True

    def close(self):
        if self._count:
            print(f"[debug] {self._count} command(s) would have been sent")


class Esp32Transport(Transport):
    """Sends a line per command over USB serial to an ESP32 running the
    sketch in firmware/esp32_hid/. The ESP32 holds the Bluetooth pairing."""

    name = "esp32"

    def __init__(self, port=None, baud=115200, timeout=0.25, **_):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._serial: Any = None
        self._warned = False

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _autodetect():
        from serial.tools import list_ports
        candidates = []
        for p in list_ports.comports():
            blob = f"{p.description} {p.manufacturer or ''} {p.hwid or ''}".lower()
            score = 0
            for hint in ("cp210", "ch340", "ch910", "esp32", "silicon labs", "wch"):
                if hint in blob:
                    score += 1
            candidates.append((score, p.device, p.description))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _open(self):
        import serial
        port = self.port or self._autodetect()
        if port is None:
            raise TransportError(
                "no serial ports found. Plug in the ESP32, or pass --port COM5")
        ser = serial.Serial(port, self.baud, timeout=self.timeout)
        time.sleep(2.0)          # the ESP32 reboots when the port opens
        ser.reset_input_buffer()
        self.port = port
        return ser

    def _handshake(self, ser):
        ser.write(b"PING\n")
        ser.flush()
        deadline = time.time() + 3.0
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", "replace").strip()
            if not line:
                continue
            if line.upper() == "PONG":
                return True
            print(f"  [esp32] {line}")
        return False

    # -- interface -----------------------------------------------------
    def status(self):
        """Ask the board whether a phone is currently connected."""
        if self._serial is None:
            return "no-serial"
        try:
            self._serial.reset_input_buffer()
            self._serial.write(b"STATUS\n")
            self._serial.flush()
            return self._serial.readline().decode("utf-8", "replace").strip() or "no-reply"
        except Exception:
            return "no-serial"

    def connect(self):
        try:
            self._serial = self._open()
        except TransportError:
            raise
        except Exception as e:
            raise TransportError(f"could not open serial port: {e}") from e

        if self._handshake(self._serial):
            state = self.status()
            print(f"[esp32] serial up on {self.port}, handshake ok, phone {state.lower()}")
            if state == "DISCONNECTED":
                print("[esp32] pair the phone with 'Vocal Remote' -- "
                      "commands will be refused until it connects")
        else:
            print(f"[esp32] connected on {self.port}, but no PONG -- "
                  f"check the sketch is running. Continuing anyway.")

    def send(self, key_code) -> bool:
        token = key_token(key_code)
        if self._serial is None and not self._reconnect():
            return False
        ser = self._serial
        if ser is None:
            return False
        try:
            ser.reset_input_buffer()               # drop any stale chatter
            ser.write(f"{token}\n".encode())
            ser.flush()
            self._warned = False

            reply = ser.readline().decode("utf-8", "replace").strip()
            if reply.startswith("ERR"):
                print(f"  [esp32] {reply} -- is the phone still paired?")
                return False
            return True
        except Exception as e:
            if not self._warned:
                print(f"  [esp32] link lost ({e}) -- will retry on next command")
                self._warned = True
            try:
                if self._serial is not None:
                    self._serial.close()
            except Exception:
                pass
            self._serial = None
            return False

    def _reconnect(self):
        try:
            self._serial = self._open()
            print(f"  [esp32] reconnected on {self.port}")
            return True
        except Exception:
            return False

    def close(self):
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None


class BluezTransport(Transport):
    """The laptop itself as a Bluetooth HID consumer-control peripheral,
    over BlueZ D-Bus. Linux only -- there is no BlueZ on Windows or macOS."""

    name = "bluez"

    # Report map: a single consumer-control input report carrying one 16-bit
    # usage code. Matches the descriptor in the ESP32 sketch.
    REPORT_MAP = bytes([
        0x05, 0x0C,        # Usage Page (Consumer)
        0x09, 0x01,        # Usage (Consumer Control)
        0xA1, 0x01,        # Collection (Application)
        0x85, 0x01,        #   Report ID (1)
        0x19, 0x00,        #   Usage Minimum (0)
        0x2A, 0x9C, 0x02,  #   Usage Maximum (0x029C)
        0x15, 0x00,        #   Logical Minimum (0)
        0x26, 0x9C, 0x02,  #   Logical Maximum (0x029C)
        0x75, 0x10,        #   Report Size (16)
        0x95, 0x01,        #   Report Count (1)
        0x81, 0x00,        #   Input (Data, Array)
        0xC0,              # End Collection
    ])

    def __init__(self, **_):
        self._bus = None
        self._profile_registered = False

    @staticmethod
    def available():
        """True only if this platform could plausibly run the BlueZ path."""
        if not sys.platform.startswith("linux"):
            return False
        try:
            import dbus  # noqa: F401  # pyright: ignore[reportMissingImports]
        except ImportError:
            return False
        import shutil
        return shutil.which("bluetoothctl") is not None

    def connect(self):
        if not sys.platform.startswith("linux"):
            raise TransportError(
                f"the bluez backend is Linux-only and this is {sys.platform}. "
                f"Use --transport esp32 for hardware, or --transport debug to "
                f"run without any.")
        try:
            import dbus  # pyright: ignore[reportMissingImports]
            import dbus.service  # noqa: F401  # pyright: ignore[reportMissingImports]
            import dbus.mainloop.glib  # noqa: F401  # pyright: ignore[reportMissingImports]
        except ImportError as e:
            raise TransportError(
                f"BlueZ backend needs dbus-python and PyGObject: {e}. "
                f"On Debian/Ubuntu: apt install python3-dbus python3-gi") from e

        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        try:
            self._bus = dbus.SystemBus()
            obj = self._bus.get_object("org.bluez", "/org/bluez")
            manager = dbus.Interface(obj, "org.bluez.ProfileManager1")
        except Exception as e:
            raise TransportError(f"cannot reach the BlueZ D-Bus service: {e}") from e

        raise TransportError(
            "the BlueZ HID peripheral profile is not registered yet.\n"
            "    Making a laptop act as a BT HID keyboard needs a "
            "org.bluez.Profile1 registration on PSM 17/19, the SDP record "
            "installed, and bluetoothd started with --noplugin=input.\n"
            "    That is Linux-side system configuration, which I have not "
            "touched. Use --transport esp32 or --transport debug.")

    def send(self, key_code):
        return False

    def close(self):
        self._bus = None


BACKENDS = {
    "debug": DebugTransport,
    "esp32": Esp32Transport,
    "bluez": BluezTransport,
}


def make_transport(name, **kwargs):
    """Build a backend by name, falling back to esp32 then debug when the
    requested one cannot run on this machine."""
    if name not in BACKENDS:
        raise TransportError(f"unknown transport {name!r}, "
                             f"choose from {sorted(BACKENDS)}")

    if name == "bluez" and not BluezTransport.available():
        print(f"[transport] bluez is unavailable on {sys.platform}; "
              f"falling back to esp32")
        name = "esp32"

    transport = BACKENDS[name](**kwargs)
    try:
        transport.connect()
        return transport
    except TransportError as e:
        if name == "debug":
            raise
        print(f"[transport] {name} unavailable: {e}")
        print("[transport] falling back to debug -- nothing will be transmitted")
        fallback = DebugTransport()
        fallback.connect()
        return fallback
