/*
 * esp32_hid.ino -- USB serial to Bluetooth HID bridge.
 *
 * The laptop does the listening and the classifying. This board does one
 * job: take a word off the serial port and emit the matching Bluetooth
 * consumer-control key to whatever phone it is paired with.
 *
 * Protocol -- one line in, one line out, 115200 baud:
 *
 *     PING       -> PONG
 *     STATUS     -> CONNECTED | DISCONNECTED
 *     PLAYPAUSE  -> OK PLAYPAUSE   (or ERR not-connected)
 *     NEXT       -> OK NEXT
 *     PREV       -> OK PREV
 *     VOLUP      -> OK VOLUP
 *     VOLDOWN    -> OK VOLDOWN
 *     <anything> -> ERR unknown <anything>
 *
 * The onboard LED is solid while a phone is connected and blinks while
 * waiting for one.
 *
 * Library: "ESP32 BLE Keyboard" by T-vK
 *   https://github.com/T-vK/ESP32-BLE-Keyboard
 *   Install the release ZIP via Sketch > Include Library > Add .ZIP Library.
 *   It is not in the Arduino Library Manager.
 *
 * Board: any ESP32 with Bluetooth (ESP32 Dev Module, WROOM-32, etc).
 *   Note that the ESP32-S2 has no Bluetooth and will not work.
 */

#include <BleKeyboard.h>

// Name shown in the phone's Bluetooth pairing list.
BleKeyboard bleKeyboard("Vocal Remote", "DIY", 100);

static const unsigned long SERIAL_BAUD = 115200;
static const uint8_t LED_PIN = 2;          // onboard LED on most dev boards
static const size_t MAX_LINE = 32;
static const unsigned long BLINK_MS = 500;

static String line;
static bool wasConnected = false;
static unsigned long lastBlink = 0;
static bool blinkState = false;


void setup() {
  Serial.begin(SERIAL_BAUD);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  line.reserve(MAX_LINE + 1);
  bleKeyboard.begin();

  Serial.println("READY vocal-remote");
  Serial.println("waiting for a phone to pair with 'Vocal Remote'");
}


// Send one consumer-control key. Returns false if no phone is connected.
static bool tap(const MediaKeyReport &key) {
  if (!bleKeyboard.isConnected()) {
    return false;
  }
  bleKeyboard.write(key);
  return true;
}


static void handleCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  if (cmd.length() == 0) {
    return;
  }

  // -- things that work whether or not a phone is connected --------------
  if (cmd == "PING") {
    Serial.println("PONG");
    return;
  }
  if (cmd == "STATUS") {
    Serial.println(bleKeyboard.isConnected() ? "CONNECTED" : "DISCONNECTED");
    return;
  }

  // -- media keys --------------------------------------------------------
  bool sent;
  if (cmd == "PLAYPAUSE") {
    sent = tap(KEY_MEDIA_PLAY_PAUSE);
  } else if (cmd == "NEXT") {
    sent = tap(KEY_MEDIA_NEXT_TRACK);
  } else if (cmd == "PREV") {
    sent = tap(KEY_MEDIA_PREVIOUS_TRACK);
  } else if (cmd == "VOLUP") {
    sent = tap(KEY_MEDIA_VOLUME_UP);
  } else if (cmd == "VOLDOWN") {
    sent = tap(KEY_MEDIA_VOLUME_DOWN);
  } else {
    Serial.println("ERR unknown " + cmd);
    return;
  }

  if (sent) {
    Serial.println("OK " + cmd);
  } else {
    Serial.println("ERR not-connected");
  }
}


static void updateLed() {
  if (bleKeyboard.isConnected()) {
    digitalWrite(LED_PIN, HIGH);
    return;
  }
  unsigned long now = millis();
  if (now - lastBlink >= BLINK_MS) {
    lastBlink = now;
    blinkState = !blinkState;
    digitalWrite(LED_PIN, blinkState ? HIGH : LOW);
  }
}


static void reportConnectionChanges() {
  bool connected = bleKeyboard.isConnected();
  if (connected != wasConnected) {
    Serial.println(connected ? "CONNECTED" : "DISCONNECTED");
    wasConnected = connected;
  }
}


void loop() {
  updateLed();
  reportConnectionChanges();

  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (line.length() > 0) {
        handleCommand(line);
        line = "";
      }
    } else if (line.length() < MAX_LINE) {
      line += c;
    } else {
      // overlong garbage: drop it rather than wedging the parser
      line = "";
      Serial.println("ERR line-too-long");
    }
  }
}
