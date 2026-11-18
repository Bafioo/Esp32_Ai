<p align="center">
  <img src="docs/logo.svg" alt="ESP32-S3 Keyword Spotting logo" width="88" height="88">
</p>

<h1 align="center">ESP32-S3 Keyword Spotting</h1>

<p align="center">
  <strong>Your wake phrase. Your language. On-device recognition.</strong><br>
  Train a small speech recognition model for the ESP32-S3 with an INMP441 microphone.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Hardware-ESP32--S3-201d1d?style=flat-square" alt="ESP32-S3 hardware">
  <img src="https://img.shields.io/badge/Training-Python-201d1d?style=flat-square&amp;logo=python&amp;logoColor=fdfcfc" alt="Python training">
  <img src="https://img.shields.io/badge/Firmware-Arduino-201d1d?style=flat-square&amp;logo=arduino&amp;logoColor=fdfcfc" alt="Arduino firmware">
</p>

<p align="center">
  <a href="#hardware-and-software">Hardware</a> &middot;
  <a href="#1-collect-your-recordings">Record</a> &middot;
  <a href="#2-train-and-export">Train and export</a> &middot;
  <a href="#3-configure-and-flash-the-assistant">Flash</a>
</p>

---

## How it works

```text
Build: INMP441 microphone -> ESP32-S3 recorder -> WAV clips -> Python training -> voice_model.h
Run:   INMP441 microphone -> ESP32-S3 inference -> wake phrase -> recognized command
```

- **Record** labeled audio over USB using the ESP32-S3 and INMP441.
- **Train** a custom wake phrase and two commands in your chosen language, then evaluate on separate test recordings.
- **Deploy** the exported TensorFlow Lite model as an Arduino C header. The ESP32-S3 runs recognition on the board.

The model scores overlapping two-second audio windows. When it detects your wake phrase, the firmware opens a command window. The serial monitor shows recognition scores and command results.

| Dataset label | What to record |
| --- | --- |
| `on` | Your phrase for the first command |
| `off` | Your phrase for the second command |
| `other` | Room noise, unrelated speech and phrases that sound similar |
| `wake` | A wake phrase of your choice |

These four identifiers define the model output order. Speak any language and keep the identifiers as shown. Recognition quality depends on the voices, pronunciation and acoustic conditions represented in your recordings.

## Hardware and software

- ESP32-S3 N16R8 with 16 MB flash and 8 MB OPI PSRAM
- INMP441 I²S microphone, USB data cable and optional buzzer on GPIO 7
- Arduino IDE with Arduino-ESP32 (development version used: 3.3.7)
- TensorFlow Lite Micro Arduino library exposing `tensorflow/lite/micro/micro_interpreter.h` (development installation: `TFLiteMicro_ArduinoESP32S3`)
- Python with TensorFlow, NumPy and pyserial for recording and training

| INMP441 pin | ESP32-S3 |
| --- | --- |
| VDD | 3.3 V |
| GND and L/R | GND |
| SCK | GPIO 4 |
| WS | GPIO 5 |
| SD | GPIO 6 |

## 1. Collect your recordings

Flash `training/AudioRecorder/AudioRecorder.ino` using **ESP32S3 Dev Module**. Close the Arduino serial monitor. From PowerShell in the repository root, install the Python dependencies and collect clips. Replace COM8 with your serial port.

```powershell
cd training
python -m pip install -r requirements.txt
foreach ($label in 'on','off','other','wake') {
    python record.py COM8 train $label 30
}
foreach ($label in 'on','off','other','wake') {
    python record.py COM8 test $label 5
}
```

Speak after the countdown. The recorder saves mono, 16-bit PCM at 16 kHz for two seconds. Record the test clips in a separate session. A second speaker or a different microphone distance makes a useful test if that matches your intended use.

## 2. Train and export

From the same training directory:

```powershell
python train.py --output model_output
```

The script trains on `dataset/train`, evaluates the exported model on `dataset/test` and prints a confusion matrix. It also writes the Arduino header to **`model_output/voice_model.h`**. Inspect missed wakes and reversed commands before installing the model. For later training, use `--resume model_output/voice.keras` and a fresh output directory.

## 3. Configure and flash the assistant

```powershell
Copy-Item model_output/voice_model.h ../firmware/VoiceControl/
Copy-Item ../firmware/VoiceControl/secrets.example.h ../firmware/VoiceControl/secrets.h
```

Edit `secrets.h` with your Wi-Fi credentials and the two local device URLs. Open `firmware/VoiceControl/VoiceControl.ino`, select **ESP32S3 Dev Module**, **16 MB flash** and **OPI PSRAM**, then upload. For native USB serial, enable **USB CDC On Boot**.

Open the serial monitor at **115200 baud**. Wait for Wi-Fi, speak your wake phrase, wait for `LISTENING`, then speak a command. Test several wake–command sequences and listen for false activations during ordinary conversation.

## Repository layout

```text
Esp32_Ai/
├── firmware/
│   └── VoiceControl/       # Arduino inference sketch and configuration template
├── training/
│   ├── AudioRecorder/     # ESP32-S3 USB audio recorder sketch
│   ├── record.py          # Labeled WAV collection
│   ├── train.py           # Training, evaluation and Arduino header export
│   └── requirements.txt
└── README.md
```

Each installation supplies its own trained header. Git ignores local credentials, recordings and default model output directories.
