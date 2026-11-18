<p align="center">
  <img src="docs/logo.svg" width="112" height="112" alt="">
</p>

<h1 align="center">ESP32-S3 Keyword Spotting</h1>

<p align="center">
  <a href="#hardware-and-software"><img src="https://img.shields.io/badge/hardware-ESP32--S3-201d1d?style=flat-square" alt="Hardware: ESP32-S3"></a>
  <a href="#2-train-and-export"><img src="https://img.shields.io/badge/training-Python-201d1d?style=flat-square" alt="Training: Python"></a>
  <a href="#3-flash-and-test"><img src="https://img.shields.io/badge/firmware-Arduino-201d1d?style=flat-square" alt="Firmware: Arduino"></a>
</p>

Train a small speech recognition model to detect a wake phrase and two spoken commands. Record audio with an ESP32-S3 and an INMP441 microphone, train on your computer, and run inference on the board.

## Workflow

1. [Record labeled WAV clips](#1-record-training-and-test-audio).
2. [Train and export the Arduino header](#2-train-and-export).
3. [Flash and test on the ESP32-S3](#3-flash-and-test).

## Hardware and software

- `[+]` **Hardware.** ESP32-S3 N16R8 with 16 MB flash and 8 MB OPI PSRAM, an INMP441 microphone, and a USB data cable.
- `[+]` **Firmware.** Arduino IDE with Arduino-ESP32 (developed with 3.3.7) and a TensorFlow Lite Micro library that provides `tensorflow/lite/micro/micro_interpreter.h`.
- `[+]` **Training.** Python with TensorFlow, NumPy, and pyserial. Install these packages in step 1.

| INMP441 | ESP32-S3 |
| --- | --- |
| VDD | 3.3 V |
| GND and L/R | GND |
| SCK | GPIO 4 |
| WS | GPIO 5 |
| SD | GPIO 6 |

## 1. Record training and test audio

Choose a wake word or short phrase and two distinct commands in any language. Keep the dataset labels below because they define the model's output order.

| Label | What to record |
| --- | --- |
| `on` | The first command |
| `off` | The second command |
| `other` | Background noise, unrelated speech, and similar-sounding phrases |
| `wake` | Your wake word or phrase |

Upload [AudioRecorder.ino](training/AudioRecorder/AudioRecorder.ino) with **ESP32S3 Dev Module** selected. Enable **USB CDC On Boot** if you use native USB serial, then close the Arduino Serial Monitor before recording.

Run these commands in PowerShell from the repository root. Replace `COM8` with your serial port.

```powershell
Set-Location training
python -m pip install -r requirements.txt

python record.py COM8 train on 30
python record.py COM8 train off 30
python record.py COM8 train other 30
python record.py COM8 train wake 30
```

Speak after the countdown. Each WAV clip contains two seconds of mono, 16-bit audio sampled at 16 kHz. Record the test clips in a separate session so they reflect how you actually plan to use the model.

```powershell
python record.py COM8 test on 5
python record.py COM8 test off 5
python record.py COM8 test other 5
python record.py COM8 test wake 5
```

## 2. Train and export

From the `training` directory, run:

```powershell
python train.py --output model_output
```

The script trains on `dataset/train`, evaluates the exported TensorFlow Lite model on `dataset/test`, and prints a confusion matrix. Check for missed wake phrases and commands confused with each other. The Arduino header is saved as **`model_output/voice_model.h`**.

## 3. Flash and test

### Use the generated model header

`voice_model.h` contains the trained TensorFlow Lite model as a C++ byte array named `voice_model`.

Copy the generated file from `training/model_output/voice_model.h` to **`firmware/VoiceControl/voice_model.h`**. Both paths are relative to the repository root. Place it in the same folder as `VoiceControl.ino`.

```text
firmware/VoiceControl/
├── VoiceControl.ino
├── voice_model.h
└── secrets.h
```

If your PowerShell session is still in the `training` directory, run:

```powershell
Copy-Item model_output/voice_model.h ../firmware/VoiceControl/voice_model.h
```

The sketch already includes the header. During setup, TensorFlow Lite Micro reads the `voice_model` array, which is compiled into the firmware.

```cpp
#include "voice_model.h"

// In the sketch's model setup:
const auto* model = tflite::GetModel(voice_model);
```

After retraining, replace the header with the new export, then compile and upload the sketch again to update the model on the board.

### Create the local secrets header

Copy [secrets.example.h](firmware/VoiceControl/secrets.example.h) to **`firmware/VoiceControl/secrets.h`**, next to `VoiceControl.ino` and `voice_model.h`. If you already have a `secrets.h`, edit that file directly. Otherwise, run this command from the `training` directory:

```powershell
Copy-Item ../firmware/VoiceControl/secrets.example.h ../firmware/VoiceControl/secrets.h
```

Open `secrets.h` in a text editor and replace the values inside the quotes:

- `WifiSsid` and `WifiPassword` are your Wi-Fi network name and password.
- `ShellyOnUrl` and `ShellyOffUrl` are the local HTTP URLs called when the sketch recognizes the `on` and `off` commands.

The sketch already includes `secrets.h`. It uses these values to connect to Wi-Fi and send the command requests. If you change a value, compile and upload the sketch again. Keep your credentials in the Git-ignored `secrets.h`.

### Upload the sketch

Open [VoiceControl.ino](firmware/VoiceControl/VoiceControl.ino), select **ESP32S3 Dev Module**, **16 MB flash**, and **OPI PSRAM**, then upload. Enable **USB CDC On Boot** for native USB serial.

Open the Serial Monitor at **115200 baud**. Say the wake phrase, wait for `LISTENING`, and say a command. Check the scores for both commands and watch for false activations during normal conversation.

Git ignores the local configuration, WAV recordings, datasets, and default model output directories.
