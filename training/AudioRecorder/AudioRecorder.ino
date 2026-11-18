#include <Arduino.h>
#include <ESP_I2S.h>

constexpr int Rate = 16000;
constexpr int BlockSize = 256;
constexpr int SampleCount = Rate * 2;
I2SClass microphone;
int32_t block[BlockSize];
int16_t recording[SampleCount];
int position = 0;
bool recordingActive = false;

void setup() {
  Serial.begin(921600);
  microphone.setPins(4, 5, -1, 6);
  if (!microphone.begin(I2S_MODE_STD, Rate, I2S_DATA_BIT_WIDTH_32BIT,
                       I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) {
    Serial.println("ERROR I2S");
    while (true) delay(1000);
  }
}

void loop() {
  while (Serial.available()) {
    if (Serial.read() == 'R' && !recordingActive) {
      position = 0;
      recordingActive = true;
    }
  }
  const size_t bytesRead = microphone.readBytes(
      reinterpret_cast<char*>(block), sizeof(block));
  if (bytesRead != sizeof(block)) {
    if (recordingActive) Serial.println("ERROR I2S");
    recordingActive = false;
    delay(10);
    return;
  }
  if (!recordingActive) return;

  for (int32_t sample : block) {
    recording[position++] = static_cast<int16_t>(sample >> 16);
  }

  if (position == SampleCount) {
    Serial.println("WAV_PCM");
    Serial.write(reinterpret_cast<const uint8_t*>(recording),
                 sizeof(recording));
    recordingActive = false;
  }
}
