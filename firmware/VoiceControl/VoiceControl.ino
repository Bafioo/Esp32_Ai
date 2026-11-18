#include <Arduino.h>
#include <ESP_I2S.h>
#include <HTTPClient.h>
#include <WiFi.h>
#include <math.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "voice_model.h"
#include "secrets.h"







constexpr int Rate = 16000;
constexpr uint32_t CommandTimeoutMs = 7000;
constexpr int Samples = 32000;
constexpr int Block = 1600;
constexpr int Window = 400;
constexpr int FFT = 512;
constexpr int Frames = 99;
constexpr int Bands = 29;
constexpr int Edges[Bands + 1] = {
  2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 17, 19, 23, 26, 31, 36, 42,
  48, 56, 66, 76, 89, 103, 120, 140, 163, 190, 221, 257
};
constexpr int ArenaBytes = 500 * 1024;
const char* const Labels[] = {"on", "off", "other", "wake"};

I2SClass mic;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

int16_t* audio = nullptr;
int32_t raw[Block];
int16_t* ring = nullptr;
SemaphoreHandle_t audioLock = nullptr;
uint32_t captured = 0;
uint32_t previousCapture = 0;
uint32_t commandStart = 0;
float latestPeak = 0;
int cursor = 0;
int filled = 0;

float hann[Window];
float realPart[FFT], imagPart[FFT], power[FFT / 2 + 1];
float twiddleCos[FFT / 2], twiddleSin[FFT / 2];

bool armed = false;
uint32_t armedAt = 0;
uint32_t rearmedAt = 0;
bool coolingDown = false;
bool wifiWasConnected = false;

void returnToWake() {
  armed = false;
  rearmedAt = millis();
  coolingDown = true;
}

void setShelly(bool turnOn) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Shelly: offline. Command skipped.");
    return;
  }

  WiFiClient client;
  HTTPClient http;
  http.setConnectTimeout(1500);
  http.setTimeout(1500);

  if (!http.begin(client, turnOn ? ShellyOnUrl : ShellyOffUrl)) {
    Serial.println("Shelly: could not start HTTP request.");
    return;
  }

  const int status = http.GET();
  http.end();

  if (status >= 200 && status < 300) {
    Serial.printf("Shelly: %s request succeeded (HTTP %d).\n",
                  turnOn ? "on" : "off", status);
  } else {
    Serial.printf("Shelly: request failed, code %d.\n", status);
  }
}

void spectrum() {
  for (int i = 1, j = 0; i < FFT; ++i) {
    int bit = FFT / 2;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;

    if (i < j) {
      const float tmp = realPart[i];
      realPart[i] = realPart[j];
      realPart[j] = tmp;
    }
  }

  for (int len = 2; len <= FFT; len *= 2) {
    for (int base = 0; base < FFT; base += len) {
      for (int j = 0; j < len / 2; ++j) {
        const int a = base + j;
        const int b = a + len / 2;
        const int k = j * (FFT / len);
        const float r = realPart[b] * twiddleCos[k] -
                        imagPart[b] * twiddleSin[k];
        const float im = realPart[b] * twiddleSin[k] +
                         imagPart[b] * twiddleCos[k];

        realPart[b] = realPart[a] - r;
        imagPart[b] = imagPart[a] - im;
        realPart[a] += r;
        imagPart[a] += im;
      }
    }
  }

  for (int i = 0; i <= FFT / 2; ++i) {
    power[i] = realPart[i] * realPart[i] + imagPart[i] * imagPart[i];
  }
}

bool recognize() {
  float mean = 0;
  for (int i = 0; i < Samples; ++i) {
    mean += audio[i];
  }
  mean /= Samples;

  float peak = 0;
  for (int i = 0; i < Samples; ++i) {
    peak = fmaxf(peak, fabsf(audio[i] - mean));
  }
  const float gain = 0.06f / fmaxf(peak, 0.008f * 32768.0f);

  for (int frame = 0; frame < Frames; ++frame) {
    memset(realPart, 0, sizeof(realPart));
    memset(imagPart, 0, sizeof(imagPart));
    const int start = frame * 320;

    for (int i = 0; i < Window; ++i) {
      realPart[i] = (audio[start + i] - mean) * gain * hann[i];
    }

    spectrum();

    for (int band = 0; band < Bands; ++band) {
      float sum = 0;
      for (int k = Edges[band]; k < Edges[band + 1]; ++k) {
        sum += power[k];
      }
      const float average = sum / (Edges[band + 1] - Edges[band]);
      input->data.f[frame * Bands + band] =
          log10f(average + 1e-6f) / 6.0f;
    }
  }

  return interpreter->Invoke() == kTfLiteOk;
}

bool startModel() {
  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddShape();
  resolver.AddStridedSlice();
  resolver.AddPack();
  resolver.AddRelu();

  auto* arena = static_cast<uint8_t*>(ps_malloc(ArenaBytes));
  audio = static_cast<int16_t*>(ps_malloc(Samples * sizeof(int16_t)));
  ring = static_cast<int16_t*>(ps_malloc(Samples * sizeof(int16_t)));

  if (!arena || !audio || !ring) {
    Serial.println("PSRAM unavailable. Enable OPI PSRAM.");
    return false;
  }

  const auto* model = tflite::GetModel(voice_model);
  if (model->version() != 3) {
    Serial.println("Unsupported TFLite schema.");
    return false;
  }

  static tflite::MicroInterpreter instance(model, resolver, arena, ArenaBytes);
  if (instance.AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed.");
    return false;
  }

  interpreter = &instance;
  input = instance.input(0);
  output = instance.output(0);

  return input->type == kTfLiteFloat32 &&
         output->type == kTfLiteFloat32 &&
         input->bytes == Frames * Bands * sizeof(float) &&
         output->bytes == 4 * sizeof(float);
}

void captureAudio(void*) {
  for (;;) {
    const size_t bytes =
        mic.readBytes(reinterpret_cast<char*>(raw), sizeof(raw));

    if (bytes != sizeof(raw)) {
      xSemaphoreTake(audioLock, portMAX_DELAY);
      filled = 0;
      xSemaphoreGive(audioLock);
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    xSemaphoreTake(audioLock, portMAX_DELAY);

    int peak = 0;
    for (int32_t sample : raw) {
      const int16_t pcm = static_cast<int16_t>(sample >> 16);
      peak = max(peak, abs(static_cast<int>(pcm)));
      ring[cursor++] = pcm;
      if (cursor == Samples) {
        cursor = 0;
      }
    }

    latestPeak = peak / 32768.0f;
    filled = min(filled + Block, Samples);
    ++captured;
    xSemaphoreGive(audioLock);
  }
}

constexpr int BuzzerPin = 7;
constexpr bool ActiveBuzzer = true;
constexpr uint32_t BeepDurationMs = 100;

void beep() {
  if (ActiveBuzzer) {
    digitalWrite(BuzzerPin, HIGH);
    delay(BeepDurationMs);
    digitalWrite(BuzzerPin, LOW);
  } else {
    tone(BuzzerPin, 2300, BeepDurationMs);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(BuzzerPin, OUTPUT);
  digitalWrite(BuzzerPin, LOW);
  delay(1500);

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WifiSsid, WifiPassword);

  Serial.print("Connecting to Wi-Fi");
  uint32_t lastRetry = millis();

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print('.');

    if (millis() - lastRetry >= 15000) {
      Serial.println();
      Serial.print("Retrying Wi-Fi connection");
      WiFi.reconnect();
      lastRetry = millis();
    }
  }

  wifiWasConnected = true;
  Serial.println();
  Serial.print("Connected. ESP32 IP: ");
  beep();
  delay(500);
  beep();
  Serial.println(WiFi.localIP());

  if (!startModel()) {
    Serial.println("Model initialization failed.");
    while (true) {
      delay(1000);
    }
  }

  for (int i = 0; i < Window; ++i) {
    hann[i] = 0.5f - 0.5f * cosf(2 * PI * i / (Window - 1));
  }

  for (int i = 0; i < FFT / 2; ++i) {
    const float phase = -2 * PI * i / FFT;
    twiddleCos[i] = cosf(phase);
    twiddleSin[i] = sinf(phase);
  }

  mic.setPins(4, 5, -1, 6);
  if (!mic.begin(I2S_MODE_STD, Rate, I2S_DATA_BIT_WIDTH_32BIT,
                 I2S_SLOT_MODE_MONO, I2S_STD_SLOT_LEFT)) {
    Serial.println("I2S initialization failed.");
    while (true) {
      delay(1000);
    }
  }

  audioLock = xSemaphoreCreateMutex();
  if (!audioLock ||
      xTaskCreatePinnedToCore(captureAudio, "voice-i2s", 4096, nullptr,
                              3, nullptr, 0) != pdPASS) {
    Serial.println("Audio task failed to start.");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("Ready. Say your wake phrase and wait for LISTENING.");
}

void loop() {
  const bool wifiConnected = WiFi.status() == WL_CONNECTED;
  if (wifiConnected != wifiWasConnected) {
    wifiWasConnected = wifiConnected;
    if (wifiConnected) {
      Serial.print("Connected. ESP32 IP: ");
      Serial.println(WiFi.localIP());
    } else {
      Serial.println("Wi-Fi disconnected. Reconnecting.");
    }
  }

  if (armed && millis() - armedAt >= CommandTimeoutMs) {
    returnToWake();
    Serial.println("Command timed out. Waiting for wake phrase.");
  }
  if (coolingDown && millis() - rearmedAt >= 300) {
    coolingDown = false;
  }

  xSemaphoreTake(audioLock, portMAX_DELAY);
  const uint32_t sequence = captured;

  if (filled != Samples || sequence == previousCapture ||
      (armed && sequence - commandStart < Samples / Block)) {
    xSemaphoreGive(audioLock);
    delay(10);
    return;
  }

  const float blockPeak = latestPeak;
  const int tail = Samples - cursor;
  memcpy(audio, ring + cursor, tail * sizeof(*audio));
  memcpy(audio + tail, ring, cursor * sizeof(*audio));
  previousCapture = sequence;
  xSemaphoreGive(audioLock);

  const uint32_t analysisStart = millis();
  if (!recognize()) {
    Serial.println("Inference failed.");
    return;
  }

  const float* scores = output->data.f;
  int best = 0;
  for (int i = 1; i < 4; ++i) {
    if (scores[i] > scores[best]) {
      best = i;
    }
  }

  Serial.printf(
      "%s %s | peak %.4f | analysis %lu ms | on %.2f off %.2f other %.2f wake %.2f\n",
      armed ? "COMMAND" : "IDLE", Labels[best], blockPeak,
      static_cast<unsigned long>(millis() - analysisStart),
      scores[0], scores[1], scores[2], scores[3]);

  static bool hadWakeEvidence = false;
  static bool hadCloseWake = false;
  static uint32_t evidenceSequence = 0;

  if (!armed && !coolingDown) {
    const float command = fmaxf(scores[0], scores[1]);
    const bool clearWake = best == 3 && scores[3] >= 0.30f;
    const bool closeWake = best == 2 && scores[3] >= 0.32f &&
                           scores[2] - scores[3] <= 0.18f &&
                           scores[3] - command >= 0.15f;
    const bool supportingWake = scores[3] >= 0.18f &&
                                scores[3] - command >= 0.10f;

    const uint32_t gap = sequence - evidenceSequence;
    const bool confirmed = hadWakeEvidence && gap > 0 &&
                           gap <= Rate / (2 * Block) &&
                           supportingWake && (closeWake || hadCloseWake);

    if (!clearWake && !confirmed) {
      if (supportingWake) {
        hadWakeEvidence = true;
        hadCloseWake = closeWake;
        evidenceSequence = sequence;
      }
      return;
    }

    hadWakeEvidence = false;
    hadCloseWake = false;

    if (confirmed && !clearWake) {
      Serial.println("Wake confirmed across two windows.");
    }

    armed = true;
    armedAt = millis();

    xSemaphoreTake(audioLock, portMAX_DELAY);
    commandStart = captured;
    xSemaphoreGive(audioLock);

    Serial.println("LISTENING: say a command.");
    beep();
  } else if (armed && (best == 0 || best == 1) && scores[best] >= 0.40f) {
    if (millis() - armedAt >= CommandTimeoutMs) {
      return;
    }

    Serial.printf("Recognized command: %s.\n", Labels[best]);
    
    setShelly(best == 0);
    returnToWake();
    Serial.println("Waiting for wake phrase.");
  }
}
