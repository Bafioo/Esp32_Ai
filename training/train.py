import argparse
import csv
import re
import wave
from pathlib import Path

import numpy as np
import tensorflow as tf

LABELS = ("on", "off", "other", "wake")
RATE = 16000
SAMPLES = 32000
INPUT_SHAPE = (1, 99, 29, 1)
EDGES = (2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 17, 19, 23, 26,
         31, 36, 42, 48, 56, 66, 76, 89, 103, 120, 140, 163, 190, 221, 257)
WINDOW = np.hanning(400).astype(np.float32)
FRAME_INDICES = np.arange(99)[:, None] * 320 + np.arange(400)


def features(samples):
    centered = samples - samples.mean()
    normalized = centered * (0.06 / max(np.max(np.abs(centered)), 0.008))
    frames = normalized[FRAME_INDICES] * WINDOW
    power = np.abs(np.fft.rfft(frames, n=512)) ** 2
    bands = np.stack(
        [power[:, a:b].mean(axis=1) for a, b in zip(EDGES, EDGES[1:])],
        axis=-1,
    )
    return (np.log10(bands + 1e-6) / 6).astype(np.float32)[..., None]


def read_split(root, split, copies, rng):
    inputs, labels, filenames = [], [], []
    for index, label in enumerate(LABELS):
        folder = root / split / label
        files = sorted(folder.glob("*.wav"))
        if not files:
            raise ValueError(f"No WAV files in {folder}")
        print(f"{split}/{label}: {len(files)} WAV", flush=True)
        for file in files:
            try:
                with wave.open(str(file), "rb") as recording:
                    if recording.getparams()[:4] != (1, 2, RATE, SAMPLES):
                        raise ValueError("expected mono 16-bit PCM at 16 kHz for two seconds")
                    pcm = np.frombuffer(recording.readframes(SAMPLES), dtype="<i2")
                if pcm.size != SAMPLES:
                    raise ValueError("incomplete audio")
            except (wave.Error, EOFError, ValueError) as exc:
                raise ValueError(f"{file}: {exc}") from exc

            signal = pcm.astype(np.float32) / 32768
            inputs.append(features(signal))
            labels.append(index)
            filenames.append(str(file.relative_to(root)))
            for _ in range(copies):
                shift = int(rng.integers(-2400, 2401))
                shifted = np.roll(signal, shift)
                if shift > 0:
                    shifted[:shift] = 0
                elif shift < 0:
                    shifted[shift:] = 0
                noise = rng.normal(0, rng.uniform(0, 0.0007), SAMPLES)
                inputs.append(features((shifted * rng.uniform(0.65, 1.35) + noise)
                                       .astype(np.float32)))
                labels.append(index)
    return np.stack(inputs), np.array(labels), filenames


def build_model():
    return tf.keras.Sequential([
        tf.keras.Input(shape=INPUT_SHAPE[1:]),
        tf.keras.layers.Conv2D(8, (5, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((4, 2)),
        tf.keras.layers.Conv2D(12, (5, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(len(LABELS), activation="softmax"),
    ])


def evaluate(binary, inputs):
    interpreter = tf.lite.Interpreter(model_content=binary, num_threads=1)
    interpreter.allocate_tensors()
    source = interpreter.get_input_details()[0]
    result = interpreter.get_output_details()[0]
    if (tuple(source["shape"]) != INPUT_SHAPE or source["dtype"] != np.float32
            or tuple(result["shape"]) != (1, len(LABELS))
            or result["dtype"] != np.float32):
        raise ValueError("Incompatible TFLite interface.")
    scores = []
    for sample in inputs:
        interpreter.set_tensor(source["index"], sample[None])
        interpreter.invoke()
        scores.append(interpreter.get_tensor(result["index"])[0])
    return np.stack(scores)


def print_metrics(name, actual, scores):
    predicted = scores.argmax(axis=1)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    np.add.at(matrix, (actual, predicted), 1)
    print(f"\n{name}: {np.trace(matrix)}/{len(actual)} correct")
    print("Rows: actual. Columns: predicted in order", ", ".join(LABELS))
    for label, row in zip(LABELS, matrix):
        print(f"{label:12} {row.tolist()}")
    print("False wake predictions:", matrix[:3, 3].sum())
    print("Missed wakes:", matrix[3, :3].sum())
    print("Reversed commands:", matrix[0, 1] + matrix[1, 0])


def read_baseline(path):
    if path.suffix.lower() != ".h":
        return path.read_bytes()
    match = re.search(r"\bvoice_model\s*\[\s*\]\s*=\s*\{([^}]*)\}",
                      path.read_text(encoding="utf-8-sig"), re.S)
    if not match:
        raise ValueError(f"{path}: array voice_model missing")
    return bytes(int(value, 16) for value in re.findall(r"0x[0-9a-fA-F]{2}", match[1]))


def write_header(path, binary):
    with path.open("w", encoding="ascii") as file:
        file.write("#pragma once\n#include <stdint.h>\n\n")
        file.write("alignas(16) const uint8_t voice_model[] = {\n")
        for start in range(0, len(binary), 16):
            row = ", ".join(f"0x{byte:02x}" for byte in binary[start:start + 16])
            file.write(f"  {row},\n")
        file.write("};\n")


def main():
    parser = argparse.ArgumentParser(description="Train a keyword classifier on local WAV files.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("model_output"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=Path, help="Compatible Keras checkpoint")
    parser.add_argument("--compare", type=Path, help="Baseline .h or .tflite")
    args = parser.parse_args()
    epochs = args.epochs if args.epochs is not None else (12 if args.resume else 30)
    if epochs < 1:
        parser.error("--epochs must be positive")
    if args.output.exists():
        parser.error("--output already exists; choose a new directory")

    tf.keras.utils.set_random_seed(77)
    rng = np.random.default_rng(12)
    train, labels, _ = read_split(args.dataset, "train", 5, rng)
    test, actual, names = read_split(args.dataset, "test", 0, rng)
    baseline = evaluate(read_baseline(args.compare), test) if args.compare else None

    model = tf.keras.models.load_model(args.resume, compile=False) if args.resume else build_model()
    if (tuple(model.input_shape[1:]) != INPUT_SHAPE[1:]
            or model.output_shape[-1] != len(LABELS)):
        raise ValueError("Incompatible checkpoint dimensions.")
    counts = np.bincount(labels, minlength=len(LABELS))
    weights = {i: len(labels) / (len(LABELS) * int(n)) for i, n in enumerate(counts)}
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0003 if args.resume else 0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, labels, batch_size=16, epochs=epochs, class_weight=weights, verbose=2)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    binary = converter.convert()
    scores = evaluate(binary, test)
    if baseline is not None:
        print_metrics("Baseline", actual, baseline)
    print_metrics("Candidate", actual, scores)

    args.output.mkdir(parents=True)
    model.save(args.output / "voice.keras")
    (args.output / "voice.tflite").write_bytes(binary)
    write_header(args.output / "voice_model.h", binary)
    with (args.output / "test_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["wav", "actual", "prediction", *LABELS]
        if baseline is not None:
            header += ["baseline_prediction", *(f"baseline_{label}" for label in LABELS)]
        writer.writerow(header)
        for i, (name, label, row) in enumerate(zip(names, actual, scores)):
            values = [name, LABELS[label], LABELS[row.argmax()], *row.tolist()]
            if baseline is not None:
                values += [LABELS[baseline[i].argmax()], *baseline[i].tolist()]
            writer.writerow(values)
    print(f"\nFiles saved in {args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
