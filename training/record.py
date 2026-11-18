import argparse
import time
import uuid
import wave
from pathlib import Path

import serial


def main():
    parser = argparse.ArgumentParser(description="Record WAV clips from the INMP441.")
    parser.add_argument("port")
    parser.add_argument("split", choices=("train", "test"))
    parser.add_argument("label", choices=("on", "off", "other", "wake"))
    parser.add_argument("count", type=int)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")

    folder = args.dataset / args.split / args.label
    folder.mkdir(parents=True, exist_ok=True)
    with serial.Serial(args.port, 921600, timeout=8) as port:
        time.sleep(2)
        for number in range(1, args.count + 1):
            choice = input(f"[{number}/{args.count}] Enter to record, q to quit: ")
            if choice.strip().lower() == "q":
                return
            for tick in (3, 2, 1):
                print(tick, flush=True)
                time.sleep(1)
            port.reset_input_buffer()
            port.write(b"R")
            time.sleep(0.15)
            print("SPEAK", flush=True)
            response = port.readline().strip()
            if response != b"WAV_PCM":
                raise SystemExit(f"Recorder response: {response!r}")
            pcm = port.read(64000)
            if len(pcm) != 64000:
                raise SystemExit(f"Incomplete audio: {len(pcm)}/64000 byte")
            filename = folder / f"{args.label}_{uuid.uuid4().hex[:12]}.wav"
            with filename.open("xb") as file, wave.open(file, "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(pcm)
            print(f"Saved: {filename}")


if __name__ == "__main__":
    main()
