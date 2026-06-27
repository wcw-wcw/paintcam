#!/usr/bin/env python3
from pathlib import Path
from urllib.request import urlopen

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DESTINATION = Path(__file__).resolve().parents[1] / "engine/models/hand_landmarker.task"


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".task.download")
    print(f"Downloading {MODEL_URL}")
    try:
        with urlopen(MODEL_URL) as response, temporary.open("wb") as output:
            output.write(response.read())
        temporary.replace(DESTINATION)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved Hand Landmarker model to {DESTINATION}")


if __name__ == "__main__":
    main()
