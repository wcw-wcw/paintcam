from __future__ import annotations

from pathlib import Path
from typing import Any

from .gestures import Hand, clamp

DEFAULT_HAND_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def resolve_hand_model_path(path: str | Path | None = None) -> Path:
    return Path(path).expanduser().resolve() if path else DEFAULT_HAND_MODEL_PATH


def model_missing_error(path: str | Path) -> dict[str, str]:
    resolved = str(resolve_hand_model_path(path))
    message = (
        f"Hand Landmarker model not found at {resolved}. "
        "MediaPipe Tasks Hand Landmarker requires a .task model bundle; "
        "download it to that location or pass --hand-model /path/to/hand_landmarker.task."
    )
    return {
        "code": "hand_landmarker_model_missing",
        "message": message,
        "last_error": message,
        "model_path": resolved,
    }


def adapt_hand_landmarker_result(result: Any) -> list[Hand]:
    landmarks_by_hand = getattr(result, "hand_landmarks", None) or []
    handedness_by_hand = getattr(result, "handedness", None) or []
    hands: list[Hand] = []
    for index, landmarks in enumerate(landmarks_by_hand):
        category = None
        if index < len(handedness_by_hand) and handedness_by_hand[index]:
            category = handedness_by_hand[index][0]
        label = (
            getattr(category, "category_name", None)
            or getattr(category, "display_name", None)
            if category is not None
            else None
        )
        score = getattr(category, "score", None) if category is not None else None
        hands.append(
            Hand(
                [
                    (
                        clamp(float(point.x), 0.0, 1.0),
                        clamp(float(point.y), 0.0, 1.0),
                    )
                    for point in landmarks
                ],
                handedness=label,
                handedness_score=float(score) if score is not None else None,
            )
        )
    return hands


class MediaPipeHandTracker:
    def __init__(self, mp: Any, model_path: str | Path) -> None:
        # Keep these imports explicit: this engine uses Tasks, never legacy mp.solutions.
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, rgb_frame: Any, timestamp_ms: int) -> list[Hand]:
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return adapt_hand_landmarker_result(result)

    def close(self) -> None:
        self._landmarker.close()
