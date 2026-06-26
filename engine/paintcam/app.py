from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Iterable, Protocol

try:
    import cv2
    import numpy as np
except ImportError as error:  # pragma: no cover - environment setup guard
    print(
        "PaintCam needs OpenCV and NumPy. Install with: python3 -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from error

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - optional at import time
    mp = None

try:
    import pyvirtualcam
except ImportError:  # pragma: no cover - optional output
    pyvirtualcam = None


Color = tuple[int, int, int]
Point = tuple[int, int]


@dataclass
class EngineConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    palette_height: int = 92
    brush_radius: int = 8
    min_zoom: float = 1.0
    max_zoom: float = 2.8
    preview: bool = True
    virtual_camera: bool = True
    palette: list[Color] = field(
        default_factory=lambda: [
            (36, 36, 36),
            (255, 255, 255),
            (52, 120, 246),
            (49, 196, 141),
            (250, 204, 21),
            (245, 101, 101),
            (168, 85, 247),
        ]
    )


@dataclass
class Hand:
    landmarks: list[tuple[float, float]]

    def point(self, index: int, width: int, height: int) -> Point:
        x, y = self.landmarks[index]
        return int(x * width), int(y * height)


@dataclass
class FrameContext:
    frame: "np.ndarray"
    canvas: "np.ndarray"
    hands: list[Hand]
    config: EngineConfig
    selected_color: Color
    zoom: float


class Gesture(Protocol):
    name: str

    def update(self, context: FrameContext) -> None:
        ...


class MediaPipeHandTracker:
    def __init__(self) -> None:
        if mp is None:
            self._hands = None
            return

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame: "np.ndarray") -> list[Hand]:
        if self._hands is None:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []

        return [
            Hand([(landmark.x, landmark.y) for landmark in hand.landmark])
            for hand in result.multi_hand_landmarks
        ]


class PaletteSelectGesture:
    name = "palette_select"

    def __init__(self, engine: "PaintCamEngine") -> None:
        self.engine = engine

    def update(self, context: FrameContext) -> None:
        hand = first_hand(context.hands)
        if hand is None:
            return

        height, width = context.frame.shape[:2]
        index_tip = hand.point(8, width, height)
        palette_top = height - context.config.palette_height
        if index_tip[1] < palette_top:
            return

        swatch_width = width / len(context.config.palette)
        selected = min(int(index_tip[0] / swatch_width), len(context.config.palette) - 1)
        self.engine.selected_color = context.config.palette[max(0, selected)]


class PinchDrawGesture:
    name = "pinch_draw"

    def __init__(self, engine: "PaintCamEngine") -> None:
        self.engine = engine
        self.previous_point: Point | None = None

    def update(self, context: FrameContext) -> None:
        hand = first_hand(context.hands)
        if hand is None:
            self.previous_point = None
            return

        height, width = context.frame.shape[:2]
        palette_top = height - context.config.palette_height
        thumb = hand.point(4, width, height)
        index = hand.point(8, width, height)

        if index[1] >= palette_top or normalized_distance(hand, 4, 8) > 0.055:
            self.previous_point = None
            return

        if self.previous_point is not None:
            cv2.line(
                context.canvas,
                self.previous_point,
                index,
                self.engine.selected_color,
                context.config.brush_radius * 2,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.circle(
                context.canvas,
                index,
                context.config.brush_radius,
                self.engine.selected_color,
                -1,
                lineType=cv2.LINE_AA,
            )

        self.previous_point = index


class TwoHandZoomGesture:
    name = "two_hand_zoom"

    def __init__(self, engine: "PaintCamEngine") -> None:
        self.engine = engine
        self.baseline_distance: float | None = None
        self.baseline_zoom: float = 1.0

    def update(self, context: FrameContext) -> None:
        if len(context.hands) < 2:
            self.baseline_distance = None
            self.baseline_zoom = self.engine.zoom
            return

        first = context.hands[0].landmarks[8]
        second = context.hands[1].landmarks[8]
        distance = math.dist(first, second)

        if self.baseline_distance is None:
            self.baseline_distance = max(distance, 0.001)
            self.baseline_zoom = self.engine.zoom
            return

        zoom = self.baseline_zoom * (distance / self.baseline_distance)
        self.engine.zoom = clamp(zoom, context.config.min_zoom, context.config.max_zoom)


class PaintCamEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.selected_color = config.palette[2]
        self.zoom = 1.0
        self.canvas: "np.ndarray | None" = None
        self.tracker = MediaPipeHandTracker()
        self.gestures: list[Gesture] = [
            PaletteSelectGesture(self),
            PinchDrawGesture(self),
            TwoHandZoomGesture(self),
        ]

    def run(self) -> None:
        capture = cv2.VideoCapture(self.config.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read camera index {self.config.camera_index}")

        frame = resize_frame(frame, self.config.width, self.config.height)
        self.canvas = np.zeros_like(frame)

        with virtual_camera_sink(self.config, frame) as virtual_camera:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                frame = resize_frame(frame, self.config.width, self.config.height)
                frame = apply_zoom(frame, self.zoom)
                hands = self.tracker.detect(frame)
                context = FrameContext(
                    frame=frame,
                    canvas=self.canvas,
                    hands=hands,
                    config=self.config,
                    selected_color=self.selected_color,
                    zoom=self.zoom,
                )

                for gesture in self.gestures:
                    gesture.update(context)

                output = compose_output(frame, self.canvas, self.config, self.selected_color)

                if virtual_camera is not None:
                    virtual_camera.send(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                    virtual_camera.sleep_until_next_frame()

                if self.config.preview:
                    cv2.imshow("PaintCam Output", output)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        capture.release()
        cv2.destroyAllWindows()


def virtual_camera_sink(config: EngineConfig, frame: "np.ndarray"):
    class NullSink:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    if not config.virtual_camera or pyvirtualcam is None:
        return NullSink()

    height, width = frame.shape[:2]
    return pyvirtualcam.Camera(width=width, height=height, fps=config.fps)


def compose_output(
    frame: "np.ndarray", canvas: "np.ndarray", config: EngineConfig, selected_color: Color
) -> "np.ndarray":
    output = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)
    draw_palette(output, config, selected_color)
    draw_status(output, selected_color)
    return output


def draw_palette(frame: "np.ndarray", config: EngineConfig, selected_color: Color) -> None:
    height, width = frame.shape[:2]
    top = height - config.palette_height
    swatch_width = width / len(config.palette)
    cv2.rectangle(frame, (0, top), (width, height), (24, 28, 33), -1)

    for index, color in enumerate(config.palette):
        x1 = int(index * swatch_width) + 12
        x2 = int((index + 1) * swatch_width) - 12
        y1 = top + 18
        y2 = height - 18
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        if color == selected_color:
            cv2.rectangle(frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (255, 255, 255), 3)


def draw_status(frame: "np.ndarray", selected_color: Color) -> None:
    cv2.circle(frame, (34, 34), 15, selected_color, -1, lineType=cv2.LINE_AA)
    cv2.putText(
        frame,
        "PaintCam",
        (58, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )


def apply_zoom(frame: "np.ndarray", zoom: float) -> "np.ndarray":
    if zoom <= 1.01:
        return frame

    height, width = frame.shape[:2]
    crop_width = int(width / zoom)
    crop_height = int(height / zoom)
    x1 = (width - crop_width) // 2
    y1 = (height - crop_height) // 2
    cropped = frame[y1 : y1 + crop_height, x1 : x1 + crop_width]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_frame(frame: "np.ndarray", width: int, height: int) -> "np.ndarray":
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def first_hand(hands: Iterable[Hand]) -> Hand | None:
    return next(iter(hands), None)


def normalized_distance(hand: Hand, first: int, second: int) -> float:
    return math.dist(hand.landmarks[first], hand.landmarks[second])


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def parse_args() -> EngineConfig:
    parser = argparse.ArgumentParser(description="Run the PaintCam gesture video engine.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-virtual-camera", action="store_true")
    args = parser.parse_args()

    return EngineConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        preview=not args.no_preview,
        virtual_camera=not args.no_virtual_camera,
    )


def main() -> None:
    engine = PaintCamEngine(parse_args())
    engine.run()
