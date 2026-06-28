from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from .gestures import (
    Color,
    DrawingState,
    GestureContext,
    GestureResult,
    GestureTuning,
    Hand,
    PaletteState,
    ZoomState,
    clamp,
    default_registry,
    normalized_distance,
    palette_index,
    palette_top,
    calculate_zoom as tuned_calculate_zoom,
)
from .hand_tracking import (
    DEFAULT_HAND_MODEL_PATH,
    HAND_CONNECTIONS,
    MediaPipeHandTracker,
    model_missing_error,
    resolve_hand_model_path,
)

cv2: Any = None
np: Any = None
mp: Any = None
pyvirtualcam: Any = None


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "timestamp": time.time(), **fields}), flush=True)


def load_dependencies() -> dict[str, bool]:
    global cv2, np, mp, pyvirtualcam
    modules = {
        "opencv": ("cv2", "cv2"),
        "numpy": ("numpy", "np"),
        "mediapipe": ("mediapipe", "mp"),
        "pyvirtualcam": ("pyvirtualcam", "pyvirtualcam"),
    }
    status: dict[str, bool] = {}
    for name, (module_name, target) in modules.items():
        try:
            globals()[target] = importlib.import_module(module_name)
            status[name] = True
        except (ImportError, OSError):
            status[name] = False
    status["mediapipe_tasks"] = bool(
        status["mediapipe"]
        and getattr(mp, "tasks", None)
        and getattr(getattr(mp.tasks, "vision", None), "HandLandmarker", None)
    )
    emit("dependency_status", dependencies=status)
    return status


@dataclass
class EngineConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    preview: bool = True
    virtual_camera: bool = True
    draw_landmarks: bool = False
    debug_overlay: bool = False
    gesture_config_path: str | None = None
    hand_model_path: Path = DEFAULT_HAND_MODEL_PATH
    doctor: bool = False
    list_cameras: bool = False
    tuning: GestureTuning = field(default_factory=GestureTuning)
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


class PaintCamEngine:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.palette = PaletteState(config.palette)
        self.drawing = DrawingState(config.tuning.brush_size)
        self.zoom = ZoomState(1.0)
        self.canvas: Any = None
        self.tracker = MediaPipeHandTracker(mp, config.hand_model_path)
        self.registry = default_registry(config.tuning)

    def apply_gesture(self, result: GestureResult) -> None:
        if not result.active:
            self.drawing.drawing = False
            self.drawing.previous_point = None
            return
        if result.action == "select_color":
            self.palette.selected_index = int(result.data["palette_index"])
            self.drawing.drawing = False
            self.drawing.previous_point = None
        elif result.action == "set_zoom":
            self.zoom.value = float(result.data["zoom"])
            self.drawing.drawing = False
            self.drawing.previous_point = None
        elif result.action == "draw":
            point = tuple(result.data["point"])
            radius = max(1, self.drawing.brush_size // 2)
            if self.drawing.previous_point is None:
                cv2.circle(
                    self.canvas,
                    point,
                    radius,
                    self.palette.selected_color,
                    -1,
                    lineType=cv2.LINE_AA,
                )
            else:
                cv2.line(
                    self.canvas,
                    self.drawing.previous_point,
                    point,
                    self.palette.selected_color,
                    self.drawing.brush_size,
                    lineType=cv2.LINE_AA,
                )
            self.drawing.previous_point = point
            self.drawing.drawing = True

    def run(self) -> None:
        capture = cv2.VideoCapture(self.config.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if not capture.isOpened():
            raise RuntimeError(camera_error(self.config.camera_index))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(camera_error(self.config.camera_index))

        frame = resize_frame(cv2.flip(frame, 1), self.config.width, self.config.height)
        self.canvas = np.zeros_like(frame)
        emit(
            "camera_opened",
            camera_index=self.config.camera_index,
            width=frame.shape[1],
            height=frame.shape[0],
            fps=self.config.fps,
        )
        last_frame_event = 0.0
        last_gesture_event = 0.0
        last_gesture_state: tuple[Any, ...] | None = None
        frame_index = 0
        try:
            with virtual_camera_sink(self.config, frame) as virtual_camera:
                virtual_active = virtual_camera is not None
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(
                            f"Camera index {self.config.camera_index} stopped returning frames."
                        )
                    frame_index += 1
                    frame = resize_frame(
                        cv2.flip(frame, 1), self.config.width, self.config.height
                    )
                    frame = apply_zoom(frame, self.zoom.value)
                    now = time.monotonic()
                    timestamp_ms = int(now * 1000)
                    hands = self.tracker.detect(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), timestamp_ms
                    )
                    context = GestureContext(
                        width=frame.shape[1],
                        height=frame.shape[0],
                        hands=hands,
                        palette=self.palette,
                        drawing=self.drawing,
                        zoom=self.zoom,
                        tuning=self.config.tuning,
                        timestamp_ms=timestamp_ms,
                        frame_index=frame_index,
                    )
                    result, all_results = self.registry.process(context)
                    self.apply_gesture(result)
                    conflicts = [
                        item.conflict or f"{item.gesture_name}:{item.cooldown_remaining_ms}ms"
                        for item in all_results
                        if item.conflict or item.cooldown_remaining_ms
                    ]

                    output = compose_output(
                        frame,
                        self.canvas,
                        self.config,
                        self.palette,
                        self.drawing,
                        self.zoom,
                    )
                    if self.config.draw_landmarks:
                        draw_hand_landmarks(output, hands)
                    if self.config.debug_overlay:
                        draw_debug_overlay(
                            output,
                            result,
                            self.palette,
                            self.drawing,
                            self.zoom,
                            len(hands),
                            virtual_active,
                            conflicts,
                        )
                    if virtual_camera is not None:
                        virtual_camera.send(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                        virtual_camera.sleep_until_next_frame()

                    gesture_state = (
                        result.gesture_name,
                        result.active,
                        round(result.confidence, 2),
                        self.palette.selected_index,
                        self.drawing.brush_size,
                        round(self.zoom.value, 3),
                        len(hands),
                        tuple(conflicts),
                    )
                    changed = gesture_state != last_gesture_state
                    if (changed and now - last_gesture_event >= 0.2) or (
                        now - last_gesture_event >= 1.0
                    ):
                        emit_gesture_state(
                            result,
                            self.palette,
                            self.drawing,
                            self.zoom,
                            len(hands),
                            conflicts,
                        )
                        last_gesture_event = now
                        last_gesture_state = gesture_state
                    if now - last_frame_event >= 1.0:
                        emit(
                            "camera_frame",
                            camera_index=self.config.camera_index,
                            width=output.shape[1],
                            height=output.shape[0],
                            fps=self.config.fps,
                            frame_index=frame_index,
                            hands_detected=len(hands),
                            active_gesture=result.gesture_name
                            if result.active
                            else "none",
                            confidence=round(result.confidence, 3),
                            selected_color=color_hex(self.palette.selected_color),
                            brush_size=self.drawing.brush_size,
                            zoom=round(self.zoom.value, 3),
                            preview_enabled=self.config.preview,
                            virtual_camera_enabled=virtual_active,
                        )
                        last_frame_event = now
                    if self.config.preview:
                        cv2.imshow("PaintCam Output", output)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        finally:
            capture.release()
            self.tracker.close()
            cv2.destroyAllWindows()


def emit_gesture_state(
    result: GestureResult,
    palette: PaletteState,
    drawing: DrawingState,
    zoom: ZoomState,
    hands_detected: int,
    conflicts: list[str],
) -> None:
    emit(
        "gesture_state",
        active_gesture=result.gesture_name if result.active else "none",
        confidence=round(result.confidence, 3),
        action=result.action,
        debug_text=result.debug_text,
        selected_color=color_hex(palette.selected_color),
        brush_size=drawing.brush_size,
        zoom=round(zoom.value, 3),
        hands_detected=hands_detected,
        cooldown_remaining_ms=result.cooldown_remaining_ms,
        conflicts=conflicts,
        consumed_frame=result.consumed_frame,
    )


@contextmanager
def virtual_camera_sink(config: EngineConfig, frame: Any):
    if not config.virtual_camera:
        emit("virtual_camera_status", status="disabled", virtual_camera_enabled=False)
        yield None
        return
    try:
        height, width = frame.shape[:2]
        camera = pyvirtualcam.Camera(width=width, height=height, fps=config.fps)
    except Exception as error:
        message = f"Virtual camera unavailable: {error}. Retry with --no-virtual-camera."
        emit(
            "virtual_camera_status",
            status="unavailable",
            virtual_camera_enabled=False,
            last_error=message,
        )
        print(message, file=sys.stderr)
        yield None
        return
    emit("virtual_camera_status", status="active", virtual_camera_enabled=True)
    try:
        with camera:
            yield camera
    finally:
        emit("virtual_camera_status", status="stopped", virtual_camera_enabled=False)


def compose_output(
    frame: Any,
    canvas: Any,
    config: EngineConfig,
    palette: PaletteState,
    drawing: DrawingState,
    zoom: ZoomState,
) -> Any:
    output = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)
    draw_palette(output, config, palette)
    draw_status(output, palette.selected_color, drawing.brush_size, zoom.value)
    return output


def draw_palette(frame: Any, config: EngineConfig, palette: PaletteState) -> None:
    height, width = frame.shape[:2]
    top = palette_top(height, config.tuning.palette_height_ratio)
    swatch_width = width / len(palette.colors)
    cv2.rectangle(frame, (0, top), (width, height), (24, 28, 33), -1)
    for index, color in enumerate(palette.colors):
        x1, x2 = (
            int(index * swatch_width) + 12,
            int((index + 1) * swatch_width) - 12,
        )
        padding = max(8, int((height - top) * 0.2))
        y1, y2 = top + padding, height - padding
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        if index == palette.selected_index:
            cv2.rectangle(
                frame, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (255, 255, 255), 3
            )


def draw_status(
    frame: Any, selected_color: Color, brush_size: int, zoom: float
) -> None:
    cv2.circle(frame, (34, 34), max(5, brush_size // 2), selected_color, -1)
    cv2.putText(
        frame,
        f"PaintCam  {zoom:.2f}x",
        (58, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )


def draw_debug_overlay(
    frame: Any,
    result: GestureResult,
    palette: PaletteState,
    drawing: DrawingState,
    zoom: ZoomState,
    hands_detected: int,
    virtual_camera_active: bool,
    conflicts: list[str],
) -> None:
    lines = [
        f"Gesture: {result.gesture_name if result.active else 'none'}",
        f"Confidence: {result.confidence:.2f}",
        f"Color: {color_hex(palette.selected_color)}",
        f"Brush: {drawing.brush_size}px",
        f"Zoom: {zoom.value:.2f}x",
        f"Hands: {hands_detected}",
        f"Virtual camera: {'active' if virtual_camera_active else 'inactive'}",
    ]
    if conflicts:
        lines.append(f"Conflict/cooldown: {', '.join(conflicts)}")
    width = min(frame.shape[1] - 24, 620)
    cv2.rectangle(frame, (12, 64), (width, 84 + len(lines) * 25), (16, 20, 24), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (24, 90 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 240, 245),
            1,
            lineType=cv2.LINE_AA,
        )


def draw_hand_landmarks(frame: Any, hands: list[Hand]) -> None:
    height, width = frame.shape[:2]
    for hand in hands:
        points = [hand.point(index, width, height) for index in range(len(hand.landmarks))]
        for first, second in HAND_CONNECTIONS:
            cv2.line(
                frame, points[first], points[second], (80, 220, 120), 2, cv2.LINE_AA
            )
        for point in points:
            cv2.circle(frame, point, 3, (255, 255, 255), -1, cv2.LINE_AA)


def apply_zoom(frame: Any, zoom: float) -> Any:
    if zoom <= 1.01:
        return frame
    height, width = frame.shape[:2]
    crop_width, crop_height = int(width / zoom), int(height / zoom)
    x1, y1 = (width - crop_width) // 2, (height - crop_height) // 2
    return cv2.resize(
        frame[y1 : y1 + crop_height, x1 : x1 + crop_width],
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def resize_frame(frame: Any, width: int, height: int) -> Any:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


# Backward-compatible helper used by the first milestone tests.
def calculate_zoom(
    baseline_zoom: float,
    baseline_distance: float,
    distance: float,
    minimum: float,
    maximum: float,
) -> float:
    return tuned_calculate_zoom(
        baseline_zoom, baseline_distance, distance, 1.0, minimum, maximum
    )


def color_hex(color: Color) -> str:
    blue, green, red = color
    return f"#{red:02x}{green:02x}{blue:02x}"


def camera_error(index: int) -> str:
    return (
        f"Could not open camera index {index}. Check camera permission and availability; "
        f"try --camera-index {index + 1} or --no-virtual-camera."
    )


def parse_args(argv: list[str] | None = None) -> EngineConfig:
    parser = argparse.ArgumentParser(description="Run the PaintCam gesture video engine.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--no-virtual-camera", action="store_true")
    parser.add_argument("--draw-landmarks", action="store_true")
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--brush-size", type=int)
    parser.add_argument("--gesture-config")
    parser.add_argument("--hand-model")
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe camera indexes 0-4 without loading MediaPipe or the hand model.",
    )
    args = parser.parse_args(argv)
    tuning = (
        GestureTuning.from_json(args.gesture_config)
        if args.gesture_config
        else GestureTuning()
    )
    if args.brush_size is not None:
        tuning.brush_size = args.brush_size
    tuning.validate()
    return EngineConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        preview=not args.no_preview,
        virtual_camera=not args.no_virtual_camera,
        draw_landmarks=args.draw_landmarks,
        debug_overlay=args.debug_overlay,
        gesture_config_path=args.gesture_config,
        hand_model_path=resolve_hand_model_path(args.hand_model),
        doctor=args.doctor,
        list_cameras=args.list_cameras,
        tuning=tuning,
    )


def emit_doctor(dependencies: dict[str, bool]) -> None:
    emit(
        "doctor",
        python_executable=sys.executable,
        python_version=sys.version.split()[0],
        opencv_version=getattr(cv2, "__version__", None),
        numpy_version=getattr(np, "__version__", None),
        mediapipe_version=getattr(mp, "__version__", None),
        mediapipe_path=getattr(mp, "__file__", None),
        mediapipe_importable=dependencies["mediapipe"],
        hand_landmarker_available=dependencies["mediapipe_tasks"],
        default_model_path=str(DEFAULT_HAND_MODEL_PATH),
        default_model_exists=DEFAULT_HAND_MODEL_PATH.is_file(),
        pyvirtualcam_available=dependencies["pyvirtualcam"],
        pyvirtualcam_version=getattr(pyvirtualcam, "__version__", None),
    )


def probe_camera_indexes(video_capture: Any, indexes: range = range(5)) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index in indexes:
        capture = video_capture(index)
        try:
            opened = bool(capture.isOpened())
            readable = False
            if opened:
                readable, _ = capture.read()
            results.append({"index": index, "opened": opened, "readable": bool(readable)})
        finally:
            capture.release()
    return results


def emit_camera_probe(dependencies: dict[str, bool]) -> int:
    if not dependencies["opencv"]:
        message = "Camera probing requires OpenCV. Install it with the resolved Python interpreter."
        emit("error", code="opencv_missing", message=message, last_error=message)
        return 2
    cameras = probe_camera_indexes(cv2.VideoCapture)
    emit(
        "camera_probe",
        python_executable=sys.executable,
        checked_indexes=list(range(5)),
        cameras=cameras,
        open_indexes=[item["index"] for item in cameras if item["opened"]],
        readable_indexes=[item["index"] for item in cameras if item["readable"]],
    )
    return 0


def main() -> int:
    try:
        config = parse_args()
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        message = f"Invalid gesture config: {error}"
        emit("error", code="invalid_gesture_config", message=message, last_error=message)
        print(message, file=sys.stderr)
        emit("engine_stopped", reason="configuration_error")
        return 2
    dependencies = load_dependencies()
    if config.doctor:
        emit_doctor(dependencies)
        return 0
    if config.list_cameras:
        return emit_camera_probe(dependencies)
    required = [
        name for name in ("opencv", "numpy", "mediapipe") if not dependencies[name]
    ]
    if required:
        message = (
            "Missing required dependencies: "
            + ", ".join(required)
            + ". Install with: python3 -m pip install -r requirements.txt"
        )
        emit("error", code="missing_dependencies", message=message, last_error=message)
        print(message, file=sys.stderr)
        emit("engine_stopped", reason="dependency_error")
        return 2
    if not dependencies["mediapipe_tasks"]:
        message = (
            "MediaPipe imported, but MediaPipe Tasks vision.HandLandmarker is unavailable. "
            "Install the current requirements and verify with --doctor."
        )
        emit("error", code="hand_landmarker_api_unavailable", message=message, last_error=message)
        print(message, file=sys.stderr)
        emit("engine_stopped", reason="dependency_error")
        return 2
    if not config.hand_model_path.is_file():
        error = model_missing_error(config.hand_model_path)
        emit("error", **error)
        print(error["message"], file=sys.stderr)
        emit("engine_stopped", reason="model_error")
        return 2
    if config.virtual_camera and not dependencies["pyvirtualcam"]:
        message = (
            "pyvirtualcam is not installed, so virtual camera output is unavailable. "
            "Install requirements or retry with --no-virtual-camera."
        )
        emit(
            "error",
            code="virtual_camera_dependency_missing",
            message=message,
            last_error=message,
        )
        print(message, file=sys.stderr)
        emit("engine_stopped", reason="dependency_error")
        return 2
    emit(
        "engine_started",
        camera_index=config.camera_index,
        width=config.width,
        height=config.height,
        fps=config.fps,
        preview_enabled=config.preview,
        virtual_camera_enabled=config.virtual_camera,
        draw_landmarks=config.draw_landmarks,
        debug_overlay=config.debug_overlay,
        gesture_config=config.gesture_config_path,
        hand_model=str(config.hand_model_path),
        gesture_tuning=config.tuning.to_dict(),
    )
    try:
        PaintCamEngine(config).run()
    except KeyboardInterrupt:
        emit("engine_stopped", reason="interrupted")
        return 0
    except Exception as error:
        message = str(error)
        emit(
            "error",
            code="engine_error",
            message=message,
            last_error=message,
            camera_index=config.camera_index,
        )
        print(f"PaintCam error: {message}", file=sys.stderr)
        emit("engine_stopped", reason="error")
        return 1
    emit("engine_stopped", reason="normal")
    return 0
