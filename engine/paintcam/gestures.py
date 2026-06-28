from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

Color = tuple[int, int, int]
Point = tuple[int, int]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass
class GestureTuning:
    pinch_distance_threshold: float = 0.05
    pinch_release_threshold: float = 0.075
    pinch_debounce_ms: int = 140
    pinch_min_stable_frames: int = 4
    draw_deadzone_px: int = 3
    palette_height_ratio: float = 0.128
    palette_hold_ms: int = 100
    palette_min_stable_frames: int = 3
    palette_cooldown_ms: int = 150
    draw_cooldown_ms: int = 100
    zoom_cooldown_ms: int = 150
    zoom_sensitivity: float = 1.0
    min_zoom: float = 1.0
    max_zoom: float = 2.8
    brush_size: int = 16

    def validate(self) -> None:
        if not 0 < self.pinch_distance_threshold < self.pinch_release_threshold:
            raise ValueError(
                "pinch_distance_threshold must be positive and less than pinch_release_threshold"
            )
        if not 0 < self.palette_height_ratio < 0.5:
            raise ValueError("palette_height_ratio must be between 0 and 0.5")
        if self.min_zoom <= 0 or self.max_zoom < self.min_zoom:
            raise ValueError("zoom bounds are invalid")
        if self.brush_size < 1:
            raise ValueError("brush_size must be positive")
        if self.pinch_min_stable_frames < 1 or self.palette_min_stable_frames < 1:
            raise ValueError("stable frame counts must be positive")
        if self.draw_deadzone_px < 0:
            raise ValueError("draw_deadzone_px cannot be negative")
        if min(
            self.pinch_debounce_ms,
            self.palette_hold_ms,
            self.palette_cooldown_ms,
            self.draw_cooldown_ms,
            self.zoom_cooldown_ms,
        ) < 0:
            raise ValueError("gesture timing values cannot be negative")

    @classmethod
    def from_json(cls, path: str | Path) -> "GestureTuning":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("gesture config must contain a JSON object")
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown gesture config fields: {', '.join(unknown)}")
        tuning = cls(**data)
        tuning.validate()
        return tuning

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Hand:
    landmarks: list[tuple[float, float]]
    handedness: str | None = None
    handedness_score: float | None = None

    def point(self, index: int, width: int, height: int) -> Point:
        x, y = self.landmarks[index]
        return int(x * width), int(y * height)


@dataclass
class PaletteState:
    colors: list[Color]
    selected_index: int = 2

    @property
    def selected_color(self) -> Color:
        return self.colors[self.selected_index]


@dataclass
class DrawingState:
    brush_size: int
    previous_point: Point | None = None
    drawing: bool = False


@dataclass
class ZoomState:
    value: float = 1.0


@dataclass
class GestureContext:
    width: int
    height: int
    hands: list[Hand]
    palette: PaletteState
    drawing: DrawingState
    zoom: ZoomState
    tuning: GestureTuning
    timestamp_ms: int
    frame_index: int

    @property
    def palette_top(self) -> int:
        return palette_top(self.height, self.tuning.palette_height_ratio)


@dataclass
class GestureResult:
    gesture_name: str
    active: bool
    confidence: float = 0.0
    action: str = "none"
    debug_text: str = ""
    consumed_frame: bool = False
    cooldown_remaining_ms: int = 0
    conflict: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


class Gesture(Protocol):
    name: str
    priority: int
    cooldown_ms: int
    confidence_threshold: float
    exclusive_group: str | None

    def evaluate(self, context: GestureContext) -> GestureResult: ...
    def deactivate(self) -> None: ...


def normalized_distance(hand: Hand, first: int, second: int) -> float:
    return math.dist(hand.landmarks[first], hand.landmarks[second])


def palette_top(height: int, height_ratio: float) -> int:
    return int(height * (1.0 - height_ratio))


def palette_hit_test(
    hand: Hand, width: int, height: int, height_ratio: float, color_count: int
) -> int | None:
    point = hand.point(8, width, height)
    if point[1] < palette_top(height, height_ratio):
        return None
    return palette_index(point[0], width, color_count)


def palette_index(x: int, width: int, count: int) -> int:
    if width <= 0 or count <= 0:
        raise ValueError("width and palette count must be positive")
    return max(0, min(count - 1, int(x / (width / count))))


def calculate_zoom(
    baseline_zoom: float,
    baseline_distance: float,
    distance: float,
    sensitivity: float,
    minimum: float,
    maximum: float,
) -> float:
    ratio = distance / max(baseline_distance, 0.001)
    adjusted_ratio = 1.0 + ((ratio - 1.0) * sensitivity)
    return clamp(baseline_zoom * adjusted_ratio, minimum, maximum)


def pinch_confidence(distance: float, threshold: float, release_threshold: float) -> float:
    span = max(release_threshold - threshold, 0.001)
    return clamp((release_threshold - distance) / span, 0.0, 1.0)


class PinchDebouncer:
    def __init__(
        self,
        threshold: float,
        release_threshold: float,
        debounce_ms: int,
        minimum_stable_frames: int = 1,
    ) -> None:
        self.threshold = threshold
        self.release_threshold = release_threshold
        self.debounce_ms = debounce_ms
        self.minimum_stable_frames = minimum_stable_frames
        self.candidate_since_ms: int | None = None
        self.stable_frames = 0
        self.active = False

    def update(self, distance: float, timestamp_ms: int) -> tuple[bool, float]:
        confidence = pinch_confidence(distance, self.threshold, self.release_threshold)
        if self.active:
            if distance >= self.release_threshold:
                self.reset()
            return self.active, confidence
        if distance <= self.threshold:
            if self.candidate_since_ms is None:
                self.candidate_since_ms = timestamp_ms
            self.stable_frames += 1
            if (
                timestamp_ms - self.candidate_since_ms >= self.debounce_ms
                and self.stable_frames >= self.minimum_stable_frames
            ):
                self.active = True
        else:
            self.candidate_since_ms = None
            self.stable_frames = 0
        return self.active, confidence

    def reset(self) -> None:
        self.candidate_since_ms = None
        self.stable_frames = 0
        self.active = False


class PaletteSelectGesture:
    name = "palette_select"
    priority = 300
    confidence_threshold = 0.5
    exclusive_group = "manipulation"

    def __init__(self, tuning: GestureTuning) -> None:
        self.cooldown_ms = tuning.palette_cooldown_ms
        self.hold_ms = tuning.palette_hold_ms
        self.minimum_stable_frames = tuning.palette_min_stable_frames
        self.candidate_index: int | None = None
        self.candidate_since_ms: int | None = None
        self.stable_frames = 0

    def evaluate(self, context: GestureContext) -> GestureResult:
        if not context.hands:
            return GestureResult(self.name, False, debug_text="no hands")
        index = palette_hit_test(
            context.hands[0],
            context.width,
            context.height,
            context.tuning.palette_height_ratio,
            len(context.palette.colors),
        )
        if index is None:
            self.deactivate()
            return GestureResult(self.name, False, debug_text="index outside palette")
        if index != self.candidate_index:
            self.candidate_index = index
            self.candidate_since_ms = context.timestamp_ms
            self.stable_frames = 1
        else:
            self.stable_frames += 1
        held_ms = context.timestamp_ms - (
            self.candidate_since_ms
            if self.candidate_since_ms is not None
            else context.timestamp_ms
        )
        confirmed = (
            held_ms >= self.hold_ms and self.stable_frames >= self.minimum_stable_frames
        )
        point_y = context.hands[0].landmarks[8][1]
        top = 1.0 - context.tuning.palette_height_ratio
        confidence = clamp((point_y - top) / context.tuning.palette_height_ratio, 0.0, 1.0)
        confidence = max(0.75, confidence)
        return GestureResult(
            self.name,
            True,
            confidence,
            "select_color" if confirmed else "none",
            f"palette swatch {index} {'confirmed' if confirmed else 'holding'}",
            True,
            data={"palette_index": index, "confirmed": confirmed},
        )

    def deactivate(self) -> None:
        self.candidate_index = None
        self.candidate_since_ms = None
        self.stable_frames = 0


class TwoHandZoomGesture:
    name = "two_hand_zoom"
    priority = 200
    confidence_threshold = 0.8
    exclusive_group = "manipulation"

    def __init__(self, tuning: GestureTuning) -> None:
        self.cooldown_ms = tuning.zoom_cooldown_ms
        self.baseline_distance: float | None = None
        self.baseline_zoom = 1.0

    def evaluate(self, context: GestureContext) -> GestureResult:
        if len(context.hands) < 2:
            self.deactivate()
            return GestureResult(self.name, False, debug_text="needs two hands")
        distance = math.dist(context.hands[0].landmarks[8], context.hands[1].landmarks[8])
        if self.baseline_distance is None:
            self.baseline_distance = max(distance, 0.001)
            self.baseline_zoom = context.zoom.value
        zoom = calculate_zoom(
            self.baseline_zoom,
            self.baseline_distance,
            distance,
            context.tuning.zoom_sensitivity,
            context.tuning.min_zoom,
            context.tuning.max_zoom,
        )
        return GestureResult(
            self.name,
            True,
            1.0,
            "set_zoom",
            f"hand distance {distance:.3f}",
            True,
            data={"zoom": zoom, "distance": distance},
        )

    def deactivate(self) -> None:
        self.baseline_distance = None


class PinchDrawGesture:
    name = "pinch_draw"
    priority = 100
    confidence_threshold = 0.55
    exclusive_group = "manipulation"

    def __init__(self, tuning: GestureTuning) -> None:
        self.cooldown_ms = tuning.draw_cooldown_ms
        self.debouncer = PinchDebouncer(
            tuning.pinch_distance_threshold,
            tuning.pinch_release_threshold,
            tuning.pinch_debounce_ms,
            tuning.pinch_min_stable_frames,
        )

    def evaluate(self, context: GestureContext) -> GestureResult:
        if len(context.hands) != 1:
            self.deactivate()
            return GestureResult(self.name, False, debug_text="needs exactly one hand")
        hand = context.hands[0]
        if palette_hit_test(
            hand,
            context.width,
            context.height,
            context.tuning.palette_height_ratio,
            len(context.palette.colors),
        ) is not None:
            self.deactivate()
            return GestureResult(self.name, False, debug_text="palette owns frame")
        distance = normalized_distance(hand, 4, 8)
        active, confidence = self.debouncer.update(distance, context.timestamp_ms)
        if not active:
            waiting = self.debouncer.candidate_since_ms is not None
            return GestureResult(
                self.name,
                False,
                confidence,
                debug_text="pinch stabilizing" if waiting else f"pinch {distance:.3f}",
            )
        return GestureResult(
            self.name,
            True,
            confidence,
            "draw",
            f"pinch {distance:.3f}",
            True,
            data={"point": hand.point(8, context.width, context.height)},
        )

    def deactivate(self) -> None:
        self.debouncer.reset()


class GestureRegistry:
    def __init__(self, gestures: list[Gesture]) -> None:
        self.gestures = sorted(gestures, key=lambda item: item.priority, reverse=True)
        self.cooldown_until: dict[str, int] = {}
        self.previous_active: set[str] = set()

    def process(self, context: GestureContext) -> tuple[GestureResult, list[GestureResult]]:
        raw = [gesture.evaluate(context) for gesture in self.gestures]
        winner, results = resolve_gesture_results(
            self.gestures, raw, context.timestamp_ms, self.cooldown_until
        )
        active_now = {winner.gesture_name} if winner.active else set()
        for gesture in self.gestures:
            if gesture.name in self.previous_active and gesture.name not in active_now:
                self.cooldown_until[gesture.name] = context.timestamp_ms + gesture.cooldown_ms
            if winner.active and gesture.name not in active_now:
                gesture.deactivate()
        self.previous_active = active_now
        return winner, results

    def deactivate_all(self) -> None:
        for gesture in self.gestures:
            gesture.deactivate()
        self.previous_active.clear()


def resolve_gesture_results(
    gestures: list[Gesture],
    results: list[GestureResult],
    timestamp_ms: int,
    cooldown_until: dict[str, int] | None = None,
) -> tuple[GestureResult, list[GestureResult]]:
    cooldown_until = cooldown_until or {}
    by_name = {gesture.name: gesture for gesture in gestures}
    ordered = sorted(results, key=lambda item: by_name[item.gesture_name].priority, reverse=True)
    winner: GestureResult | None = None
    occupied_groups: set[str] = set()
    resolved: list[GestureResult] = []
    for result in ordered:
        gesture = by_name[result.gesture_name]
        remaining = max(0, cooldown_until.get(gesture.name, 0) - timestamp_ms)
        if result.active and remaining:
            result.active = False
            result.action = "none"
            result.cooldown_remaining_ms = remaining
            result.debug_text = f"cooldown {remaining}ms"
        if result.active and result.confidence < gesture.confidence_threshold:
            result.active = False
            result.action = "none"
            result.debug_text = "below confidence threshold"
        group = gesture.exclusive_group
        if result.active and group and group in occupied_groups:
            result.active = False
            result.action = "none"
            result.conflict = winner.gesture_name if winner else "exclusive group"
            result.debug_text = f"blocked by {result.conflict}"
        if result.active:
            if group:
                occupied_groups.add(group)
            if winner is None:
                winner = result
        resolved.append(result)
    if winner is None:
        candidate = max(resolved, key=lambda item: item.confidence, default=None)
        winner = GestureResult(
            "none",
            False,
            confidence=candidate.confidence if candidate else 0.0,
            debug_text=candidate.debug_text if candidate else "no gesture",
        )
    return winner, resolved


def default_registry(tuning: GestureTuning) -> GestureRegistry:
    return GestureRegistry(
        [
            PaletteSelectGesture(tuning),
            TwoHandZoomGesture(tuning),
            PinchDrawGesture(tuning),
        ]
    )
