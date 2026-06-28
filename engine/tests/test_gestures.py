import unittest

from paintcam.gestures import (
    DrawingState,
    GestureContext,
    GestureTuning,
    Hand,
    PaletteState,
    PinchDebouncer,
    ZoomState,
    calculate_zoom,
    default_registry,
    palette_hit_test,
    pinch_confidence,
)


COLORS = [(0, 0, 0), (255, 255, 255), (10, 20, 30)]


def hand(index=(0.5, 0.4), thumb=(0.48, 0.4)):
    landmarks = [(0.5, 0.5)] * 21
    landmarks[4] = thumb
    landmarks[8] = index
    return Hand(landmarks)


def context(hands, timestamp_ms=0):
    tuning = GestureTuning()
    return GestureContext(
        width=1000,
        height=500,
        hands=hands,
        palette=PaletteState(COLORS, selected_index=0),
        drawing=DrawingState(tuning.brush_size),
        zoom=ZoomState(),
        tuning=tuning,
        timestamp_ms=timestamp_ms,
        frame_index=1,
    )


class GestureStateTests(unittest.TestCase):
    def test_pinch_threshold_confidence(self):
        self.assertEqual(pinch_confidence(0.055, 0.055, 0.075), 1.0)
        self.assertAlmostEqual(pinch_confidence(0.065, 0.055, 0.075), 0.5)
        self.assertEqual(pinch_confidence(0.08, 0.055, 0.075), 0.0)

    def test_pinch_debounce_and_release_hysteresis(self):
        pinch = PinchDebouncer(0.055, 0.075, 100, 3)
        self.assertFalse(pinch.update(0.04, 0)[0])
        self.assertFalse(pinch.update(0.04, 99)[0])
        self.assertTrue(pinch.update(0.04, 100)[0])
        self.assertTrue(pinch.update(0.065, 110)[0])
        self.assertFalse(pinch.update(0.075, 120)[0])

    def test_palette_hit_testing(self):
        self.assertIsNone(palette_hit_test(hand(index=(0.2, 0.5)), 1000, 500, 0.128, 3))
        self.assertEqual(
            palette_hit_test(hand(index=(0.75, 0.95)), 1000, 500, 0.128, 3),
            2,
        )

    def test_zoom_sensitivity_and_clamping(self):
        self.assertEqual(calculate_zoom(1.0, 0.5, 1.0, 1.0, 1.0, 2.8), 2.0)
        self.assertEqual(calculate_zoom(2.0, 0.5, 1.0, 1.0, 1.0, 2.8), 2.8)
        self.assertEqual(calculate_zoom(1.5, 1.0, 0.0, 2.0, 1.0, 2.8), 1.0)

    def test_palette_priority_wins_over_two_hand_zoom(self):
        registry = default_registry(GestureTuning())
        winner, results = registry.process(
            context(
                [
                    hand(index=(0.3, 0.95)),
                    hand(index=(0.8, 0.4), thumb=(0.78, 0.4)),
                ]
            )
        )
        self.assertEqual(winner.gesture_name, "palette_select")
        zoom = next(item for item in results if item.gesture_name == "two_hand_zoom")
        self.assertFalse(zoom.active)
        self.assertEqual(zoom.conflict, "palette_select")

    def test_palette_requires_stable_hold_before_selection(self):
        registry = default_registry(GestureTuning())
        palette_hand = hand(index=(0.75, 0.95))
        first, _ = registry.process(context([palette_hand], 0))
        second, _ = registry.process(context([palette_hand], 50))
        third, _ = registry.process(context([palette_hand], 100))
        self.assertEqual(first.action, "none")
        self.assertEqual(second.action, "none")
        self.assertEqual(third.action, "select_color")

    def test_two_hand_zoom_blocks_drawing(self):
        registry = default_registry(GestureTuning())
        winner, _ = registry.process(
            context(
                [
                    hand(index=(0.3, 0.4), thumb=(0.31, 0.4)),
                    hand(index=(0.8, 0.4), thumb=(0.79, 0.4)),
                ]
            )
        )
        self.assertEqual(winner.gesture_name, "two_hand_zoom")

    def test_draw_requires_stable_pinch(self):
        registry = default_registry(GestureTuning())
        pinching_hand = hand(index=(0.5, 0.4), thumb=(0.51, 0.4))
        first, _ = registry.process(context([pinching_hand], 0))
        second, _ = registry.process(context([pinching_hand], 50))
        third, _ = registry.process(context([pinching_hand], 100))
        fourth, _ = registry.process(context([pinching_hand], 140))
        self.assertFalse(first.active)
        self.assertFalse(second.active)
        self.assertFalse(third.active)
        self.assertEqual(fourth.gesture_name, "pinch_draw")
        self.assertTrue(fourth.active)


if __name__ == "__main__":
    unittest.main()
