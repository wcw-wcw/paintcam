import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from paintcam.hand_tracking import (
    DEFAULT_HAND_MODEL_PATH,
    adapt_hand_landmarker_result,
    model_missing_error,
    resolve_hand_model_path,
)


def point(x, y):
    return SimpleNamespace(x=x, y=y, z=0.0)


class HandTrackingAdapterTests(unittest.TestCase):
    def test_zero_hands(self):
        result = SimpleNamespace(hand_landmarks=[], handedness=[])
        self.assertEqual(adapt_hand_landmarker_result(result), [])

    def test_one_hand_with_handedness_and_clamped_landmarks(self):
        category = SimpleNamespace(category_name="Left", score=0.91)
        result = SimpleNamespace(
            hand_landmarks=[[point(-0.2, 1.2), point(0.25, 0.75)]],
            handedness=[[category]],
        )
        hands = adapt_hand_landmarker_result(result)
        self.assertEqual(len(hands), 1)
        self.assertEqual(hands[0].landmarks, [(0.0, 1.0), (0.25, 0.75)])
        self.assertEqual(hands[0].handedness, "Left")
        self.assertAlmostEqual(hands[0].handedness_score, 0.91)

    def test_two_hands_without_handedness(self):
        result = SimpleNamespace(
            hand_landmarks=[[point(0.1, 0.2)], [point(0.8, 0.7)]],
            handedness=[],
        )
        hands = adapt_hand_landmarker_result(result)
        self.assertEqual(len(hands), 2)
        self.assertIsNone(hands[0].handedness)
        self.assertEqual(hands[1].landmarks, [(0.8, 0.7)])


class HandModelPathTests(unittest.TestCase):
    def test_default_model_path(self):
        self.assertEqual(resolve_hand_model_path(), DEFAULT_HAND_MODEL_PATH)
        self.assertTrue(str(DEFAULT_HAND_MODEL_PATH).endswith("engine/models/hand_landmarker.task"))

    def test_override_model_path_is_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "custom.task"
            self.assertEqual(resolve_hand_model_path(expected), expected.resolve())

    def test_missing_model_error_is_structured(self):
        error = model_missing_error("/tmp/not-a-real-hand-model.task")
        self.assertEqual(error["code"], "hand_landmarker_model_missing")
        self.assertIn(".task model bundle", error["message"])
        self.assertTrue(error["model_path"].endswith("not-a-real-hand-model.task"))


if __name__ == "__main__":
    unittest.main()
