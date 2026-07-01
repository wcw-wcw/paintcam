import unittest

from paintcam.app import (
    Hand,
    calculate_zoom,
    clamp,
    normalized_distance,
    palette_index,
    pixel_distance,
    probe_camera_indexes,
    probe_virtual_camera,
    validate_engine_command,
    virtual_camera_backend,
)


class HelperTests(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual(clamp(-1, 0, 2), 0)
        self.assertEqual(clamp(1, 0, 2), 1)
        self.assertEqual(clamp(3, 0, 2), 2)

    def test_normalized_distance(self):
        hand = Hand([(0.0, 0.0), (0.3, 0.4)])
        self.assertAlmostEqual(normalized_distance(hand, 0, 1), 0.5)

    def test_palette_index_clamps_edges(self):
        self.assertEqual(palette_index(-4, 700, 7), 0)
        self.assertEqual(palette_index(350, 700, 7), 3)
        self.assertEqual(palette_index(900, 700, 7), 6)

    def test_zoom_math_and_clamping(self):
        self.assertEqual(calculate_zoom(1.0, 0.5, 1.0, 1.0, 2.8), 2.0)
        self.assertEqual(calculate_zoom(2.0, 1.0, 2.0, 1.0, 2.8), 2.8)

    def test_camera_probe_releases_captures_and_reports_readability(self):
        captures = []

        class FakeCapture:
            def __init__(self, index):
                self.index = index
                self.released = False
                captures.append(self)

            def isOpened(self):
                return self.index == 1

            def read(self):
                return self.index == 1, object()

            def release(self):
                self.released = True

        self.assertEqual(
            probe_camera_indexes(FakeCapture, range(3)),
            [
                {"index": 0, "opened": False, "readable": False},
                {"index": 1, "opened": True, "readable": True},
                {"index": 2, "opened": False, "readable": False},
            ],
        )
        self.assertTrue(all(capture.released for capture in captures))

    def test_command_validation_is_bounded(self):
        self.assertEqual(
            validate_engine_command({"command": "set_brush_size", "brush_size": 24}),
            {"command": "set_brush_size", "brush_size": 24},
        )
        with self.assertRaises(ValueError):
            validate_engine_command({"command": "set_brush_size", "brush_size": 101})
        with self.assertRaises(ValueError):
            validate_engine_command({"command": "unknown"})

    def test_pixel_deadzone_distance(self):
        self.assertEqual(pixel_distance((1, 1), (4, 5)), 5.0)

    def test_virtual_camera_probe_does_not_require_a_real_backend(self):
        cameras = []

        class FakeCamera:
            backend = "test-backend"

            def __init__(self, width, height, fps):
                self.size = (width, height, fps)
                self.closed = False
                cameras.append(self)

            def close(self):
                self.closed = True

        class FakeModule:
            Camera = FakeCamera

        result = probe_virtual_camera(FakeModule(), width=320, height=240, fps=15)
        self.assertEqual(result["status"], "active")
        self.assertEqual(result["backend"], "test-backend")
        self.assertTrue(result["created"])
        self.assertTrue(cameras[0].closed)

    def test_virtual_camera_probe_reports_missing_import(self):
        result = probe_virtual_camera(None)
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["importable"])
        self.assertIn("not installed", result["last_error"])

    def test_virtual_camera_backend_normalizes_name(self):
        self.assertEqual(
            virtual_camera_backend(type("Camera", (), {"backend": "obs"})()), "obs"
        )
        self.assertIsNone(virtual_camera_backend(object()))


if __name__ == "__main__":
    unittest.main()
