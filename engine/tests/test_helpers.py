import unittest

from paintcam.app import Hand, calculate_zoom, clamp, normalized_distance, palette_index


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


if __name__ == "__main__":
    unittest.main()
