"""Unit tests for head_overlay.py (no MuJoCo needed)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from head_overlay import HEAD, offsets_rad, overlay, parse_fracs  # noqa: E402

# Playground b9be205 head joint ranges: neck_pitch, head_pitch, head_yaw, head_roll
RANGE = np.array([[-0.349, 1.134], [-0.785, 0.785], [-2.793, 2.793], [-0.524, 0.524]])


class HeadOverlayTest(unittest.TestCase):
    def test_parse_fracs_rejects_bad_input(self):
        self.assertEqual(parse_fracs("0.15,0,-0.15,0").tolist(), [0.15, 0, -0.15, 0])
        with self.assertRaises(ValueError):
            parse_fracs("0.15,0,0")
        with self.assertRaises(ValueError):
            parse_fracs("1.5,0,0,0")

    def test_offsets_are_fraction_of_full_range(self):
        off = offsets_rad(np.array([0.15, -0.15, 0.15, 0.0]), RANGE)
        np.testing.assert_allclose(off, [0.15 * 1.483, -0.15 * 1.570, 0.15 * 5.586, 0.0])

    def test_overlay_touches_only_head_and_clips(self):
        t = np.arange(14, dtype=float) * 0.01
        out, hits = overlay(t, np.array([0.2, 0.0, 0.0, 1.0]), RANGE[:, 0], RANGE[:, 1])
        np.testing.assert_array_equal(np.delete(out, range(5, 9)), np.delete(t, range(5, 9)))
        np.testing.assert_allclose(out[HEAD], [0.05 + 0.2, 0.06, 0.07, 0.524])
        self.assertEqual(hits, 1)


if __name__ == "__main__":
    unittest.main()
