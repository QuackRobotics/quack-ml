"""Tests for body_vx.py."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from body_vx import body_forward_speed  # noqa: E402


class BodyForwardSpeed(unittest.TestCase):
    def test_straight_along_world_x(self):
        t = np.arange(0, 10, 0.02)
        self.assertAlmostEqual(body_forward_speed(0.1 * t, 0 * t, 0 * t, 0.02), 0.1, places=9)

    def test_straight_at_heading_matches_speed_not_world_x(self):
        t = np.arange(0, 10, 0.02); th = np.radians(100)
        v = body_forward_speed(0.1 * t * np.cos(th), 0.1 * t * np.sin(th), np.full_like(t, th), 0.02)
        self.assertAlmostEqual(v, 0.1, places=9)

    def test_turning_circle_gives_tangential_speed(self):
        t = np.arange(0, 100, 0.02); w = np.radians(100) / 100; r = 0.1 / w
        yaw = w * t
        v = body_forward_speed(r * np.sin(yaw), r * (1 - np.cos(yaw)), yaw, 0.02)
        self.assertAlmostEqual(v, 0.1, places=5)

    def test_sideways_motion_is_not_forward(self):
        t = np.arange(0, 10, 0.02)
        self.assertAlmostEqual(body_forward_speed(0 * t, 0.1 * t, 0 * t, 0.02), 0.0, places=9)

    def test_bad_input_raises(self):
        with self.assertRaises(ValueError):
            body_forward_speed([0.0], [0.0], [0.0], 0.02)


if __name__ == "__main__":
    unittest.main()
