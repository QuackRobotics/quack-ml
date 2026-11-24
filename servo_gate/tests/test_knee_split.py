"""Tests for knee_split.py."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from knee_split import phase_rms  # noqa: E402


class PhaseRms(unittest.TestCase):
    def test_constant_force_per_phase(self):
        contact = np.array([1, 0] * 50)
        force = np.repeat(np.where(contact > 0, 1.0, 3.0), 10)
        r = phase_rms(force, contact, 10, 0.002, 0.0)
        self.assertAlmostEqual(r["stance"], 1.0)
        self.assertAlmostEqual(r["swing"], 3.0)
        self.assertAlmostEqual(r["all"], np.sqrt(5.0))
        self.assertAlmostEqual(r["stance_frac"], 0.5)

    def test_window_drops_early_steps(self):
        contact = np.ones(100)
        force = np.r_[np.full(500, 9.0), np.full(500, 2.0)]
        self.assertAlmostEqual(phase_rms(force, contact, 10, 0.002, 1.001)["all"], 2.0)  # step t is (i+1)*dt, so t=1.0 is still a 9.0 step

    def test_length_mismatch_fails(self):
        with self.assertRaises(ValueError):
            phase_rms(np.zeros(99), np.zeros(10), 10, 0.002, 0.0)


if __name__ == "__main__":
    unittest.main()
