"""Tests for camera_mass.py. Need PG_ROOT pointing at a Playground b9be205 checkout."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from camera_mass import (CAMERA_POS_WORLD_HOME, WOZ_POS_HEAD, head_com_home, load_with_camera,  # noqa: E402
                         parse_head_mass)

PG_ROOT = os.environ.get("PG_ROOT")
XML = os.path.join(PG_ROOT or "", "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml")


@unittest.skipUnless(PG_ROOT, "PG_ROOT not set")
class CameraMassTest(unittest.TestCase):
    def test_zero_grams_is_the_stock_model(self):
        import mujoco
        stock = mujoco.MjModel.from_xml_path(XML)
        m = load_with_camera(XML, 0)
        self.assertEqual(m.nbody, stock.nbody)
        np.testing.assert_array_equal(m.body_mass, stock.body_mass)

    def test_camera_adds_mass_at_the_mount_point(self):
        m0, m = load_with_camera(XML, 0), load_with_camera(XML, 10)
        (mass0, com0), (mass1, com1) = head_com_home(m0), head_com_home(m)
        self.assertAlmostEqual(mass1 - mass0, 0.010, places=9)
        self.assertEqual(m.nq, m0.nq)
        import mujoco
        d = mujoco.MjData(m)
        d.qpos[:] = m.keyframe("home").qpos
        mujoco.mj_kinematics(m, d)
        np.testing.assert_allclose(d.xpos[m.body("camera_mass").id], CAMERA_POS_WORLD_HOME, atol=1e-9)
        expect = (mass0 * com0 + 0.010 * CAMERA_POS_WORLD_HOME) / mass1
        np.testing.assert_allclose(com1, expect, atol=1e-9)

    def test_mount_point_in_head_frame(self):
        import mujoco
        m = load_with_camera(XML, 10, WOZ_POS_HEAD)
        d = mujoco.MjData(m)
        d.qpos[:] = m.keyframe("home").qpos
        mujoco.mj_kinematics(m, d)
        h = m.body("head_assembly").id
        local = d.xmat[h].reshape(3, 3).T @ (d.xpos[m.body("camera_mass").id] - d.xpos[h])
        np.testing.assert_allclose(local, WOZ_POS_HEAD, atol=1e-9)
        # head frame at home: +x is world up, -z is world forward (+x)
        np.testing.assert_allclose(d.xmat[h].reshape(3, 3)[:, 0], [0, 0, 1], atol=1e-3)
        np.testing.assert_allclose(d.xmat[h].reshape(3, 3)[:, 2], [-1, 0, 0], atol=1e-3)

    def test_negative_mass_rejected(self):
        with self.assertRaises(ValueError):
            load_with_camera(XML, -1)


if __name__ == "__main__":
    unittest.main()


class ParseHeadMassTest(unittest.TestCase):
    def test_parse(self):
        k, p = parse_head_mass("-12.5@22,47,6")
        self.assertAlmostEqual(k, -0.0125)
        np.testing.assert_allclose(p, [0.022, 0.047, 0.006])

    def test_bad_input_raises(self):
        for s in ("18", "18@1,2", "x@1,2,3"):
            with self.assertRaises(ValueError):
                parse_head_mass(s)


@unittest.skipUnless(PG_ROOT, "PG_ROOT not set")
class HeadMassMergeTest(unittest.TestCase):
    def test_merged_mass_matches_child_body(self):
        a = load_with_camera(XML, 10, WOZ_POS_HEAD)
        b = load_with_camera(XML, 0, None, [(0.010, WOZ_POS_HEAD)])
        (ma, ca), (mb, cb) = head_com_home(a), head_com_home(b)
        self.assertAlmostEqual(ma, mb, places=9)
        np.testing.assert_allclose(ca, cb, atol=1e-9)

    def test_add_then_remove_restores_stock(self):
        import mujoco
        stock = mujoco.MjModel.from_xml_path(XML)
        m = load_with_camera(XML, 0, None, [(0.018, np.array([0.022, 0.047, 0.006])),
                                             (-0.018, np.array([0.022, 0.047, 0.006]))])
        h = stock.body("head_assembly").id
        self.assertAlmostEqual(m.body_mass[h], stock.body_mass[h], places=9)
        np.testing.assert_allclose(m.body_ipos[h], stock.body_ipos[h], atol=1e-9)
        np.testing.assert_allclose(np.sort(m.body_inertia[h]), np.sort(stock.body_inertia[h]), rtol=1e-6)

    def test_removal_below_zero_raises(self):
        with self.assertRaises(ValueError):
            load_with_camera(XML, 0, None, [(-1.0, np.zeros(3))])
