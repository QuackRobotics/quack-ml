"""Unit tests for servo.py. The Playground tests need PG_ROOT pointing at a b9be205 checkout."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from servo import apply_spec, load_spec  # noqa: E402

PG_ROOT = os.environ.get("PG_ROOT")
XMLS = os.path.join(PG_ROOT or "", "playground", "open_duck_mini_v2", "xmls")
SPECS = os.path.join(os.path.dirname(HERE), "specs")


def write_spec(spec: dict) -> str:
    fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(spec, fh)
    fh.close()
    return fh.name


class LoadSpecTest(unittest.TestCase):
    def assert_rejected(self, spec: dict) -> None:
        with self.assertRaises(ValueError):
            load_spec(write_spec(spec))

    def test_rejects_unknown_group(self):
        self.assert_rejected({"name": "x", "groups": {"elbow": {"kp": 1}}})

    def test_rejects_unknown_field(self):
        self.assert_rejected({"name": "x", "groups": {"knee": {"kv": 1}}})

    def test_rejects_non_positive_and_bool(self):
        self.assert_rejected({"name": "x", "groups": {"knee": {"kp": 0}}})
        self.assert_rejected({"name": "x", "groups": {"knee": {"kp": True}}})

    def test_rejects_unsafe_name(self):
        self.assert_rejected({"name": "../x", "groups": {"knee": {"kp": 1}}})

    def test_shipped_specs_load(self):
        for fn in os.listdir(SPECS):
            load_spec(os.path.join(SPECS, fn))


@unittest.skipUnless(PG_ROOT, "set PG_ROOT to a Playground b9be205 checkout")
class ApplySpecTest(unittest.TestCase):
    def model(self, scene: str = "scene_flat_terrain.xml") -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_path(os.path.join(XMLS, scene))

    def test_knee_group_changes_only_knees(self):
        before, after = self.model(), self.model()
        apply_spec(after, {"groups": {"knee": {"forcerange": 4.0, "kp": 20.0, "armature": 0.05}}})
        knees = [k for k in range(after.nu) if after.actuator(k).name.endswith("_knee")]
        self.assertEqual(len(knees), 2)
        for k in range(after.nu):
            dof = after.jnt_dofadr[after.actuator_trnid[k, 0]]
            if k in knees:
                self.assertEqual(tuple(after.actuator_forcerange[k]), (-4.0, 4.0))
                self.assertEqual(after.actuator_gainprm[k, 0], 20.0)
                self.assertEqual(after.actuator_biasprm[k, 1], -20.0)
                self.assertEqual(after.dof_armature[dof], 0.05)
            else:
                np.testing.assert_array_equal(after.actuator_forcerange[k], before.actuator_forcerange[k])
                np.testing.assert_array_equal(after.actuator_gainprm[k], before.actuator_gainprm[k])
                self.assertEqual(after.dof_armature[dof], before.dof_armature[dof])

    def test_damping_friction_and_mass_on_parent_body(self):
        before, after = self.model(), self.model()
        apply_spec(after, {"groups": {"knee": {"damping": 0.82, "frictionloss": 0.1,
                                               "added_mass": 0.034}}})
        changed = set()
        for k in range(after.nu):
            if not after.actuator(k).name.endswith("_knee"):
                continue
            dof = after.jnt_dofadr[after.actuator_trnid[k, 0]]
            self.assertEqual(after.dof_damping[dof], 0.82)
            self.assertEqual(after.dof_frictionloss[dof], 0.1)
            body = after.body_parentid[after.jnt_bodyid[after.actuator_trnid[k, 0]]]
            changed.add(body)
            self.assertAlmostEqual(after.body_mass[body], before.body_mass[body] + 0.034)
            np.testing.assert_allclose(after.body_inertia[body] / before.body_inertia[body],
                                       after.body_mass[body] / before.body_mass[body])
        self.assertEqual(len(changed), 2)
        for b in range(after.nbody):
            if b not in changed:
                self.assertEqual(after.body_mass[b], before.body_mass[b])
        self.assertAlmostEqual(after.body_subtreemass[0], before.body_subtreemass[0] + 0.068)

    def test_pg_cap191_spec_matches_xml_capped_model(self):
        """The spec must reproduce the XML edit used for the formal gate run pg_cap191."""
        with tempfile.TemporaryDirectory() as tmp:
            xmls = shutil.copytree(XMLS, os.path.join(tmp, "xmls"))
            subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "make_pg_cap.py"),
                            xmls, "1.91"], check=True)
            from_xml = mujoco.MjModel.from_xml_path(os.path.join(xmls, "scene_flat_terrain_cap.xml"))
        from_spec = self.model()
        apply_spec(from_spec, load_spec(os.path.join(SPECS, "pg_cap191.json")))
        for field in ("actuator_forcerange", "actuator_forcelimited", "actuator_gainprm",
                      "actuator_biasprm", "dof_armature"):
            np.testing.assert_array_equal(getattr(from_spec, field), getattr(from_xml, field), field)


if __name__ == "__main__":
    unittest.main()


class PerturbJointsTest(unittest.TestCase):
    def test_same_seed_same_pose_and_only_joints_move(self) -> None:
        from start_state import JOINT_NOISE_RAD, perturb_joints
        qpos = np.zeros(10)
        qadr = np.array([7, 8, 9])
        first, second = perturb_joints(qpos, qadr, 3), perturb_joints(qpos, qadr, 3)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:7], 0.0)
        self.assertTrue(np.all(np.abs(first[qadr]) <= JOINT_NOISE_RAD))
        self.assertFalse(np.array_equal(first, perturb_joints(qpos, qadr, 4)))

    def test_rejects_non_positive_amplitude(self) -> None:
        from start_state import perturb_joints
        with self.assertRaises(ValueError):
            perturb_joints(np.zeros(3), np.array([0]), 0, amplitude=0.0)
