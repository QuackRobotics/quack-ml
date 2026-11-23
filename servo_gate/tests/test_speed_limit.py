"""Unit tests for servo.SpeedLimit (no Playground needed)."""
import os
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from servo import SpeedLimit, apply_spec  # noqa: E402

XML = """<mujoco><worldbody>
 <body name="a"><joint name="lj" type="hinge"/><geom size="0.1" mass="1"/></body>
 <body name="b" pos="1 0 0"><joint name="rj" type="hinge"/><geom size="0.1" mass="1"/></body>
</worldbody><actuator>
 <position name="left_knee" joint="lj" kp="10" forcelimited="true" forcerange="-5 5"/>
 <position name="right_knee" joint="rj" kp="10" forcelimited="true" forcerange="-5 5"/>
</actuator></mujoco>"""


class SpeedLimitTest(unittest.TestCase):
    def setUp(self):
        self.m = mujoco.MjModel.from_xml_string(XML)
        self.d = mujoco.MjData(self.m)

    def test_line_and_flat_clip(self):
        spec = {"name": "t", "groups": {"knee": {"forcerange": 1.0, "stall_torque": 0.8, "noload_speed": 2.0}}}
        apply_spec(self.m, spec)
        lim = SpeedLimit(self.m, spec)
        self.d.qvel[:] = [1.0, -1.0]
        lim.apply(self.m, self.d)
        # w = +1: motoring cap 0.8*(1-0.5) = 0.4, braking -0.8*(1.5) = -1.2 -> clipped to -1.0
        np.testing.assert_allclose(self.m.actuator_forcerange[0], [-1.0, 0.4])
        np.testing.assert_allclose(self.m.actuator_forcerange[1], [-0.4, 1.0])
        self.d.qvel[:] = [3.0, 0.0]  # beyond no-load: cannot push forward at all
        lim.apply(self.m, self.d)
        np.testing.assert_allclose(self.m.actuator_forcerange[0], [-1.0, -0.4])
        np.testing.assert_allclose(self.m.actuator_forcerange[1], [-0.8, 0.8])

    def test_force_obeys_envelope(self):
        spec = {"name": "t", "groups": {"knee": {"stall_torque": 0.8, "noload_speed": 2.0}}}
        apply_spec(self.m, spec)
        lim = SpeedLimit(self.m, spec)
        self.d.ctrl[:] = [3.0, 3.0]
        for _ in range(500):
            w = self.d.qvel[0]
            lim.apply(self.m, self.d)
            mujoco.mj_step(self.m, self.d)
            self.assertLessEqual(self.d.actuator_force[0], 0.8 * (1 - w / 2.0) + 1e-9)
        self.assertLess(self.d.qvel[0], 2.0)

    def test_no_line_is_noop(self):
        spec = {"name": "t", "groups": {"knee": {"forcerange": 1.0}}}
        apply_spec(self.m, spec)
        lim = SpeedLimit(self.m, spec)
        self.d.qvel[:] = [1.0, 1.0]
        lim.apply(self.m, self.d)
        np.testing.assert_allclose(self.m.actuator_forcerange, [[-1, 1], [-1, 1]])


if __name__ == "__main__":
    unittest.main()
