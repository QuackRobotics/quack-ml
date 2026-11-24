"""Policy loop for any MJCF, copied step-for-step from Playground mujoco_infer.py @ b9be205.

Differences from the reference (by design):
 - headless, 5000 control steps, logs (same as the allowed headless patch)
 - the policy drives only the non-antenna actuators (14); antenna actuators, if the
   model has them, stay at their home ctrl (0). Obs fields use those 14 joints.
Timing matches MJInferBase: one mj_step from qpos0, then home qpos/ctrl, then the
policy acts after every 10th physics step.
Usage: harness.py <model.xml> <onnx> <outdir> [--pg <pg_repo_root>] [--servo <spec.json>]
--servo overrides forcerange, kp, armature, damping, frictionloss and servo mass per leg joint group (see servo.py) on the
compiled model before the first step; without it the run is unchanged.
"""
import argparse
import os
import sys

import mujoco
import numpy as np
import onnxruntime

from camera_mass import load_with_camera, parse_head_mass
from servo import SpeedLimit, apply_spec, load_spec
from start_state import JOINT_NOISE_RAD, perturb_joints

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("onnx")
ap.add_argument("out")
ap.add_argument("--pg", default=os.path.join(os.path.dirname(__file__), "..", "pg"))
ap.add_argument("--servo", help="servo spec JSON with per-group forcerange/kp/armature/damping/frictionloss/added_mass")
ap.add_argument("--acc-offset", type=float, default=1.3,
                help="added to accel x in the obs; 1.3 = mujoco_infer.py, 0 = what joystick.py trains on (its +1.3 is a no-op)")
ap.add_argument("--seed", type=int, help=f"perturb the home joint angles by uniform +-{JOINT_NOISE_RAD} rad with this seed")
ap.add_argument("--head-mass", action="append", default=[], metavar="G@X,Y,Z",
                help="extra head point mass in grams at x,y,z mm in head_assembly (repeatable; negative removes)")
ap.add_argument("--vx", type=float, default=0.15, help="commanded forward speed m/s (default 0.15 = arrow-up = COMMANDS_RANGE_X[1])")
ap.add_argument("--wz", type=float, default=0.0, help="commanded yaw rate rad/s (default 0)")
ap.add_argument("--heading-hold-k", type=float, metavar="K",
                help="heading hold gain (rad/s)/rad: wz_cmd = wz + clip(K*wrap(psi_ref - psi), +-clamp), psi = IMU framequat yaw; off by default")
ap.add_argument("--heading-hold-clamp", type=float, default=0.1, help="heading hold correction clamp rad/s (default 0.1)")
ap.add_argument("--wz-max", type=float, default=1.0, help="final wz_cmd clip rad/s (default 1.0 = trained ang_vel_yaw range)")
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.pg, "playground", "common"))
from poly_reference_motion_numpy import PolyReferenceMotion  # noqa: E402

SIM_DT, DECIM, N_CTRL = 0.002, 10, 5000
DOF_VEL_SCALE, ACTION_SCALE, MAX_MOTOR_VEL = 0.05, 0.25, 5.24
COMMANDS = np.array([a.vx, 0.0, a.wz, 0.0, 0.0, 0.0, 0.0])  # default vx 0.15 = arrow-up = COMMANDS_RANGE_X[1]

m = (load_with_camera(a.model, 0.0, None, [parse_head_mass(x) for x in a.head_mass])
     if a.head_mass else mujoco.MjModel.from_xml_path(a.model))
m.opt.timestep = SIM_DT
spec = load_spec(a.servo) if a.servo else None
servo_rows = apply_spec(m, spec) if spec else []
speed_limit = SpeedLimit(m, spec) if spec else None
d = mujoco.MjData(m)
if speed_limit is not None:
    speed_limit.apply(m, d)
mujoco.mj_step(m, d)

names = [m.actuator(k).name for k in range(m.nu)]
pol = [k for k, n in enumerate(names) if "antenna" not in n]
jids = [m.actuator_trnid[k, 0] for k in pol]
qadr = np.array([m.jnt_qposadr[j] for j in jids])
vadr = np.array([m.jnt_dofadr[j] for j in jids])
key = m.keyframe("home")
default = key.ctrl[pol].copy()
d.qpos[:] = key.qpos if a.seed is None else perturb_joints(key.qpos, qadr, a.seed)
d.ctrl[:] = key.ctrl
motor_targets = default.copy()
prev_motor_targets = default.copy()

gyro_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")]
acc_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer")]
quat_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "orientation")]
floor_b, lf_b, rf_b = (d.body(n).id for n in ("floor", "foot_assembly", "foot_assembly_2"))


def heading_hold(psi, psi_ref, wz_user):
    """Atlas v1 heading hold, 50 Hz, no integrator: returns (wz_cmd, psi_ref, error, saturated).

    psi_ref is set at the first tick and reset whenever the user commands a turn (|wz_user| > 0).
    """
    if psi_ref is None or abs(wz_user) > 0:
        psi_ref = psi
    err = (psi_ref - psi + np.pi) % (2 * np.pi) - np.pi
    u = a.heading_hold_k * err
    sat = abs(u) >= a.heading_hold_clamp
    u = float(np.clip(u, -a.heading_hold_clamp, a.heading_hold_clamp))
    return float(np.clip(wz_user + u, -a.wz_max, a.wz_max)), psi_ref, float(err), bool(sat)


def contact(b1, b2):
    for i in range(d.ncon):
        g1, g2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
        if (g1 == b1 and g2 == b2) or (g1 == b2 and g2 == b1):
            return True
    return False


PRM = PolyReferenceMotion(os.path.join(a.pg, "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"))
sess = onnxruntime.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
n = len(pol)
last, last2, last3 = np.zeros(n), np.zeros(n), np.zeros(n)
imitation_i, phase = 0, np.array([0, 0])
saved_obs, log_force, log_base, log_targets, log_feet = [], [], [], [], []
log_qvel, log_hit, log_qpos = [], [], []
log_hh, psi_ref = [], None  # per control step: psi, error, wz_cmd, saturated (heading hold only)  # per physics step: joint speed (rad/s), force at its limit
counter = 0
while counter < N_CTRL * DECIM:
    if speed_limit is not None:
        speed_limit.apply(m, d)
    mujoco.mj_step(m, d)
    log_force.append(d.actuator_force[pol].copy())
    log_qvel.append(d.qvel[vadr].astype(np.float32))
    fr = m.actuator_forcerange[pol]
    log_hit.append((d.actuator_force[pol] >= fr[:, 1] - 1e-6) | (d.actuator_force[pol] <= fr[:, 0] + 1e-6))
    log_base.append(d.qpos[:7].copy())
    counter += 1
    if counter % DECIM == 0:
        imitation_i = (imitation_i + 1.0) % PRM.nb_steps_in_period
        ang = imitation_i / PRM.nb_steps_in_period * 2 * np.pi
        phase = np.array([np.cos(ang), np.sin(ang)])
        if a.heading_hold_k is not None:
            qw, qx, qy, qz = d.sensordata[quat_adr:quat_adr + 4]
            psi = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
            COMMANDS[2], psi_ref, err, sat = heading_hold(psi, psi_ref, a.wz)
            log_hh.append((psi, err, COMMANDS[2], sat))
        acc = d.sensordata[acc_adr:acc_adr + 3].copy()
        acc[0] += a.acc_offset
        obs = np.concatenate([
            d.sensordata[gyro_adr:gyro_adr + 3], acc, COMMANDS,
            d.qpos[qadr] - default, d.qvel[vadr] * DOF_VEL_SCALE,
            last, last2, last3, motor_targets,
            (contact(lf_b, floor_b), contact(rf_b, floor_b)), phase])
        saved_obs.append(obs)
        log_feet.append(obs[-4:-2].copy())  # (left, right) foot-floor contact
        action = sess.run(None, {"obs": [obs]})[0][0]
        last3, last2, last = last2.copy(), last.copy(), action.copy()
        motor_targets = default + action * ACTION_SCALE
        motor_targets = np.clip(motor_targets,
                                prev_motor_targets - MAX_MOTOR_VEL * (SIM_DT * DECIM),
                                prev_motor_targets + MAX_MOTOR_VEL * (SIM_DT * DECIM))
        prev_motor_targets = motor_targets.copy()
        d.ctrl[pol] = motor_targets.copy()
        log_targets.append(motor_targets.copy())
        log_qpos.append(d.qpos[qadr].copy())  # measured joint angles at the same 50 Hz tick as motor_targets

os.makedirs(a.out, exist_ok=True)
np.save(f"{a.out}/actuator_force.npy", np.array(log_force))
np.save(f"{a.out}/base_qpos.npy", np.array(log_base))
np.save(f"{a.out}/obs_first50.npy", np.array(saved_obs[:50]))
np.save(f"{a.out}/motor_targets.npy", np.array(log_targets))
np.save(f"{a.out}/joint_qpos.npy", np.array(log_qpos))
np.save(f"{a.out}/feet_contact.npy", np.array(log_feet))
np.save(f"{a.out}/joint_qvel.npy", np.array(log_qvel))
np.save(f"{a.out}/cap_hit.npy", np.array(log_hit))
if a.heading_hold_k is not None:
    np.save(f"{a.out}/heading_hold.npy", np.array(log_hh))
np.save(f"{a.out}/actuator_forcerange.npy", m.actuator_forcerange[pol])
open(f"{a.out}/actuator_names.txt", "w").write(" ".join(names[k] for k in pol))
print("mujoco", mujoco.__version__, "onnxruntime", onnxruntime.__version__)
print("model", a.model, "nu", m.nu, "policy actuators", n, "obs dim", len(saved_obs[0]))
print("actuators:", [names[k] for k in pol])
print("kp", np.round(m.actuator_gainprm[pol, 0], 3).tolist())
print("forcerange", np.round(m.actuator_forcerange[pol], 3).tolist())
print("servo spec", a.servo or "none")
if speed_limit is not None and speed_limit.k.size:
    print("torque-speed line", [(m.actuator(int(k)).name, float(t), float(w)) for k, t, w in
                                zip(speed_limit.k, speed_limit.ts, speed_limit.w0)])
print("acc x offset", a.acc_offset)
print("heading hold", "off" if a.heading_hold_k is None else
      f"k {a.heading_hold_k} clamp {a.heading_hold_clamp} wz_max {a.wz_max}")
print("head mass add", a.head_mass or "none")
print("start seed", "none" if a.seed is None else f"{a.seed} (joint noise +-{JOINT_NOISE_RAD} rad)")
for row in servo_rows:
    print("  servo override %s kp %.4g forcerange %.4g armature %.4g damping %.4g"
          " frictionloss %.4g servo_body %s mass %.4g" % row)
print("armature", np.round(m.dof_armature[vadr], 4).tolist())
print("physics steps:", counter, "control steps:", len(saved_obs))
