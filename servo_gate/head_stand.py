"""Stand-up regression with a head-offset overlay: zero command, BEST_WALK_ONNX_2 on any MJCF.

Same loop as harness.py (Playground mujoco_infer.py @ b9be205), with three differences:
 - command is all zeros (standing; slots 9-12 stay zero), 1500 control steps (30 s)
 - from --onset-s on, head offsets (fractions of each head joint's full range) are added to the
   policy's head targets before the rate clip; the overlaid targets are fed back as obs 83-96
 - writes stand.json with fall / tilt / drift / head tracking / obs feedback checks
Usage: head_stand.py <model.xml> <onnx> <outdir> --pg <pg> --head n,p,y,r [--seed S] [--servo spec]
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np
import onnxruntime

from camera_mass import WOZ_POS_HEAD, head_com_home, load_with_camera, parse_head_mass
from head_overlay import HEAD, HEAD_JOINTS, offsets_rad, overlay, parse_fracs
from servo import SpeedLimit, apply_spec, load_spec
from start_state import perturb_joints

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("onnx")
ap.add_argument("out")
ap.add_argument("--pg", required=True)
ap.add_argument("--head", default="0,0,0,0", help="head offsets as fractions of full joint range")
ap.add_argument("--onset-s", type=float, default=5.0)
ap.add_argument("--seed", type=int)
ap.add_argument("--servo")
ap.add_argument("--acc-offset", type=float, default=1.3,
                help="added to accel x in the obs; 0 matches training (joystick.py +1.3 is a no-op)")
ap.add_argument("--feedback", choices=("post", "pre"), default="post",
                help="obs 83-96: post = targets sent incl. head overlay; pre = policy's own targets (as trained)")
ap.add_argument("--camera-g", type=float, default=0.0, help="camera point mass on the head front face (camera_mass.py)")
ap.add_argument("--camera-pos", choices=("face", "woz"), default="face",
                help="face = front-face centreline point (default, earlier runs); woz = Woz ray-cast point")
ap.add_argument("--head-mass", action="append", default=[], metavar="G@X,Y,Z",
                help="extra head point mass, grams at mm in the head_assembly frame; negative = removal; repeatable")
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.pg, "playground", "common"))
from poly_reference_motion_numpy import PolyReferenceMotion  # noqa: E402

SIM_DT, DECIM, N_CTRL = 0.002, 10, 1500
DOF_VEL_SCALE, ACTION_SCALE, MAX_MOTOR_VEL = 0.05, 0.25, 5.24
FALL_TILT_DEG, FALL_Z = 45.0, 0.08
COMMANDS = np.zeros(7)

m = load_with_camera(a.model, a.camera_g, WOZ_POS_HEAD if a.camera_pos == "woz" else None,
                     [parse_head_mass(s) for s in a.head_mass])
head_mass0, head_com0 = head_com_home(load_with_camera(a.model, 0))
head_mass, head_com = head_com_home(m)
m.opt.timestep = SIM_DT
speed_limit = None
if a.servo:
    spec = load_spec(a.servo)
    apply_spec(m, spec)
    speed_limit = SpeedLimit(m, spec)
d = mujoco.MjData(m)
if speed_limit is not None:
    speed_limit.apply(m, d)
mujoco.mj_step(m, d)

names = [m.actuator(k).name for k in range(m.nu)]
pol = [k for k, n in enumerate(names) if "antenna" not in n]
assert tuple(names[k] for k in pol[5:9]) == HEAD_JOINTS, [names[k] for k in pol[5:9]]
jids = [m.actuator_trnid[k, 0] for k in pol]
qadr = np.array([m.jnt_qposadr[j] for j in jids])
vadr = np.array([m.jnt_dofadr[j] for j in jids])
head_pol = np.array(pol)[HEAD]
ctrl_lo, ctrl_hi = m.actuator_ctrlrange[head_pol, 0], m.actuator_ctrlrange[head_pol, 1]
fracs = parse_fracs(a.head)
off = offsets_rad(fracs, m.jnt_range[np.array(jids)[HEAD]])
key = m.keyframe("home")
default = key.ctrl[pol].copy()
d.qpos[:] = key.qpos if a.seed is None else perturb_joints(key.qpos, qadr, a.seed)
d.ctrl[:] = key.ctrl
motor_targets = default.copy()
prev_motor_targets = default.copy()
sent, prev_sent = default.copy(), default.copy()  # ctrl incl. overlay (pre mode only)

gyro_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")]
acc_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "accelerometer")]
floor_b, lf_b, rf_b = (d.body(n).id for n in ("floor", "foot_assembly", "foot_assembly_2"))


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
imitation_i = 0
onset = int(round(a.onset_s / (SIM_DT * DECIM)))
log = {"base": [], "tilt": [], "head_q": [], "head_t": [], "feet": [], "fb_err": [], "knee_f": []}
clip_hits, applied_ctrl, step = 0, None, 0
counter = 0
while counter < N_CTRL * DECIM:
    if speed_limit is not None:
        speed_limit.apply(m, d)
    mujoco.mj_step(m, d)
    log["knee_f"].append(d.actuator_force[[pol[3], pol[12]]].copy())
    counter += 1
    if counter % DECIM:
        continue
    imitation_i = (imitation_i + 1.0) % PRM.nb_steps_in_period
    ang = imitation_i / PRM.nb_steps_in_period * 2 * np.pi
    acc = d.sensordata[acc_adr:acc_adr + 3].copy()
    acc[0] += a.acc_offset
    feet = np.array([contact(lf_b, floor_b), contact(rf_b, floor_b)], dtype=float)
    obs = np.concatenate([
        d.sensordata[gyro_adr:gyro_adr + 3], acc, COMMANDS,
        d.qpos[qadr] - default, d.qvel[vadr] * DOF_VEL_SCALE,
        last, last2, last3, motor_targets, feet, [np.cos(ang), np.sin(ang)],
    ])
    if applied_ctrl is not None:  # obs 83-96 must be the expected targets from last step
        log["fb_err"].append(float(np.max(np.abs(obs[83:97] - applied_ctrl))))
    action = sess.run(None, {"obs": [obs]})[0][0]
    last3, last2, last = last2.copy(), last.copy(), action.copy()
    targets = default + action * ACTION_SCALE
    dmax = MAX_MOTOR_VEL * (SIM_DT * DECIM)
    if a.feedback == "post":  # overlay before the rate clip; obs sees what was sent
        if step >= onset:
            targets, hits = overlay(targets, off, ctrl_lo, ctrl_hi)
            clip_hits += hits
        motor_targets = np.clip(targets, prev_motor_targets - dmax, prev_motor_targets + dmax)
        prev_motor_targets = motor_targets.copy()
        d.ctrl[pol] = motor_targets.copy()
        applied_ctrl = d.ctrl[pol].copy()
    else:  # policy chain exactly as trained; overlay only on the ctrl, rate-clipped on its own
        motor_targets = np.clip(targets, prev_motor_targets - dmax, prev_motor_targets + dmax)
        prev_motor_targets = motor_targets.copy()
        sent = motor_targets.copy()
        if step >= onset:
            sent, hits = overlay(sent, off, ctrl_lo, ctrl_hi)
            clip_hits += hits
        sent = np.clip(sent, prev_sent - dmax, prev_sent + dmax)
        prev_sent = sent.copy()
        d.ctrl[pol] = sent
        applied_ctrl = motor_targets.copy()
    zb = d.xmat[1].reshape(3, 3)[:, 2]
    log["tilt"].append(float(np.degrees(np.arccos(np.clip(zb[2], -1, 1)))))
    log["base"].append(d.qpos[:7].copy())
    log["head_q"].append(d.qpos[qadr[HEAD]].copy())
    log["head_t"].append(d.ctrl[np.array(pol)[HEAD]].copy())
    log["feet"].append(feet)
    step += 1

base = np.array(log["base"])
tilt = np.array(log["tilt"])
hq, ht = np.array(log["head_q"]), np.array(log["head_t"])
steady = slice(onset + 250, None)  # 5 s after onset to the end


def yaw(q):
    w, x, y, z = q
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


kf = np.array(log["knee_f"])[(onset + 250) * DECIM:]
res = {
    "mujoco": mujoco.__version__, "seed": a.seed, "head_fracs": fracs.tolist(),
    "acc_offset": a.acc_offset, "feedback": a.feedback, "camera_g": a.camera_g, "camera_pos": a.camera_pos,
    "head_mass_add": a.head_mass,
    "head_mass_g": round(head_mass * 1000, 2),
    "head_com_shift_mm_home": np.round((head_com - head_com0) * 1000, 3).tolist(),
    "head_offset_rad": np.round(off, 4).tolist(), "onset_s": a.onset_s,
    "fell": bool(np.any(tilt > FALL_TILT_DEG) or np.any(base[:, 2] < FALL_Z)),
    "max_tilt_deg_after_onset": round(float(tilt[onset:].max()), 2),
    "mean_tilt_deg_steady": round(float(tilt[steady].mean()), 2),
    "min_base_z": round(float(base[:, 2].min()), 4),
    "xy_drift_mm_after_onset": round(float(np.linalg.norm(base[-1, :2] - base[onset, :2]) * 1000), 1),
    "yaw_drift_deg_after_onset": round(float(yaw(base[-1, 3:7]) - yaw(base[onset, 3:7])), 2),
    "both_feet_down_frac_steady": round(float(np.mean(np.all(np.array(log["feet"])[steady] > 0, axis=1))), 3),
    "head_target_minus_default_steady": np.round(ht[steady].mean(0) - default[HEAD], 3).tolist(),
    "head_q_minus_default_steady": np.round(hq[steady].mean(0) - default[HEAD], 3).tolist(),
    "head_tracking_err_rms_steady": np.round(np.sqrt(np.mean((hq[steady] - ht[steady]) ** 2, axis=0)), 3).tolist(),
    "head_ctrlrange_clip_hits": clip_hits,
    "obs_feedback_max_abs_err": max(log["fb_err"]),
    "knee_rms_nm_steady": np.round(np.sqrt(np.mean(kf ** 2, axis=0)), 3).tolist(),
}
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(f"{a.out}/stand.json", "w"), indent=1)
np.savez(f"{a.out}/stand_logs.npz", base=base, tilt=tilt, head_q=hq, head_t=ht, feet=np.array(log["feet"]))
print(json.dumps(res))
