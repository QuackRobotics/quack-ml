"""Knee torque RMS (overall/stance/swing, t >= 20 s), knee tracking and crouch for harness run folders.

Usage: knee_score.py <run_dir> [<run_dir> ...]   prints one JSON line per run.
Needs actuator_force.npy, feet_contact.npy (left, right), motor_targets.npy, joint_qpos.npy, gates.json.
"""
import json
import sys

import numpy as np

from knee_split import phase_rms

DT, DECIM, T0 = 0.002, 10, 20.0

for run in sys.argv[1:]:
    names = open(f"{run}/actuator_names.txt").read().split()
    force = np.load(f"{run}/actuator_force.npy")
    contact = np.load(f"{run}/feet_contact.npy")
    targets, qpos = np.load(f"{run}/motor_targets.npy"), np.load(f"{run}/joint_qpos.npy")
    gates = json.load(open(f"{run}/gates.json"))
    steady = np.arange(1, len(targets) + 1) * DT * DECIM >= T0
    row = {"run": run, "fell": gates["fell"], "body_vx": gates["body_vx_steady"]}
    for side, foot in (("left", 0), ("right", 1)):
        k = names.index(f"{side}_knee")
        row[f"{side}_knee_rms"] = {key: round(v, 4) for key, v in phase_rms(force[:, k], contact[:, foot], DECIM, DT, T0).items()}
        err = targets[steady, k] - qpos[steady, k]
        row[f"{side}_knee_track_rms_rad"] = round(float(np.sqrt(np.mean(err ** 2))), 4)
        row[f"{side}_knee_qpos_mean_rad"] = round(float(np.mean(qpos[steady, k])), 4)
    print(json.dumps(row))
