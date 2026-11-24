"""Gate metrics for one run folder (actuator_force.npy, base_qpos.npy per physics step).

Usage: analyze.py <outdir> [<cap_Nm>]
Cap hits use each actuator's own limit from actuator_forcerange.npy when the run saved it;
otherwise <cap_Nm> applies to every actuator (runs made before per-group servo specs).
Steady window t >= 20 s. Physics dt 0.002 s. Body-forward vx sampled at the 50 Hz control rate. Path speed = mean 10 s chord length / 10 s.
Writes per_joint.csv (full run), per_joint_steady.csv, base_trace.csv (50 Hz), gates.json.
"""
import json
import os
import sys

import numpy as np

from body_vx import body_forward_speed

out = sys.argv[1]
DT, T0 = 0.002, 20.0
F = np.load(f"{out}/actuator_force.npy")
B = np.load(f"{out}/base_qpos.npy")
names = open(f"{out}/actuator_names.txt").read().split()
assert F.shape[1] == len(names), (F.shape, names)
if os.path.exists(f"{out}/actuator_forcerange.npy"):
    caps = np.abs(np.load(f"{out}/actuator_forcerange.npy")[:, 1])
elif len(sys.argv) > 2:
    caps = np.full(len(names), float(sys.argv[2]))
else:
    sys.exit("no actuator_forcerange.npy in run folder: pass <cap_Nm>")
n = len(B)
t = np.arange(1, n + 1) * DT
w, x, y, z = B[:, 3], B[:, 4], B[:, 5], B[:, 6]
yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
px, py, pz = B[:, 0], B[:, 1], B[:, 2]
np.savetxt(f"{out}/base_trace.csv", np.c_[t, px, py, pz, np.degrees(yaw)][::10],
           delimiter=",", header="t,x,y,z,yaw_deg", comments="", fmt="%.5f")
s = t >= T0
ts = t[s][-1] - t[s][0]
idx = np.arange(np.argmax(s), n, int(10.0 / DT))
chords = [np.hypot(px[j] - px[i], py[j] - py[i]) for i, j in zip(idx[:-1], idx[1:])]
g = {
    "steady_window_s": [round(float(t[s][0]), 3), round(float(t[s][-1]), 3)],
    "world_vx_full": (px[-1] - px[0]) / (t[-1] - t[0]),
    "world_vx_steady": (px[s][-1] - px[s][0]) / ts,
    "body_vx_steady": body_forward_speed(px[s][::10], py[s][::10], yaw[s][::10], 10 * DT),
    "path_speed_steady": float(np.mean(chords)) / 10.0,
    "world_y_drift_per_10s_steady": (py[s][-1] - py[s][0]) / ts * 10,
    "heading_total_deg": float(np.degrees(yaw[-1] - yaw[0])),
    "heading_drift_deg_per_10s_steady": float(np.degrees(yaw[s][-1] - yaw[s][0]) / ts * 10),
    "heading_change_per_10s_window_deg": [round(float(v), 1) for v in np.diff(np.degrees(yaw[idx]))],
    "dx_total_m": px[-1] - px[0], "dy_total_m": py[-1] - py[0],
    "z_first": float(pz[0]), "z_min": float(pz.min()), "fell": bool(pz.min() < 0.08),
}


def table(Fw):
    rows = []
    for k, nm in enumerate(names):
        f = np.abs(Fw[:, k])
        rows.append([nm, caps[k], f.max(), np.sqrt(np.mean(f ** 2)), 100 * np.mean(f > 0.49),
                     100 * np.mean(f > 1.91), int(np.sum(f >= caps[k] - 1e-6))])
    return rows


for fn, Fw in (("per_joint.csv", F), ("per_joint_steady.csv", F[s])):
    with open(f"{out}/{fn}", "w") as fh:
        fh.write("actuator,cap_Nm,peak_abs_Nm,rms_Nm,pct_gt_0.49,pct_gt_1.91,steps_at_cap\n")
        for r in table(Fw):
            fh.write(f"{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.2f},{r[5]:.2f},{r[6]}\n")
g["steady_knee_hip_pitch"] = {r[0]: {"cap": round(r[1], 4), "peak": round(r[2], 4), "rms": round(r[3], 4),
                                     "pct_gt_1.91": round(r[5], 2), "cap_hits": r[6]}
                              for r in table(F[s]) if "knee" in r[0] or "hip_pitch" in r[0]}
g = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in g.items()}
json.dump(g, open(f"{out}/gates.json", "w"), indent=1)
print(json.dumps(g, indent=1))
print("per_joint_steady.csv (t>=20 s):")
print(open(f"{out}/per_joint_steady.csv").read())
