# servo_gate

CPU-only sim gate for leg servo candidates on the Playground Open Duck Mini v2 model
(Playground `b9be205`, policy `BEST_WALK_ONNX_2.onnx`, sha256 `3c606f93?`, 0.15 m/s forward command,
5000 control steps at 50 Hz). No training, no GPU.

- `harness.py`: policy loop copied step for step from Playground `mujoco_infer.py`; its first 50
  observations match Playground's own loop exactly (max abs diff 0.0).
- `servo.py`: `--servo <spec.json>` sets forcerange, kp and armature per leg joint group
  (`hip_yaw`, `hip_roll`, `hip_pitch`, `knee`, `ankle`, left and right together) on the compiled
  model. Omitted groups and fields keep Playground's values (kp 13.37, ?3.23 Nm, armature 0.027).
  Head actuators are never touched.
- `analyze.py`: path speed, world vx, y drift, heading drift, per-joint RMS/peak and cap hits
  (counted against each actuator's own limit), steady window t ? 20 s.
- `run_gate.sh <spec.json>`: runs both under a 3 GB memory cap, writes COMMAND.txt, stdout.txt,
  versions.txt and the spec into `$OUT_ROOT/<name>/`, then uploads it to `$GATE_BACKUP_URI/<name>/`.
- `make_pg_cap.py`: the XML edit behind the 2026-11-22 formal gate run `pg_cap191`;
  kept so the test can prove `specs/pg_cap191.json` reproduces it.

## Spec

```json
{"name": "servo_x", "source": "datasheet / measurement reference",
 "groups": {"knee": {"forcerange": 3.0, "kp": 15.0, "armature": 0.03}}}
```

forcerange is the symmetric limit in Nm, kp the position gain in Nm/rad, armature the reflected
rotor inertia in kg?m?. Unknown groups or fields, and non-positive values, are rejected.

## Run

```bash
export PY=~/quack-core/hinton/.venv/bin/python PG_ROOT=<playground checkout> ONNX=<BEST_WALK_ONNX_2.onnx> \
       OUT_ROOT=<run folders> GATE_BACKUP_URI=gs://<bucket>/quack/torque_v2/razer
servo_gate/run_gate.sh servo_gate/specs/pg_cap191.json
PG_ROOT=<playground checkout> $PY -m unittest discover -s servo_gate/tests -v
```

Gate (Atlas): path speed ? 0.12 m/s with heading drift reported, knee RMS within Woz's thermal bands
(? 0.49 Nm OK; 0.49?0.6 only with ? 50 ?C case monitoring and firmware cutback; > 0.7 reject).
Hip pitch is reported against the knee bands as an assumption until Woz gives a hip limit.
No policy ships without sim2real validation on the physical robot.
