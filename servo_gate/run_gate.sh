#!/bin/bash
# Run the servo gate for one spec on the Playground model and back the folder up.
# Usage: run_gate.sh <spec.json>
# Env (all required): PY (python with mujoco + onnxruntime), PG_ROOT (Playground checkout),
#   ONNX (policy file), OUT_ROOT (local run folders), GATE_BACKUP_URI (gs://.../razer).
#   NO_BACKUP=1 skips the upload explicitly. SEED=<int> perturbs the start pose; the run is named <name>_s<seed>.
#   ACC_OFFSET=<float> sets the accel-x obs offset (default 1.3, as mujoco_infer.py); the run is named <name>_acc<offset>.
set -euo pipefail
SPEC=$(realpath "$1")
: "${PY:?}" "${PG_ROOT:?}" "${ONNX:?}" "${OUT_ROOT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
NAME=$("$PY" -c "import sys; sys.path.insert(0, '$HERE'); from servo import load_spec; print(load_spec('$SPEC')['name'])")
MODEL=$PG_ROOT/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml
SEED_ARG=""
if [ -n "${SEED:-}" ]; then NAME=${NAME}_s$SEED; SEED_ARG="--seed $SEED"; fi
if [ -n "${ACC_OFFSET:-}" ]; then NAME=${NAME}_acc$ACC_OFFSET; SEED_ARG="$SEED_ARG --acc-offset $ACC_OFFSET"; fi
OUT=$OUT_ROOT/$NAME
if [ -e "$OUT" ]; then echo "refusing to overwrite $OUT" >&2; exit 2; fi
mkdir -p "$OUT"
cp "$SPEC" "$OUT/servo_spec.json"
CMD="ulimit -v 3000000; $PY $HERE/harness.py $MODEL $ONNX $OUT --pg $PG_ROOT --servo $OUT/servo_spec.json $SEED_ARG && $PY $HERE/analyze.py $OUT"
echo "$CMD" > "$OUT/COMMAND.txt"
{
  echo "quack-ml $(git -C "$HERE" describe --always --dirty --abbrev=40)"
  echo "playground $(git -C "$PG_ROOT" rev-parse HEAD)"
  echo "onnx sha256 $(sha256sum "$ONNX" | cut -d' ' -f1)"
  "$PY" -c "import mujoco, onnxruntime, numpy; print('mujoco', mujoco.__version__, 'onnxruntime', onnxruntime.__version__, 'numpy', numpy.__version__)"
} > "$OUT/versions.txt"
bash -c "$CMD" > "$OUT/stdout.txt" 2>&1 || { echo "run failed, see $OUT/stdout.txt" >&2; exit 1; }
if [ "${NO_BACKUP:-0}" = 1 ]; then echo "backup skipped (NO_BACKUP=1)"; exit 0; fi
: "${GATE_BACKUP_URI:?}"
timeout 300 gcloud storage cp -r "$OUT" "$GATE_BACKUP_URI/"
echo "backed up $(timeout 120 gcloud storage ls "$GATE_BACKUP_URI/$NAME/" | wc -l) files to $GATE_BACKUP_URI/$NAME/"
