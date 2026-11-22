#!/bin/bash
# Run every sts3046 spec, 4 at a time; each run_gate.sh backs its folder up.
cd "$(dirname "$0")"
ls specs/sts3046_*.json | xargs -P 4 -I{} bash -c './run_gate.sh {} > "$OUT_ROOT/log_$(basename {} .json).txt" 2>&1; echo "$(basename {}) exit $?"'
