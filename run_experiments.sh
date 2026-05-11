#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" gaussian_additivity_experiment.py
"$PYTHON_BIN" gaussian_saturation_experiment.py
"$PYTHON_BIN" nn_drift_repro_experiment.py

# Plot NN results using the newest nn_multiT run
latest_nn=$(ls -1td out/nn_multiT_* 2>/dev/null | head -n1 || true)
if [[ -n "${latest_nn:-}" ]]; then
  "$PYTHON_BIN" nn_repro_plotting.py --outdir "$latest_nn" --target delta_rep
else
  echo "No nn_multiT outputs found; skipping NN plotting" >&2
fi

"$PYTHON_BIN" fisher_rao_footprint_demo.py \
  --T 2000 --d 5 --regime mixed --C-exo 2.0 --gamma 0.01 --k 0.25 \
  --kdim 2 --sigmaK 0.2 --extra-kernels \
  --burst --burst-period 600 --burst-hi 4.0 \
  --rate-window 60
