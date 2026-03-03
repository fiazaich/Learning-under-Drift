#!/usr/bin/env bash
set -euo pipefail

python gaussian_additivity_experiment.py
python gaussian_saturation_experiment.py
python nn_drift_repro_experiment.py

# Plot NN results using the newest nn_multiT run
latest_nn=$(ls -1td out/nn_multiT_* 2>/dev/null | head -n1 || true)
if [[ -n "${latest_nn:-}" ]]; then
  python nn_repro_plotting.py --outdir "$latest_nn" --target delta_rep
else
  echo "No nn_multiT outputs found; skipping NN plotting" >&2
fi

python fisher_rao_footprint_demo.py
