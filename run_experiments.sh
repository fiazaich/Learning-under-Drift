#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/private/tmp/pycache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/mpl-cache}"
export PYTHONPYCACHEPREFIX MPLCONFIGDIR

run_regional_feedback_observability() {
  (
    cd regional_feedback_exp
    "$PYTHON_BIN" observability_feedback_experiment.py
  )
}

"$PYTHON_BIN" gaussian_additivity_experiment.py
"$PYTHON_BIN" gaussian_saturation_experiment.py
"$PYTHON_BIN" experiments/nn_timing_balanced/balanced_timing_experiment.py run \
  --output-tag nn_timing_balanced
run_regional_feedback_observability

"$PYTHON_BIN" fisher_rao_footprint_demo.py \
  --T 2000 --d 5 --regime mixed --C-exo 2.0 --gamma 0.01 --k 0.25 \
  --kdim 2 --sigmaK 0.2 --extra-kernels \
  --burst --burst-period 600 --burst-hi 4.0 \
  --rate-window 60
