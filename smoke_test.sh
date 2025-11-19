#!/usr/bin/env bash
set -euo pipefail

echo "Compiling all experiment scripts..."
PYTHONPYCACHEPREFIX=. python3 -m py_compile \
    NN_test_balanced.py \
    NN_plotting_tool.py \
    gaussian_regime_recovery.py \
    gaussian_additivity.py \
    gaussian_geometry_natural_gradient.py \
    common_sim.py \
    out_utils.py

echo "Smoke test finished: files compile successfully."
