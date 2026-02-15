#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Running gaussian_regime_recovery.py (trajectory population risk)"
python3 gaussian_regime_recovery.py --metric traj_error

echo "[2/3] Running gaussian_additivity.py (trajectory population risk)"
python3 gaussian_additivity.py --metric traj_error

echo "[3/3] Running gaussian_geometry_natural_gradient.py"
python3 gaussian_geometry_natural_gradient.py

echo "Gaussian experiments completed."
