#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Running gaussian_regime_recovery.py"
python3 gaussian_regime_recovery.py

echo "[2/3] Running gaussian_additivity.py"
python3 gaussian_additivity.py

echo "[3/3] Running gaussian_geometry_natural_gradient.py"
python3 gaussian_geometry_natural_gradient.py

echo "Gaussian experiments completed."
