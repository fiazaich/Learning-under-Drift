#!/usr/bin/env bash
set -euo pipefail

echo "[1/2] Running NN_test_balanced.py"
python3 NN_test_balanced.py

echo "[2/2] Plotting latest NN results"
latest_dir="$(ls -dt out/nn_multiT_* 2>/dev/null | head -n1 || true)"
if [[ -z "${latest_dir}" ]]; then
    echo "No nn_multiT_* directory found under ./out. Did the simulation run succeed?"
    exit 1
fi

python3 NN_plotting_tool.py \
    --summary "${latest_dir}/figNN_additivity_summary.csv" \
    --meta "${latest_dir}/figNN_additivity_meta.json" \
    --raw "${latest_dir}/figNN_additivity_raw.csv"

echo "NN simulation and plotting finished. Results: ${latest_dir}"
