## Drift Experiments — Reproducibility Guide

This repository bundles the exact scripts used for the paper’s Gaussian toy-model experiments and the neural-network simulation. Everything runs on CPU; no external datasets are required.

### Environment

1. Use Python 3.9+.
2. Install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   PyTorch is listed with a CPU wheel; feel free to swap the install command to match your platform (CUDA/cuDNN) if needed.

Typical runtimes on a 2021 MacBook Pro (M1 Pro, single-threaded):

- Gaussian suite — ~6 minutes total (each script ≤2 minutes).
- NN suite — ~30 minutes for the simulation; plotting completes in <1 minute.

### Reproducing Figures

Two helper scripts orchestrate the full runs. They drop outputs under `out/<experiment>_<timestamp>_<id>/`.

#### Gaussian suite (Figures 1–3)

```bash
./run_gaussian_figures.sh
```

This sequentially executes:
- `gaussian_regime_recovery.py` — regime comparison,
- `gaussian_additivity.py` — budget/additivity sweep,
- `gaussian_geometry_natural_gradient.py` — geometry experiment.

Each script writes summaries (+ CSV/JSON) plus publication-style figures into a fresh directory inside `out/`.

#### Neural-network suite (Figure 4)

```bash
./run_nn_with_plotting.sh
```

This script (i) runs `NN_test_balanced.py` to generate the raw/summary tables and (ii) automatically invokes `NN_plotting_tool.py` on the newest `out/nn_multiT_*` directory to regenerate all NN plots.
To experiment with different learner widths, call the simulator directly, e.g. `python3 NN_test_balanced.py --hidden-dim 64`, then rerun `NN_plotting_tool.py` on the resulting directory.

### Manual usage

- Every Python file exposes a CLI (`python3 script.py --help`) so you can tweak metrics or inputs.
- `NN_plotting_tool.py` can be pointed at any prior run:
  ```bash
  python3 NN_plotting_tool.py \
    --summary out/nn_multiT_YYYYMMDD_HHMMSS_xxxxxx/figNN_additivity_summary.csv \
    --meta    out/nn_multiT_YYYYMMDD_HHMMSS_xxxxxx/figNN_additivity_meta.json \
    --raw     out/nn_multiT_YYYYMMDD_HHMMSS_xxxxxx/figNN_additivity_raw.csv
  ```

### Quick validation

For a smoke test, run `python3 -m py_compile *.py` (already used in automation) or execute the helper scripts; both rely solely on the files tracked here and therefore confirm the install is healthy.

### Notes

- All numerical seeds and grids are defined at the top of each script; for convenience they’re mirrored in `configs/gaussian.yaml` and `configs/nn.yaml`. Leave them untouched to reproduce the paper’s figures.
- Outputs are intentionally timestamped to avoid overwriting previous runs; delete `out/` if you need a clean slate.

### Citation

If you build on this code, please cite the accompanying paper:

```
@article{YOURKEY2025,
  title   = {Learning under Drift: Speed Limits for Feedback-Rich Environments},
  author  = {A. Author and B. Author},
  journal = {NeurIPS},
  year    = {2025}
}
```
