## Learning under Distributional Drift— Reproducibility Guide

This repository bundles the exact scripts used for the Gaussian toy-model experiments and the neural-network simulation used in the paper **Learning under Distributional Drift: Reproducibility as an
Intrinsic Statistical Resource** by Sofiya Zaichyk.

### Environment

1. Use Python 3.9+.
2. Install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   PyTorch is listed with a CPU wheel; feel free to swap the install command to match your platform (CUDA/cuDNN) if needed.

### Reproducing Figures

Two helper scripts orchestrate the full runs. They drop outputs under `out/<experiment>_<timestamp>_<id>/`.

#### Gaussian suite (Figures 3a and 3b)

```bash
./run_gaussian_figures.sh
```

This sequentially executes:
- `gaussian_regime_recovery.py` — regime comparison,
- `gaussian_additivity.py` — budget/additivity sweep,
- `gaussian_geometry_natural_gradient.py` — geometry experiment.

Each script writes summaries (+ CSV/JSON) plus publication-style figures into a fresh directory inside `out/`.

#### Neural-network suite (Figures 4a and 4b)

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
@article{YZaichyk2025,
  title   = {Learning under Distributional Drift: Reproducibility as an
Intrinsic Statistical Resource},
  author  = {S. Zaichyk},
  journal = {},
  year    = {2025}
}
```
