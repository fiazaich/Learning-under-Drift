## Learning under Distributional Drift — Reproducibility Guide

Code for the Gaussian experiments, neural-network drift simulation, and Fisher–Rao footprint demo used in the paper **Learning under Distributional Drift: Reproducibility as an Intrinsic Statistical Resource** (S. Zaichyk).

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Gaussian experiments

1) Additivity / budget sweep  
```bash
python gaussian_additivity_experiment.py
```
Outputs to `results/gaussian_additivity_<tag>/` with CSVs plus `budget_scatter.(pdf|png|svg)`.

2) T-sweep saturation curves  
```bash
python gaussian_saturation_experiment.py
```
Writes `saturation_*.(pdf|png|svg)` under `results/gaussian_T_sweep_reflect_<tag>/`.

### Neural-network drift experiment

1) Generate data and metrics  
```bash
python nn_drift_repro_experiment.py --outdir nn_repro_results
```

2) Plot (point to the run directory that contains `figNN_additivity_raw.csv`)  
```bash
python nn_repro_plotting.py --outdir out/nn_multiT_YYYYMMDD_HHMMSS_xxxxxx --target delta_rep
# variants:
#   --collapse-mode residual
#   --do-3d
#   --T-hold <T>
```

### Fisher–Rao footprint demo

```bash
python fisher_rao_footprint_demo.py --T 4000 --d 5 --regime mixed --C-exo 2.0 --gamma 0.01 --k 0.25
```
Produces `fr_footprint_rate_demo.(pdf|png)` and `fr_footprint_contraction_demo.(pdf|png)` under `results/fr_footprint_<tag>/`. Use `--multi-seed N` to also get `fr_footprint_rate_scatter.(pdf|png)`.

### Quick checks

- `python -m py_compile *.py` to verify imports.
- Each script supports `--help` for full CLI options.

### One-shot run

```bash
./run_experiments.sh
```
Runs the Gaussian additivity and saturation experiments, the NN drift experiment, plots the newest `nn_multiT_*` run, and generates the Fisher–Rao footprint figures.

### Citation

If you use this code, please cite the accompanying paper:

```
@misc{zaichyk2026learningdistributionaldriftreproducibility,
      title={Learning under Distributional Drift: Reproducibility as an Intrinsic Statistical Resource}, 
      author={Sofiya Zaichyk},
      year={2026},
      eprint={2512.13506},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.13506}, 
}
```
