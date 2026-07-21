## Learning under Distributional Drift — Reproducibility Guide

Code for the Gaussian experiments, neural-network drift simulation, and Fisher–Rao footprint demo used in the paper **Learning under Distributional Drift: Prequential Reproducibility as an Intrinsic Statistical Resource** (S. Zaichyk).

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

2) T-sweep components overlay
```bash
python gaussian_saturation_experiment.py
```
Writes `components_overlay.(pdf|png|svg)` under `results/gaussian_T_sweep_reflect_<tag>/`.

### Neural-network drift experiment

```bash
python experiments/nn_timing_balanced/balanced_timing_experiment.py run \
  --output-tag nn_timing_balanced
```
Writes raw runs, envelope diagnostics, and figures under `experiments/nn_timing_balanced/runs/`.

### Fisher–Rao footprint demo

```bash
python fisher_rao_footprint_demo.py \
  --T 2000 --d 5 --regime mixed --C-exo 2.0 --gamma 0.01 --k 0.25 \
  --kdim 2 --sigmaK 0.2 --extra-kernels \
  --burst --burst-period 600 --burst-hi 4.0 \
  --rate-window 60
```
Produces `fr_footprint_rate_demo.(pdf|png)` and `fr_footprint_contraction_demo.(pdf|png)` under `results/fr_footprint_<tag>/`. Use `--multi-seed N` to also get `fr_footprint_rate_scatter.(pdf|png)`.

### Regional feedback / observability experiment

```bash
cd regional_feedback_exp
python observability_feedback_experiment.py
```
Requires `US_Regional_Sales_Data.csv` in `regional_feedback_exp/`. This single entry point runs the regional feedback observability experiment and writes outputs to `regional_feedback_exp/regional_feedback_outputs/`.

Main figures:

- `figure_main_channel_association_summary.(pdf|png|svg)`: raw observed Fisher-rate association with \(V_T\), \(\Delta_T^{\mathrm{preq}}\), and per-transition \(v_t\).
- `figure_main_step_fr_association.(pdf|png|svg)`: per-transition Spearman association between channel Fisher motion and \(v_t\).
- `figure_main_feedback_manipulation_check.(pdf|png|svg)`: feedback-strength manipulation check for risk quantities and raw observed Fisher rates.

Supporting CSVs are also written in the same directory: run-level results, round-level records, condition summaries, raw-rate regressions, step diagnostics, and `main_figure_values_summary.txt`.

### Quick checks

- `python -m py_compile *.py` to verify imports.
- Each script supports `--help` for full CLI options.

### One-shot run

```bash
./run_experiments.sh
```
Runs the Gaussian additivity and saturation experiments, the balanced NN envelope experiment, the regional feedback / observability experiment, and the Fisher–Rao footprint figures.

### Citation

If you use this code, please cite the accompanying paper:

```
@misc{zaichyk2026learningdistributionaldriftprequential,
      title={Learning under Distributional Drift: Prequential Reproducibility as an Intrinsic Statistical Resource}, 
      author={Sofiya Zaichyk},
      year={2026},
      eprint={2512.13506},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.13506}, 
}
```
