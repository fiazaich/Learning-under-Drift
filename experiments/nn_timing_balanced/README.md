# Balanced NN Timing-Allocation Experiment

This directory contains a standalone experiment for the Reviewer 2 revision.
It does not modify the original NN experiment scripts, and it writes outputs
under `experiments/nn_timing_balanced/runs/`.

Design:

- Cross horizon, total exogenous budget, policy-sensitive budget, and temporal allocation.
- Divide normalized time into 8 bins.
- Draw bin allocation weights from fixed allocation families at every horizon:
  `uniform`, `early`, `late`, and `bursty`.
- Cross simulation seeds with allocation seeds, so timing templates are not paired one-to-one with learner/data randomness.
- Use the same allocation template across horizons and budget levels for each allocation seed and concentration level.
- Distribute each bin's assigned budget uniformly within that bin.
- Record exogenous motion, policy-sensitive motion, actual total Fisher path length `A_T`, squared motion, max per-step motion, timing center of mass, and allocation entropy.
- Include a zero-drift exogenous budget control.
- Use one fixed empirical population to define the Fisher Gram matrix and evaluate population risks.
- Use reduced default budgets for a local-step regime:
  `C_exo/T in {0, 0.0025, 0.005, 0.01, 0.015}` and
  `gamma in {0, 0.001, 0.0025, 0.005}`.

Primary analysis:

- Fit the primary diagnostics on raw runs, not seed-averaged cells.
- Calibrate one risk-to-path constant on `A_T/T` using only the calibration split.
- Transport those constants without refitting to held-out horizons, exogenous budget levels, policy budget levels, allocation family, simulation seeds, and allocation seeds.
- Report coverage, median ratio, 95% ratio, and worst-case ratio for `V_T / B_T`.
- Compare the two principal theorem-aligned envelopes using the same calibrated constant:
  `A_T/T` and `dbar + kbar`.
- Treat fitted `alpha` as an empirical tightening only, not as the theoretical comparison constant.
- Keep fitted-alpha results in CSV/JSON diagnostics only, not the principal figure.
- Omitted-component failures are reported as counts in the locality diagnostics rather than as primary envelope models.
- Treat prequential-gap analysis as secondary; it is not the primary theorem-aligned test.

Primary output files:

- `balanced_timing_envelope_summary.csv`
- `balanced_timing_diagnostics.txt`
- `balanced_timing_envelope_results.json`
- `fig_envelope_heldout_ratios.{png,pdf}`

Run a smoke test:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pycache MPLCONFIGDIR=/private/tmp/mpl-cache \
  /usr/bin/python3 experiments/nn_timing_balanced/balanced_timing_experiment.py run --quick
```

Run the full experiment:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pycache MPLCONFIGDIR=/private/tmp/mpl-cache \
  /usr/bin/python3 experiments/nn_timing_balanced/balanced_timing_experiment.py run \
  --output-tag nn_timing_balanced
```

Analyze an existing run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pycache MPLCONFIGDIR=/private/tmp/mpl-cache \
  /usr/bin/python3 experiments/nn_timing_balanced/balanced_timing_experiment.py analyze \
  --run-dir experiments/nn_timing_balanced/runs/<run-directory>
```
