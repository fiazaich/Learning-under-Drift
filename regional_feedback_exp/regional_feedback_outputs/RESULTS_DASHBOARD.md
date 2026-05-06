# Regional Feedback Results Dashboard

Open this file first. The public result is operational observability, not calibrated severity estimation.

## Purpose

Raw observed Fisher motion is an uncalibrated but meaningful channel-level footprint of drift-induced risk motion.
The experiment deliberately uses simple fixed monitoring channels rather than optimized residual/error/loss-aware channels.
Raw observed Fisher rates are computed directly from monitoring streams, without using feedback strength as a predictor and without subtracting a matched zero-feedback counterfactual.

## Primary Evidence

1. `figure_main_operational_raw_rate_r2_heatmap.*`: raw-rate single-channel regression $R^2$ by target and channel.
   This is the main operational diagnostic: no feedback-strength covariate, no excess-rate subtraction, and no zero-feedback baseline matching.
2. `figure_main_step_fr_association.*`: per-transition Spearman association between per-step observed FR and $v_t$, sorted by channel strength.
   This shows channel dependence without using individual-run scatter plots.
3. `figure_main_feedback_manipulation_check.*`: a secondary manipulation check that feedback strength moves risk quantities and raw observed FR rates.

## Raw-Rate Regression Highlights

Best channel by target in the operational raw-rate $R^2$ heatmap:

| target_label | channel_label | r2_in_sample |
| --- | --- | --- |
| loss motion | Task minus cost | 0.936 |
| error motion | Task minus cost | 0.919 |
| $V_T$ | Task minus cost | 0.855 |
| $\Delta_T^{\mathrm{rep}}$ | Task minus cost | 0.529 |
| $\Delta_T^{\mathrm{sam}}$ | Task minus cost | 0.089 |

## Per-Transition Association

Bars in `figure_main_step_fr_association.*` report Spearman $\rho$ over positive-feedback transitions.

| channel_label | n_steps | spearman |
| --- | --- | --- |
| Task minus subgroup | 570 | 0.578 |
| Task-aligned | 570 | 0.573 |
| Task minus cost | 570 | 0.569 |
| Weak blind | 570 | 0.462 |
| Task minus quantity | 570 | 0.304 |
| Coarse score | 570 | 0.162 |

## Manipulation Check Means

These means support that the feedback intervention moves the system; they are not the main observability evidence.

| $\mu$ | $V_T$ mean | $\Delta_T^{\mathrm{rep}}$ mean | Weak blind raw FR mean | Coarse score raw FR mean | Task-aligned raw FR mean |
| --- | --- | --- | --- | --- | --- |
| 0.000 | 1485.754 | 2682.794 | 0.003 | 0.005 | 0.050 |
| 0.050 | 2632.074 | 3483.200 | 0.006 | 0.043 | 0.098 |
| 0.100 | 5082.420 | 4750.336 | 0.010 | 0.057 | 0.126 |
| 0.200 | 1.581e+04 | 1.534e+04 | 0.013 | 0.070 | 0.169 |

## Interpretation

Observable FR is a contracted footprint, not an estimate of the full intrinsic $C_T/T$ budget.
The footprint is channel-dependent and target-dependent.
The strongest alignment is with $V_T$ and direct loss/error motion; the weaker relationship with $\Delta_T^{\mathrm{rep}}$ is expected because the prequential gap includes sampling deviation and possible cancellation.
Weak association with $\Delta_T^{\mathrm{sam}}$ is reassuring: raw observed FR is not simply tracking generic sampling noise.
The strongest task channel may be an ablation such as Task minus cost or Task minus subgroup. This is not a failure; adding coordinates can introduce sparsity, nuisance variation, or cancellation.

## Calibration Caveat

Raw observed FR is not a calibrated estimator of run-level $V_T$.
Calibration or thresholding would require a deployment-specific operating-envelope layer using replay, simulation, or held-out monitoring.
That calibration problem is future work; this experiment establishes the observability principle.

## Appendix File Map

| File | Use |
| --- | --- |
| `RESULTS_DASHBOARD.md` | Start here. Human-readable summary. |
| `README.md` | Main vs appendix output map. |
| `main_figure_values_summary.txt` | Exact values used in the main figures. |
| `figure_main_operational_raw_rate_r2_heatmap.*` | Primary raw-rate regression heatmap. |
| `figure_main_step_fr_association.*` | Primary transition-level channel association figure. |
| `figure_main_feedback_manipulation_check.*` | Secondary manipulation check. |
| `table_operational_raw_rate_diagnostics.csv` | Main deployable raw-rate and per-step diagnostic table. |
| `regional_feedback_step_diagnostics.csv` | Per-transition FR vs risk/motion diagnostics. |
| `regional_feedback_summary.csv` | Main metrics by mu. |
| `regional_feedback_rounds.csv` | Round-level raw data. Large/detail file. |
| `regional_feedback_results_by_seed.csv` | Condition/seed-level raw data. |
| `appendix_diagnostics/` | Excess-rate, leave-one-$\mu$-out, within-$\mu$, individual-run scatter, and condition-mean descriptive diagnostics. |
