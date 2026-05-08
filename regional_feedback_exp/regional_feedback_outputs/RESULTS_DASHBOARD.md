# Regional Feedback Results Dashboard

Open this file first. The public result is operational observability, not calibrated severity estimation.

## Purpose

Raw observed Fisher motion is an uncalibrated but meaningful channel-level footprint of drift-induced risk motion.
The experiment deliberately uses simple fixed monitoring channels rather than optimized residual/error/loss-aware channels.
Raw observed Fisher rates are computed directly from monitoring streams, without using feedback strength as a predictor and without subtracting a matched zero-feedback counterfactual.

## Primary Evidence

1. `figure_main_channel_association_summary.*`: run-level raw-rate $R^2$ and transition-level Spearman $\rho$ by channel.
   Channels are compared on their association with the paper-aligned drift quantities: $V_T$, $\Delta_T^{\mathrm{rep}}$, and $v_t$.
2. `figure_main_step_fr_association.*`: per-transition Spearman association between per-step observed FR and $v_t$, including the null blind channel.
   This shows transition-level observability and channel dependence without using individual-run scatter plots.
3. `figure_main_feedback_manipulation_check.*`: a secondary manipulation check that feedback strength moves risk quantities and raw observed FR rates.

## Main Figure Caption

Channels are compared on their association with the paper-aligned drift quantities. Panel (a) reports run-level raw-rate $R^2$ for $V_T$ and $\Delta_T^{\mathrm{rep}}$. Panel (b) reports transition-level Spearman association with $v_t$. The null blind channel is a fixed row-bucket negative control. Pointwise loss-motion diagnostics are reported separately in the appendix because they are unsigned cancellation-free quantities.

## Channel Association Highlights

The null blind channel is a fixed row/hash partition and has no observed Fisher motion. Coarse score is the prediction-score baseline.
Primary drift quantities are $V_T$ and $v_t$; $\Delta_T^{\mathrm{rep}}$ is reported as the paper-aligned prequential gap and is expected to be noisier because it includes sampling deviation and possible cancellation.

| scale | target_label | channel_label | value |
| --- | --- | --- | --- |
| run_level | $\Delta_T^{\mathrm{rep}}$ | Task minus cost | 0.559 |
| run_level | $\Delta_T^{\mathrm{rep}}$ | Task-aligned | 0.518 |
| run_level | $\Delta_T^{\mathrm{rep}}$ | Task minus subgroup | 0.518 |
| run_level | $\Delta_T^{\mathrm{rep}}$ | Task minus quantity | 0.421 |
| run_level | $\Delta_T^{\mathrm{rep}}$ | Coarse score | 0.402 |
| run_level | $\Delta_T^{\mathrm{rep}}$ | Null blind | 0.000 |
| run_level | $V_T$ | Task minus cost | 0.848 |
| run_level | $V_T$ | Task minus subgroup | 0.803 |
| run_level | $V_T$ | Task-aligned | 0.802 |
| run_level | $V_T$ | Task minus quantity | 0.650 |
| run_level | $V_T$ | Coarse score | 0.613 |
| run_level | $V_T$ | Null blind | 0.000 |
| transition_level | $v_t$ | Task-aligned | 0.635 |
| transition_level | $v_t$ | Task minus subgroup | 0.631 |
| transition_level | $v_t$ | Task minus cost | 0.623 |
| transition_level | $v_t$ | Task minus quantity | 0.425 |
| transition_level | $v_t$ | Coarse score | 0.272 |
| transition_level | $v_t$ | Null blind | 0.000 |

## Per-Transition Association

Bars in `figure_main_step_fr_association.*` report Spearman $\rho$ over positive-feedback transitions.

| channel_label | n_steps | spearman |
| --- | --- | --- |
| Task-aligned | 570 | 0.635 |
| Task minus subgroup | 570 | 0.631 |
| Task minus cost | 570 | 0.623 |
| Task minus quantity | 570 | 0.425 |
| Coarse score | 570 | 0.272 |
| Null blind | 570 | 0.000 |

## Manipulation Check Means

These means support that the feedback intervention moves the system; they are not the main observability evidence.

| $\mu$ | $V_T$ mean | $\Delta_T^{\mathrm{rep}}$ mean | Null blind raw FR mean | Coarse score raw FR mean | Task-aligned raw FR mean |
| --- | --- | --- | --- | --- | --- |
| 0.000 | 1514.856 | 2424.382 | 0.000 | 0.006 | 0.051 |
| 0.050 | 3146.001 | 3433.010 | 0.000 | 0.042 | 0.097 |
| 0.100 | 6078.380 | 4771.597 | 0.000 | 0.056 | 0.126 |
| 0.200 | 1.896e+04 | 1.732e+04 | 0.000 | 0.071 | 0.170 |

## Interpretation

The null blind channel is a fixed row-bucket partition and is independent of the feedback-targeted variables. It is a negative control for categorical Fisher motion induced by finite-sample/binning artifacts.
The coarse score channel is a prediction-score baseline, not a null channel. Task channels add task structure beyond the score baseline.
Observable FR is a contracted footprint, not an estimate of the full intrinsic $C_T/T$ budget.
The footprint is channel-dependent and target-dependent; the useful comparison is between the coarse score baseline and task-relevant channels, with Null blind as a negative control.
Unsigned pointwise loss motion is supporting evidence only and is reported in the appendix because it removes cancellation before averaging.
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
| `figure_main_channel_association_summary.*` | Primary channel association summary for $V_T$, $\Delta_T^{\mathrm{rep}}$, and $v_t$. |
| `figure_main_step_fr_association.*` | Primary transition-level channel association figure. |
| `figure_main_feedback_manipulation_check.*` | Secondary manipulation check. |
| `table_channel_association_summary.csv` | Values shown in `figure_main_channel_association_summary.*`. |
| `table_operational_raw_rate_diagnostics.csv` | Main deployable raw-rate and per-step diagnostic table. |
| `regional_feedback_step_diagnostics.csv` | Per-transition FR vs risk/motion diagnostics. |
| `regional_feedback_summary.csv` | Main metrics by mu. |
| `regional_feedback_rounds.csv` | Round-level raw data. Large/detail file. |
| `regional_feedback_results_by_seed.csv` | Condition/seed-level raw data. |
| `appendix_diagnostics/appendix_table_motion_diagnostics.csv` | Unsigned pointwise loss/error motion and held-out step-shift support diagnostics. |
| `appendix_diagnostics/` | Excess-rate, leave-one-$\mu$-out, within-$\mu$, pooled R2 heatmap, individual-run scatter, motion diagnostics, and condition-mean descriptive diagnostics. |
