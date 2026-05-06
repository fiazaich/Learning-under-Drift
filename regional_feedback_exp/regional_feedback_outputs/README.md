# Regional Feedback Outputs

Main outputs use raw observed Fisher rates and operational diagnostics.
Appendix diagnostics are robustness and stress-test outputs.

## Main Claim

Raw observed Fisher motion is an uncalibrated but meaningful channel-level footprint of drift-induced risk motion.
It is not a calibrated estimator of run-level $V_T$ and it is not an estimate of the intrinsic $C_T/T$ budget.
The null blind channel is a fixed row/hash bucket partition and is used as a negative control.

## Main Outputs

- `RESULTS_DASHBOARD.md`: paper-facing result summary.
- `main_figure_values_summary.txt`: exact values used in the main figures.
- `figure_main_channel_association_summary.*`: run-level raw-rate $R^2$ for $V_T$ and $\Delta_T^{\mathrm{rep}}$, plus transition-level Spearman $\rho$ for $v_t$.
- `figure_main_step_fr_association.*`: positive-feedback transition Spearman association between per-step observed FR and $v_t$, including null.
- `figure_main_feedback_manipulation_check.*`: secondary manipulation check against feedback strength $\mu$.
- `table_channel_association_summary.csv`: values shown in the main channel association figure.
- `regional_feedback_raw_rate_regressions.csv`: raw-rate single-channel regressions with no $\mu$ covariate.
- `regional_feedback_step_diagnostics.csv`: transition-level association diagnostics.
- `table_operational_raw_rate_diagnostics.csv`: compact operational diagnostics table.

## Appendix Diagnostics

Appendix files live in `appendix_diagnostics/`.
CSV files use the literal channel id `null`; with pandas, read these files with `keep_default_na=False` if you need to preserve that string instead of parsing it as missing.
The pooled raw-rate $R^2$ heatmap is appendix-only because it primarily summarizes regime-level co-movement and should not be interpreted as calibrated run-level prediction.
`appendix_table_gain_over_coarse_score.csv` reports channel minus Coarse score diagnostics for task-relevant channels.
`appendix_table_motion_diagnostics.csv` reports unsigned pointwise loss/error motion and held-out step-shift diagnostics.
Unsigned pointwise loss motion removes cancellation before averaging, so it is expected to align more strongly with unsigned Fisher motion than $\Delta_T^{\mathrm{rep}}$.
Excess rates are controlled attribution diagnostics only, because deployment does not have matched $\mu=0$ counterfactuals.
Leave-one-$\mu$-out asks a stronger cross-regime extrapolation question than the observability claim.
Seed-level within-$\mu$ associations ask whether naive fixed channels rank run-level severity at fixed feedback strength.
Condition-mean and individual-run scatter plots are descriptive diagnostics, not calibrated monitoring results.

## Caveat

The experiment uses simple fixed binning rather than optimized channels.
Calibration or thresholding would require a deployment-specific operating-envelope calibration layer using replay, simulation, or held-out monitoring.