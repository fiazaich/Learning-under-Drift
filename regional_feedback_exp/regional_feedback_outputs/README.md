# Regional Feedback Outputs

Main outputs use raw observed Fisher rates and operational diagnostics.
Appendix diagnostics are robustness and stress-test outputs.

## Main Claim

Raw observed Fisher motion is an uncalibrated but meaningful channel-level footprint of drift-induced risk motion.
It is not a calibrated estimator of run-level $V_T$ and it is not an estimate of the intrinsic $C_T/T$ budget.

## Main Outputs

- `RESULTS_DASHBOARD.md`: paper-facing result summary.
- `main_figure_values_summary.txt`: exact values used in the main figures.
- `figure_main_operational_raw_rate_r2_heatmap.*`: raw-rate regression $R^2$ by target and channel.
- `figure_main_step_fr_association.*`: positive-feedback transition Spearman association between per-step observed FR and $v_t$.
- `figure_main_feedback_manipulation_check.*`: secondary manipulation check against feedback strength $\mu$.
- `regional_feedback_raw_rate_regressions.csv`: raw-rate single-channel regressions with no $\mu$ covariate.
- `regional_feedback_step_diagnostics.csv`: transition-level association diagnostics.
- `table_operational_raw_rate_diagnostics.csv`: compact operational diagnostics table.

## Appendix Diagnostics

Appendix files live in `appendix_diagnostics/`.
Excess rates are controlled attribution diagnostics only, because deployment does not have matched $\mu=0$ counterfactuals.
Leave-one-$\mu$-out asks a stronger cross-regime extrapolation question than the observability claim.
Seed-level within-$\mu$ associations ask whether naive fixed channels rank run-level severity at fixed feedback strength.
Condition-mean and individual-run scatter plots are descriptive diagnostics, not calibrated monitoring results.

## Caveat

The experiment uses simple fixed binning rather than optimized channels.
Calibration or thresholding would require a deployment-specific operating-envelope calibration layer using replay, simulation, or held-out monitoring.