"""
Figure 1 — Regime-recovery sanity check.

Produces the styled PDF/SVG/PNG plus raw/summary/meta tables by running a set of
regimes (iid, exogenous-only, feedback-only) over a grid of horizons.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import scienceplots

from common_sim import RunStats, simulate_one_run
from out_utils import create_results_dir

plt.style.use(['science', 'ieee'])

FIGSIZE = (3.2, 2.3)
SAVEFIG_KW = {}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    attr: Optional[str]
    row_key: str
    label: str
    tag: str
    uses_gap_abs: bool = False
    subtract_trace: bool = False

    def value_from_stats(self, stats: RunStats, trace_sigma: float) -> float:
        if self.uses_gap_abs:
            return float(stats.gap_abs)
        if not self.attr:
            raise ValueError(f"Metric '{self.name}' missing attr.")
        val = float(getattr(stats, self.attr))
        if self.subtract_trace:
            return val - trace_sigma
        return val


METRIC_SPECS: Dict[str, MetricSpec] = {
    "err": MetricSpec(
        name="err",
        attr="err_T",
        row_key="err",
        label=r"$\mathbb{E}_{\text{seeds}}\!\left[\|\hat{\mu}_T-\theta_T\|_2\right]$",
        tag="err",
    ),
    "traj_gap": MetricSpec(
        name="traj_gap",
        attr=None,
        row_key="mean_traj_gap_abs",
        label=r"$|R_T - R|$",
        tag="traj_gap_abs",
        uses_gap_abs=True,
    ),
    "traj_error": MetricSpec(
        name="traj_error",
        attr="traj_pop_risk",
        row_key="mean_traj_pop_risk_excess",
        label=r"$\widehat R_T - \operatorname{tr}(\Sigma)$",
        tag="traj_error_excess",
        subtract_trace=True,
    ),
}
DEFAULT_METRIC = "err"


@dataclass
class RegimeConfig:
    d: int = 5
    Sigma_diag: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0)
    Ts: Sequence[int] = (500, 1000, 2000)
    seeds: Sequence[int] = tuple(range(20))
    k_policy: float = 0.25
    C_exo: float = 15.0
    gammas_feedback: Sequence[float] = (0.25, 0.5, 1.0)

    @property
    def Sigma(self) -> np.ndarray:
        return np.diag(self.Sigma_diag)


def run_simulations(cfg: RegimeConfig) -> Dict[str, Dict]:
    results: Dict[str, Dict[int, List[RunStats]]] = {
        "iid": {T: [] for T in cfg.Ts},
        "exo": {T: [] for T in cfg.Ts},
        "feedback": {g: {T: [] for T in cfg.Ts} for g in cfg.gammas_feedback},
    }

    for T in cfg.Ts:
        for s in cfg.seeds:
            r_iid = simulate_one_run(
                T=T, d=cfg.d, Sigma=cfg.Sigma,
                regime="iid", C_exo=0.0, gamma=0.0,
                policy_k=cfg.k_policy, seed=s,
            )
            results["iid"][T].append(r_iid)

            r_exo = simulate_one_run(
                T=T, d=cfg.d, Sigma=cfg.Sigma,
                regime="exo", C_exo=cfg.C_exo, gamma=0.0,
                policy_k=cfg.k_policy, seed=s,
            )
            results["exo"][T].append(r_exo)

        for g in cfg.gammas_feedback:
            for s in cfg.seeds:
                r_fb = simulate_one_run(
                    T=T, d=cfg.d, Sigma=cfg.Sigma,
                    regime="endogenous", C_exo=0.0, gamma=g,
                    policy_k=cfg.k_policy, seed=s,
                )
                results["feedback"][g][T].append(r_fb)

    return results


def compute_tables(results: Dict[str, Dict],
                   cfg: RegimeConfig) -> Tuple[List[Dict], List[Dict]]:
    raw_rows: List[Dict[str, object]] = []
    tr_sigma = float(np.sum(cfg.Sigma_diag))
    for T in cfg.Ts:
        for i, stats in enumerate(results["iid"][T]):
            raw_rows.append({
                "regime": "iid",
                "gamma": None,
                "T": T,
                "seed_idx": i,
                "err_T": float(stats.err_T),
                "traj_emp_risk": float(stats.traj_emp_risk),
                "traj_pop_risk": float(stats.traj_pop_risk),
                "traj_emp_risk_excess": float(stats.traj_emp_risk - tr_sigma),
                "traj_pop_risk_excess": float(stats.traj_pop_risk - tr_sigma),
                "traj_gap": float(stats.traj_gap),
                "traj_gap_abs": float(stats.gap_abs),
                "gap_abs": float(stats.gap_abs),
            })
        for i, stats in enumerate(results["exo"][T]):
            raw_rows.append({
                "regime": "exo",
                "gamma": None,
                "T": T,
                "seed_idx": i,
                "err_T": float(stats.err_T),
                "traj_emp_risk": float(stats.traj_emp_risk),
                "traj_pop_risk": float(stats.traj_pop_risk),
                "traj_emp_risk_excess": float(stats.traj_emp_risk - tr_sigma),
                "traj_pop_risk_excess": float(stats.traj_pop_risk - tr_sigma),
                "traj_gap": float(stats.traj_gap),
                "traj_gap_abs": float(stats.gap_abs),
                "gap_abs": float(stats.gap_abs),
            })
        for g in cfg.gammas_feedback:
            for i, stats in enumerate(results["feedback"][g][T]):
                raw_rows.append({
                    "regime": "feedback",
                    "gamma": float(g),
                    "T": T,
                    "seed_idx": i,
                    "err_T": float(stats.err_T),
                    "traj_emp_risk": float(stats.traj_emp_risk),
                    "traj_pop_risk": float(stats.traj_pop_risk),
                    "traj_emp_risk_excess": float(stats.traj_emp_risk - tr_sigma),
                    "traj_pop_risk_excess": float(stats.traj_pop_risk - tr_sigma),
                    "traj_gap": float(stats.traj_gap),
                    "traj_gap_abs": float(stats.gap_abs),
                    "gap_abs": float(stats.gap_abs),
                })

    def mean_se(vals: Sequence[float]) -> Tuple[float, float]:
        vals = np.asarray(vals, dtype=float)
        n = vals.size
        if n == 0:
            return float("nan"), float("nan")
        m = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        return m, se

    metrics = {
        "err_T": ("mean_err", "se_err"),
        "traj_gap": ("mean_traj_gap", "se_traj_gap"),
        "traj_gap_abs": ("mean_traj_gap_abs", "se_traj_gap_abs"),
        "traj_pop_risk": ("mean_traj_pop_risk", "se_traj_pop_risk"),
        "traj_pop_risk_excess": ("mean_traj_pop_risk_excess", "se_traj_pop_risk_excess"),
        "traj_emp_risk": ("mean_traj_emp_risk", "se_traj_emp_risk"),
        "traj_emp_risk_excess": ("mean_traj_emp_risk_excess", "se_traj_emp_risk_excess"),
    }

    def metric_attr(stats: RunStats, key: str) -> float:
        if key == "traj_gap_abs":
            return float(stats.gap_abs)
        if key == "traj_pop_risk_excess":
            return float(stats.traj_pop_risk - tr_sigma)
        if key == "traj_emp_risk_excess":
            return float(stats.traj_emp_risk - tr_sigma)
        return float(getattr(stats, key))

    def build_summary_row(regime: str, gamma_val: Optional[float], T: int, stats_list: Sequence[RunStats]) -> Dict[str, object]:
        row: Dict[str, object] = {"regime": regime, "gamma": gamma_val, "T": T, "n": len(stats_list)}
        for attr, (mean_key, se_key) in metrics.items():
            vals = [metric_attr(stats, attr) for stats in stats_list]
            m, se = mean_se(vals)
            row[mean_key] = m
            row[se_key] = se
        return row

    summary_rows: List[Dict[str, object]] = []
    for T in cfg.Ts:
        summary_rows.append(build_summary_row("iid", None, T, results["iid"][T]))
        summary_rows.append(build_summary_row("exo", None, T, results["exo"][T]))
        for g in cfg.gammas_feedback:
            summary_rows.append(build_summary_row("feedback", float(g), T, results["feedback"][g][T]))

    return raw_rows, summary_rows


def build_meta(cfg: RegimeConfig,
               results: Dict[str, Dict],
               metric_spec: MetricSpec) -> Dict[str, object]:
    iid_means_for_meta = []
    trace_sigma = float(np.sum(cfg.Sigma_diag))
    for T in cfg.Ts:
        vals = [metric_spec.value_from_stats(stats, trace_sigma) for stats in results["iid"][T]]
        iid_means_for_meta.append(float(np.mean(vals)) if vals else float("nan"))
    if all((m > 0) for m in iid_means_for_meta):
        slope_iid, intercept_iid = np.polyfit(np.log(cfg.Ts), np.log(iid_means_for_meta), 1)
    else:
        slope_iid, intercept_iid = float("nan"), float("nan")

    return {
        "figure": "Figure 1 — Regime recovery",
        "d": cfg.d,
        "Sigma_diag": list(map(float, cfg.Sigma_diag)),
        "Ts": list(map(int, cfg.Ts)),
        "seeds": list(map(int, cfg.seeds)),
        "k_policy": cfg.k_policy,
        "C_exo": cfg.C_exo,
        "gammas_feedback": list(map(float, cfg.gammas_feedback)),
        "trace_Sigma": trace_sigma,
        "metric": {
            "name": metric_spec.name,
            "label": metric_spec.label,
            "attr": metric_spec.attr,
            "tag": metric_spec.tag,
        },
        "metric_loglog_slope": float(slope_iid),
        "metric_loglog_intercept": float(intercept_iid),
        "notes": "Slope computed for the requested metric (log–log when positive).",
    }


def plot_regime_chart(results: Dict[str, Dict],
                      cfg: RegimeConfig,
                      metric_spec: MetricSpec,
                      out_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, layout='constrained')
    fig.set_size_inches(*FIGSIZE)
    trace_sigma = float(np.sum(cfg.Sigma_diag))

    def err_stats(vals: Sequence[float]) -> Tuple[float, float]:
        m = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals))
        return float(m), float(se)

    colors = {
        "iid": "#26413c",
        "exo": "#b54c4a",
        0.25: "#2f75b3",
        0.5: "#f08a2b",
        1.0: "#7d5ab3",
    }
    markers = {"iid": "o", "exo": "s", "fb": "^"}

    def plot_series(key, label, marker):
        means, ses = [], []
        for T in cfg.Ts:
            stats_list = results[key][T] if key in ("iid", "exo") else results["feedback"][key][T]
            vals = [metric_spec.value_from_stats(stats, trace_sigma) for stats in stats_list]
            m, e = err_stats(vals)
            means.append(m)
            ses.append(e)
        means = np.asarray(means, float)
        ses = np.asarray(ses, float)
        ax.errorbar(
            cfg.Ts, means, yerr=ses,
            marker=marker,
            color=colors[key if key in colors else key],
            markersize=3.0,
            linewidth=0.75,
            capsize=2.5,
            label=label,
        )
        return means, ses

    means_all, lowers, uppers = [], [], []
    m, s = plot_series("iid", r"i.i.d. ($\gamma=0$)", markers["iid"])
    means_all.append(m)
    lowers.append(np.maximum(m - s, 1e-6))
    uppers.append(m + s)
    m, s = plot_series("exo", rf"exo ($C_{{exo}}={int(cfg.C_exo)})$", markers["exo"])
    means_all.append(m)
    lowers.append(np.maximum(m - s, 1e-6))
    uppers.append(m + s)
    for g in cfg.gammas_feedback:
        m, s = plot_series(g, rf"$\gamma={g}$", markers["fb"])
        means_all.append(m)
        lowers.append(np.maximum(m - s, 1e-6))
        uppers.append(m + s)

    ax.set_xscale("log")
    use_log_y = all(np.all(series > 0) for series in means_all)
    ax.set_yscale("log" if use_log_y else "linear")
    ax.set_xlabel(r"Sample size $T$", fontsize=10)
    ax.set_ylabel(metric_spec.label, labelpad=2, fontsize=10)

    from matplotlib.ticker import (
        LogLocator,
        LogFormatterMathtext,
        NullFormatter,
        MaxNLocator,
        StrMethodFormatter,
        AutoMinorLocator,
    )
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.,)))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if use_log_y:
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.,)))
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(np.min(cfg.Ts), np.max(cfg.Ts))
    ax.tick_params(axis='both', which='major', labelsize=6.0)
    ax.tick_params(axis='both', which='minor', labelsize=5.0)

    mean_vals = np.concatenate(means_all) if means_all else np.array([1.0])
    lower_vals = np.concatenate(lowers) if lowers else mean_vals
    upper_vals = np.concatenate(uppers) if uppers else mean_vals
    if use_log_y:
        ymin = max(1e-6, np.nanmin(lower_vals) * 0.8)
        ymax = max(ymin * 10, np.nanmax(upper_vals) * 1.25)
    else:
        ymin = 0.0
        ymax = max(1.0, np.nanmax(upper_vals) * 1.15)
    ax.set_ylim(ymin, ymax)
    ax.grid(which="major", color="#e0e0e0", linewidth=0.4)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
        fontsize=8,
        markerscale=0.7,
        columnspacing=0.5,
    )

    fig.savefig(out_prefix.with_suffix(".pdf"), **SAVEFIG_KW)
    fig.savefig(out_prefix.with_suffix(".svg"), **SAVEFIG_KW)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=600, **SAVEFIG_KW)


def save_tables(raw_rows: Sequence[Dict[str, object]],
                summary_rows: Sequence[Dict[str, object]],
                meta: Dict[str, object],
                results_dir: Path) -> None:
    import csv
    import json

    raw_path = results_dir / "fig1_regime_recovery_raw.csv"
    summary_path = results_dir / "fig1_regime_recovery_summary.csv"
    meta_path = results_dir / "fig1_regime_recovery_meta.json"

    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "regime", "gamma", "T", "seed_idx",
                "err_T",
                "traj_emp_risk", "traj_emp_risk_excess",
                "traj_pop_risk", "traj_pop_risk_excess",
                "traj_gap", "traj_gap_abs", "gap_abs",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "regime", "gamma", "T", "n",
                "mean_err", "se_err",
                "mean_traj_gap", "se_traj_gap",
                "mean_traj_gap_abs", "se_traj_gap_abs",
                "mean_traj_pop_risk", "se_traj_pop_risk",
                "mean_traj_pop_risk_excess", "se_traj_pop_risk_excess",
                "mean_traj_emp_risk", "se_traj_emp_risk",
                "mean_traj_emp_risk_excess", "se_traj_emp_risk_excess",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def main():
    args = parse_args()
    metric_spec = METRIC_SPECS[args.metric]
    cfg = RegimeConfig()
    results_dir = create_results_dir(f"gaussian_regime_recovery_{metric_spec.name}")
    results = run_simulations(cfg)
    raw_rows, summary_rows = compute_tables(results, cfg)
    meta = build_meta(cfg, results, metric_spec)
    plot_regime_chart(results, cfg, metric_spec, results_dir / "fig1_regime_recovery")
    save_tables(raw_rows, summary_rows, meta, results_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    choice_list = ", ".join(
        f"{name}→{spec.label}" for name, spec in METRIC_SPECS.items()
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_SPECS.keys()),
        default=DEFAULT_METRIC,
        help=f"Metric to plot (options: {choice_list}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
