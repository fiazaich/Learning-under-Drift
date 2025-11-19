"""
Figure 2 — Additivity & budget scaling (Gaussian toy model).

This script mirrors the NN simulation pipeline (smoke_test2.py) but uses an
analytically tractable Gaussian environment. It now logs trajectory-averaged
empirical/population risks so the reproducibility gap matches the NN setting.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from common_sim import simulate_one_run, fit_linear_plane
from out_utils import create_results_dir

plt.style.use(['science', 'ieee'])

FIGSIZE = (3.2, 2.3)
SAVEFIG_KW = {}

@dataclass(frozen=True)
class MetricSpec:
    name: str
    row_key: str
    label: str
    tag: str


METRIC_SPECS: Dict[str, MetricSpec] = {
    "err": MetricSpec(
        name="err",
        row_key="err",
        label=r"$\mathbb{E}_{\text{seeds}}\!\left[\|\hat{\mu}_T-\theta_T\|_2\right]$",
        tag="err",
    ),
    "traj_gap": MetricSpec(
        name="traj_gap",
        row_key="traj_gap_abs",
        label=r"$|R_T - R|$",
        tag="traj_gap_abs",
    ),
    "traj_error": MetricSpec(
        name="traj_error",
        row_key="traj_pop_risk_excess",
        label=r"$\widehat R_T - \operatorname{tr}(\Sigma)$",
        tag="traj_error_excess",
    ),
}
DEFAULT_METRIC = "err"


@dataclass
class ExperimentConfig:
    d: int
    Sigma: np.ndarray
    T: int
    seeds: Sequence[int]
    k_policy: float
    C_exo_grid: Sequence[float]
    gamma_grid: Sequence[float]


@dataclass
class RegressionSummary:
    plane_coef: np.ndarray
    plane_r2: float
    alpha_opt: float
    collapse_coef: Tuple[float, float]
    collapse_r2: float
    y: np.ndarray
    C_over_T: np.ndarray
    regime: np.ndarray
    sum_dt_over_T: np.ndarray
    sum_kappa_over_T: np.ndarray
    inv_sqrt_T: np.ndarray
    metric_spec: MetricSpec


@dataclass
class ExportPaths:
    plane_fit: Path
    raw_csv: Path
    summary_csv: Path
    meta_json: Path
    fig_prefix: Path


def regime_name(C_exo: float, gamma: float) -> str:
    if C_exo == 0.0 and gamma == 0.0:
        return "iid"
    if C_exo > 0.0 and gamma == 0.0:
        return "exogenous-only"
    if C_exo == 0.0 and gamma > 0.0:
        return "endogenous-only"
    return "mixed"


def build_export_paths(results_dir: Path) -> ExportPaths:
    return ExportPaths(
        plane_fit=results_dir / "fig2_plane_fit.txt",
        raw_csv=results_dir / "fig2_additivity_raw.csv",
        summary_csv=results_dir / "fig2_additivity_summary.csv",
        meta_json=results_dir / "fig2_additivity_meta.json",
        fig_prefix=results_dir / "fig2_budget_scatter",
    )


def run_grid(cfg: ExperimentConfig) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    rows: List[Dict[str, float]] = []
    raw_rows: List[Dict[str, float]] = []
    tr_sigma = float(np.trace(cfg.Sigma))

    for C_exo in cfg.C_exo_grid:
        for gamma in cfg.gamma_grid:
            err_all, sum_dt_all, sum_kappa_all = [], [], []
            traj_emp_all, traj_pop_all = [], []
            traj_emp_excess_all, traj_pop_excess_all = [], []
            traj_gap_abs_all = []
            pop_init_all, pop_final_all = [], []
            reg = regime_name(C_exo, gamma)
            for seed in cfg.seeds:
                stats = simulate_setting(cfg, C_exo, gamma, seed)
                err_all.append(stats.err_T)
                sum_dt_all.append(stats.sum_dt)
                sum_kappa_all.append(stats.sum_kappa)
                traj_emp_all.append(stats.traj_emp_risk)
                traj_pop_all.append(stats.traj_pop_risk)
                traj_emp_excess_all.append(stats.traj_emp_risk - tr_sigma)
                traj_pop_excess_all.append(stats.traj_pop_risk - tr_sigma)
                traj_gap_abs_all.append(stats.gap_abs)
                pop_init_all.append(stats.pop_mse_initial)
                pop_final_all.append(stats.pop_mse_final)
                raw_rows.append({
                    "C_exo": float(C_exo),
                    "gamma": float(gamma),
                    "T": int(cfg.T),
                    "seed": int(seed),
                    "regime": reg,
                    "err_T": float(stats.err_T),
                    "sum_dt": float(stats.sum_dt),
                    "sum_kappa": float(stats.sum_kappa),
                    "traj_emp_risk": float(stats.traj_emp_risk),
                    "traj_pop_risk": float(stats.traj_pop_risk),
                    "traj_emp_risk_excess": float(stats.traj_emp_risk - tr_sigma),
                    "traj_pop_risk_excess": float(stats.traj_pop_risk - tr_sigma),
                    "traj_gap_traj": float(stats.traj_gap),
                    "traj_gap_abs": float(stats.gap_abs),
                    "pop_mse_initial": float(stats.pop_mse_initial),
                    "pop_mse_final": float(stats.pop_mse_final),
                })
            rows.append({
                "C_exo": float(C_exo),
                "gamma": float(gamma),
                "T": int(cfg.T),
                "regime": reg,
                "err": float(np.mean(err_all)),
                "sum_dt": float(np.mean(sum_dt_all)),
                "sum_kappa": float(np.mean(sum_kappa_all)),
                "traj_emp_risk": float(np.mean(traj_emp_all)),
                "traj_pop_risk": float(np.mean(traj_pop_all)),
                "traj_emp_risk_excess": float(np.mean(traj_emp_excess_all)),
                "traj_pop_risk_excess": float(np.mean(traj_pop_excess_all)),
                "traj_pop_initial": float(np.mean(pop_init_all)),
                "traj_pop_final": float(np.mean(pop_final_all)),
                "traj_gap_abs": float(np.mean(traj_gap_abs_all)),
                "gen_gap": float(np.mean(traj_gap_abs_all)),
            })

    return rows, raw_rows


def simulate_setting(cfg: ExperimentConfig, C_exo: float, gamma: float, seed: int):
    reg = regime_name(C_exo, gamma)
    regime_map = {
        "iid": ("iid", 0.0, 0.0),
        "exogenous-only": ("exo", C_exo, 0.0),
        "endogenous-only": ("endogenous", 0.0, gamma),
        "mixed": ("mixed", C_exo, gamma),
    }
    regime_name_str, C_use, gamma_use = regime_map[reg]
    stats = simulate_one_run(
        T=cfg.T,
        d=cfg.d,
        Sigma=cfg.Sigma,
        regime=regime_name_str,
        C_exo=C_use,
        gamma=gamma_use,
        policy_k=cfg.k_policy,
        seed=seed,
    )
    return stats


def compute_regression_summary(rows: Sequence[Dict[str, float]],
                               metric_key: str,
                               metric_spec: MetricSpec) -> RegressionSummary:
    y = np.array([r[metric_key] for r in rows], dtype=float)
    T_arr = np.array([r["T"] for r in rows], dtype=float)
    inv_sqrt_T = T_arr ** -0.5
    sum_dt_over_T = np.array([r["sum_dt"] / r["T"] for r in rows], dtype=float)
    sum_kappa_over_T = np.array([r["sum_kappa"] / r["T"] for r in rows], dtype=float)

    X = np.c_[np.ones_like(y), inv_sqrt_T, sum_dt_over_T, sum_kappa_over_T]
    coef, r2 = fit_linear_plane(y, X)
    b1, b2 = float(coef[2]), float(coef[3])
    alpha_opt = b2 / b1 if abs(b1) > 1e-12 else 1.0
    C_over_T = sum_dt_over_T + alpha_opt * sum_kappa_over_T
    X1 = np.c_[np.ones_like(C_over_T), C_over_T]
    coef_coll, r2_coll = fit_linear_plane(y, X1)
    regimes = np.array([r["regime"] for r in rows], dtype=str)

    return RegressionSummary(
        plane_coef=np.asarray(coef, dtype=float),
        plane_r2=float(r2),
        alpha_opt=float(alpha_opt),
        collapse_coef=(float(coef_coll[0]), float(coef_coll[1])),
        collapse_r2=float(r2_coll),
        y=y,
        C_over_T=C_over_T,
        regime=regimes,
        sum_dt_over_T=sum_dt_over_T,
        sum_kappa_over_T=sum_kappa_over_T,
        inv_sqrt_T=inv_sqrt_T,
        metric_spec=metric_spec,
    )


def build_summary_rows(raw_rows: Sequence[Dict[str, float]], alpha_opt: float) -> List[Dict[str, float]]:
    from collections import defaultdict

    def mean_se(vals: Iterable[float]) -> Tuple[float, float, int]:
        vals = list(vals)
        n = len(vals)
        m = float(np.mean(vals)) if n else float("nan")
        se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        return m, se, n

    group = defaultdict(lambda: {
        "err_T": [], "sum_dt": [], "sum_kappa": [],
        "traj_emp": [], "traj_pop": [],
        "traj_emp_excess": [], "traj_pop_excess": [],
        "traj_gap_abs": [], "gen_gap": [],
        "pop_init": [], "pop_final": [], "T": None, "regime": None,
    })

    for r in raw_rows:
        key = (r["C_exo"], r["gamma"])
        g = group[key]
        g["err_T"].append(r["err_T"])
        g["sum_dt"].append(r["sum_dt"])
        g["sum_kappa"].append(r["sum_kappa"])
        g["traj_emp"].append(r["traj_emp_risk"])
        g["traj_pop"].append(r["traj_pop_risk"])
        g["traj_emp_excess"].append(r.get("traj_emp_risk_excess", float("nan")))
        g["traj_pop_excess"].append(r.get("traj_pop_risk_excess", float("nan")))
        g["traj_gap_abs"].append(r.get("traj_gap_abs", r.get("traj_gap_traj", r.get("gen_gap_traj", float("nan")))))
        g["gen_gap"].append(r.get("traj_gap_traj", r.get("gen_gap_traj", float("nan"))))
        g["pop_init"].append(r["pop_mse_initial"])
        g["pop_final"].append(r["pop_mse_final"])
        g["T"] = r["T"]
        g["regime"] = r["regime"]

    summary_rows: List[Dict[str, float]] = []
    for (C_exo, gamma), g in sorted(group.items()):
        m_err, se_err, n = mean_se(g["err_T"])
        m_dt, se_dt, _ = mean_se(g["sum_dt"])
        m_kap, se_kap, _ = mean_se(g["sum_kappa"])
        m_emp, se_emp, _ = mean_se(g["traj_emp"])
        m_pop, se_pop, _ = mean_se(g["traj_pop"])
        m_gap, se_gap, _ = mean_se(g["gen_gap"])
        m_emp_excess, se_emp_excess, _ = mean_se(g["traj_emp_excess"])
        m_pop_excess, se_pop_excess, _ = mean_se(g["traj_pop_excess"])
        m_gap_abs, se_gap_abs, _ = mean_se(g["traj_gap_abs"])
        m_init, se_init, _ = mean_se(g["pop_init"])
        m_final, se_final, _ = mean_se(g["pop_final"])
        summary_rows.append({
            "C_exo": float(C_exo),
            "gamma": float(gamma),
            "T": int(g["T"]),
            "regime": g["regime"],
            "mean_err_T": m_err,
            "se_err_T": se_err,
            "n": int(n),
            "mean_sum_dt": m_dt,
            "se_sum_dt": se_dt,
            "mean_sum_kappa": m_kap,
            "se_sum_kappa": se_kap,
            "mean_gen_gap_traj": m_gap,
            "se_gen_gap_traj": se_gap,
            "mean_traj_gap_abs": m_gap_abs,
            "se_traj_gap_abs": se_gap_abs,
            "mean_traj_emp_risk": m_emp,
            "se_traj_emp_risk": se_emp,
            "mean_traj_pop_risk": m_pop,
            "se_traj_pop_risk": se_pop,
            "mean_traj_emp_risk_excess": m_emp_excess,
            "se_traj_emp_risk_excess": se_emp_excess,
            "mean_traj_pop_risk_excess": m_pop_excess,
            "se_traj_pop_risk_excess": se_pop_excess,
            "mean_pop_mse_initial": m_init,
            "se_pop_mse_initial": se_init,
            "mean_pop_mse_final": m_final,
            "se_pop_mse_final": se_final,
            "mean_C_over_T(alpha_opt)": float((m_dt + alpha_opt * m_kap) / g["T"])
            if g["T"] else float("nan"),
        })

    return summary_rows


def write_plane_fit(path: Path, stats: RegressionSummary) -> None:
    metric_tag = stats.metric_spec.tag
    b0, b_s, b1, b2 = map(float, stats.plane_coef)
    a0, a1 = stats.collapse_coef
    with open(path, "w") as f:
        f.write(f"Gaussian additivity plane: {metric_tag} ~ b0 + b_s*T^{-1/2} + b1*(sum_dt/T) + b2*(sum_kappa/T)\n")
        f.write(f"b0   = {b0:.6f}\n")
        f.write(f"b_s  = {b_s:.6f}\n")
        f.write(f"b1   = {b1:.6f}\n")
        f.write(f"b2   = {b2:.6f}\n")
        f.write(f"R^2  = {stats.plane_r2:.4f}\n")
        f.write(f"\nCollapse: {metric_tag} ~ a0 + a1 * (C_T/T)\n")
        f.write(f"alpha_opt = {stats.alpha_opt:.6f}\n")
        f.write(f"a0 = {a0:.6f}\n")
        f.write(f"a1 = {a1:.6f}\n")
        f.write(f"R^2  = {stats.collapse_r2:.4f}\n")


def save_tables(paths: ExportPaths,
                raw_rows: Sequence[Dict[str, float]],
                summary_rows: Sequence[Dict[str, float]],
                meta: Dict[str, object]) -> None:
    import csv

    with open(paths.raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "C_exo", "gamma", "T", "seed", "regime",
                "err_T", "sum_dt", "sum_kappa",
                "traj_gap_traj", "traj_gap_abs",
                "traj_emp_risk", "traj_pop_risk",
                "traj_emp_risk_excess", "traj_pop_risk_excess",
                "pop_mse_initial", "pop_mse_final",
            ],
        )
        writer.writeheader()
        for row in raw_rows:
            writer.writerow(row)

    with open(paths.summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "C_exo", "gamma", "T", "regime",
                "mean_err_T", "se_err_T", "n",
                "mean_sum_dt", "se_sum_dt",
                "mean_sum_kappa", "se_sum_kappa",
                "mean_gen_gap_traj", "se_gen_gap_traj",
                "mean_traj_gap_abs", "se_traj_gap_abs",
                "mean_traj_emp_risk", "se_traj_emp_risk",
                "mean_traj_emp_risk_excess", "se_traj_emp_risk_excess",
                "mean_traj_pop_risk", "se_traj_pop_risk",
                "mean_traj_pop_risk_excess", "se_traj_pop_risk_excess",
                "mean_pop_mse_initial", "se_pop_mse_initial",
                "mean_pop_mse_final", "se_pop_mse_final",
                "mean_C_over_T(alpha_opt)",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with open(paths.meta_json, "w") as f:
        json.dump(meta, f, indent=2)


def build_meta(cfg: ExperimentConfig,
               stats: RegressionSummary,
               summary_rows: Sequence[Dict[str, float]],
               raw_rows: Sequence[Dict[str, float]]) -> Dict[str, object]:
    metric_tag = stats.metric_spec.tag
    trace_sigma = float(np.trace(cfg.Sigma))
    return {
        "figure": "Additivity & budget scaling (Gaussian)",
        "d": cfg.d,
        "Sigma_diag": list(np.diag(cfg.Sigma).astype(float)),
        "trace_Sigma": trace_sigma,
        "T": int(cfg.T),
        "seeds": list(map(int, cfg.seeds)),
        "k_policy": float(cfg.k_policy),
        "C_exo_grid": list(map(float, cfg.C_exo_grid)),
        "gamma_grid": list(map(float, cfg.gamma_grid)),
        "metric": {
            "name": stats.metric_spec.name,
            "label": stats.metric_spec.label,
            "row_key": stats.metric_spec.row_key,
            "tag": stats.metric_spec.tag,
        },
        "plane_fit": {
            "spec": f"{metric_tag} ~ b0 + b_s*T^{-1/2} + b1*(sum_dt/T) + b2*(sum_kappa/T)",
            "b0": float(stats.plane_coef[0]),
            "b_s": float(stats.plane_coef[1]),
            "b1": float(stats.plane_coef[2]),
            "b2": float(stats.plane_coef[3]),
            "R2": float(stats.plane_r2),
        },
        "alpha_opt": float(stats.alpha_opt),
        "collapse_fit": {
            "spec": f"{metric_tag} ~ a0 + a1 * C_T/T",
            "a0": float(stats.collapse_coef[0]),
            "a1": float(stats.collapse_coef[1]),
            "R2": float(stats.collapse_r2),
        },
        "traj_pop_mse_initial_mean": float(np.mean([r["mean_pop_mse_initial"] for r in summary_rows])) if summary_rows else float("nan"),
        "traj_pop_mse_final_mean": float(np.mean([r["mean_pop_mse_final"] for r in summary_rows])) if summary_rows else float("nan"),
        "traj_pop_risk_excess_mean": float(np.mean([r["mean_traj_pop_risk_excess"] for r in summary_rows])) if summary_rows else float("nan"),
        "traj_gap_abs_mean": float(np.mean([r["mean_traj_gap_abs"] for r in summary_rows])) if summary_rows else float("nan"),
        "summary_preview": summary_rows[:min(8, len(summary_rows))],
        "raw_head": raw_rows[:min(10, len(raw_rows))],
    }


def plot_budget_scatter(stats: RegressionSummary, fig_prefix: Path) -> None:
    y = stats.y
    C_over_T = stats.C_over_T
    regime = stats.regime
    a0, a1 = stats.collapse_coef
    r2_line = stats.collapse_r2
    alpha_opt = stats.alpha_opt

    fig, ax = plt.subplots(figsize=FIGSIZE, layout='constrained')
    fig.set_size_inches(*FIGSIZE)

    palette = {
        "iid": "#2f4f4f",
        "exogenous-only": "#b35b45",
        "endogenous-only": "#6b8c42",
        "mixed": "#6c5b9c",
    }
    markers = {"iid": "o", "exogenous-only": "s", "endogenous-only": "d", "mixed": "^"}
    scatter_kw = dict(s=16, linewidths=0.5, alpha=0.85, zorder=3)

    xmax = float(C_over_T.max()) if C_over_T.size else 1.0
    xmin = -0.02 * (xmax if xmax > 0 else 1.0)
    ax.set_xlim(xmin, xmax * 1.05)

    for reg, color in palette.items():
        idx = (regime == reg)
        if not np.any(idx):
            continue
        ax.scatter(
            C_over_T[idx],
            y[idx],
            marker=markers[reg],
            facecolor=color,
            edgecolor=color,
            label=reg.replace("-", " "),
            **scatter_kw,
        )

    xgrid = np.linspace(0.0, xmax * 1.05, 200)
    ax.plot(
        xgrid,
        a0 + a1 * xgrid,
        lw=0.9,
        ls='-',
        color='0.25',
        label='linear fit',
        zorder=2,
    )

    ax.set_xlabel(r'Budget Ratio $C_T/T$', fontsize=10)
    ax.set_ylabel(stats.metric_spec.label, labelpad=2, fontsize=10)
    ax.set_ylim(0.0, (y.max() * 1.15) if y.size else 1.0)
    ax.tick_params(axis='both', which='major', labelsize=6.0)
    ax.tick_params(axis='both', which='minor', labelsize=5.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis='y', which='major', alpha=0.45)

    residuals = y - (a0 + a1 * C_over_T)
    inset = inset_axes(
        ax,
        width="28%",
        height="28%",
        loc="lower right",
        bbox_to_anchor=(-0.025, 0.06, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.2,
    )
    inset.axhline(0, color='0.5', lw=0.6, ls='--')
    inset.scatter(C_over_T, residuals, s=10, color="#4f65a3", alpha=0.8, linewidths=0)
    inset.set_xticks([0.0, round(float(xmax) / 2, 2), round(float(xmax), 2)])
    inset.set_yticks([])
    inset.tick_params(labelsize='x-small', pad=1.2, length=2.0)
    inset.set_title('Residuals', fontsize='small', pad=1.5)
    inset.spines['right'].set_visible(False)
    inset.spines['top'].set_visible(False)

    legend_handles = [
        Line2D([], [], marker='o', markersize=4.5, markerfacecolor=palette['iid'], markeredgecolor='none', linestyle='none', label='iid'),
        Line2D([], [], marker='s', markersize=4.5, markerfacecolor=palette['exogenous-only'], markeredgecolor='none', linestyle='none', label='exogenous only'),
        Line2D([], [], marker='d', markersize=4.5, markerfacecolor=palette['endogenous-only'], markeredgecolor='none', linestyle='none', label='endogenous only'),
        Line2D([], [], marker='^', markersize=4.5, markerfacecolor=palette['mixed'], markeredgecolor='none', linestyle='none', label='mixed'),
    ]
    ax.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
        frameon=False,
        fontsize=8,
        columnspacing=0.5,
    )

    ax.text(
        0.02,
        0.95,
        rf"$\alpha^\star = {alpha_opt:.2f}$",
        fontsize=8,
        color='0.25',
        transform=ax.transAxes,
        va='top',
        ha='left',
    )
    ax.text(
        0.72,
        0.6,
        fr"$R^2 = {r2_line:.2f}$",
        fontsize=8,
        color='0.25',
        transform=ax.transAxes,
        va='top',
        ha='left',
    )

    fig.savefig(fig_prefix.with_suffix(".pdf"), **SAVEFIG_KW)
    fig.savefig(fig_prefix.with_suffix(".svg"), **SAVEFIG_KW)
    fig.savefig(fig_prefix.with_suffix(".png"), dpi=600, **SAVEFIG_KW)
    plt.close(fig)


def main():
    args = parse_args()
    metric_spec = METRIC_SPECS[args.metric]
    cfg = ExperimentConfig(
        d=5,
        Sigma=np.diag([1.0]*5),
        T=2000,
        seeds=tuple(range(12)),
        k_policy=0.25,
        C_exo_grid=(0.0, 2.0, 4.0, 8.0, 16.0, 32.0),
        gamma_grid=(0.0, 0.01, 0.02, 0.04, 0.08),
    )
    results_dir = create_results_dir(f"gaussian_additivity_{metric_spec.name}")
    rows, raw_rows = run_grid(cfg)
    stats = compute_regression_summary(rows, metric_spec.row_key, metric_spec)
    summary_rows = build_summary_rows(raw_rows, stats.alpha_opt)
    paths = build_export_paths(results_dir)
    write_plane_fit(paths.plane_fit, stats)
    meta = build_meta(cfg, stats, summary_rows, raw_rows)
    save_tables(paths, raw_rows, summary_rows, meta)
    plot_budget_scatter(stats, paths.fig_prefix)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    choice_list = ", ".join(
        f"{name}→{spec.label}" for name, spec in METRIC_SPECS.items()
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_SPECS.keys()),
        default=DEFAULT_METRIC,
        help=f"Metric to target (options: {choice_list}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
