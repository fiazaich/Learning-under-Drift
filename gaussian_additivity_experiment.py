#!/usr/bin/env python3
"""Gaussian additivity drift experiment (paper-aligned).

Drift functional (paper-aligned):
    V_T = (1/T) sum_{t=1}^T | R(theta_{t+1}, mu_{t-1}) - R(theta_t, mu_{t-1}) |
with closed-form Gaussian risk:
    R(theta, mu) = tr(Sigma) + ||theta - mu||^2

Budget plumbing (Fisher-consistent):
- dt_t    = || v_exo_t ||_{Sigma^{-1}}
- kappa_t = || gamma*u_t ||_{Sigma^{-1}}
and exogenous increments are generated with Fisher-normalized directions so that
sum_t dt_t ≈ C_exo (when exo drift enabled).

Outputs:
- additivity_raw.csv (per seed)
- additivity_summary.csv (mean ± SE per grid point)
- plane_fit.txt
- budget_scatter.(pdf|svg|png)

Default metric for regression/plot: V_T
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ------------------------------ plotting deps ------------------------------
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  

try:
    import scienceplots  
    plt.style.use(["science", "ieee"])
except Exception:
    plt.style.use(["default"])

mpl.rcParams.update(
    {
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
    }
)

from matplotlib.lines import Line2D  
from matplotlib.ticker import AutoMinorLocator, MaxNLocator 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  


# ----------------------- Data structures & logging -----------------------

@dataclass
class RunStats:
    # final estimation error vs last *sampled* theta (theta_T)
    err_T: float

    # drift functional V_T (paper-aligned: uses mu_{t-1} inside risks)
    V_T: float

    # paper-aligned gaps
    traj_samp_gap: float     # | mean empirical - mean population at theta_t with mu_{t-1} |
    traj_rep_gap: float      # | mean empirical - mean population at theta_{t+1} with mu_{t-1} |  (Delta_rep)

    # budgets (Fisher-consistent)
    sum_dt: float            # sum_t || v_exo_t ||_{Sigma^{-1}}
    sum_kappa: float         # sum_t || gamma*u_t ||_{Sigma^{-1}}
    fisher_path_len: float   # sum_t || theta_{t+1}-theta_t ||_{Sigma^{-1}}

    # extra telemetry
    traj_emp_risk: float
    traj_pop_risk_pre: float     # mean_t R(theta_t, mu_{t-1})
    traj_pop_risk_post: float    # mean_t R(theta_{t+1}, mu_{t-1})
    pop_mse_initial: float
    pop_mse_final: float


# ------------------------------ Utilities --------------------------------

EPS = 1e-12
FIGSIZE = (3.2, 2.3)
SAVEFIG_KW = {}


def create_results_dir(prefix: str) -> Path:
    base = Path("results")
    base.mkdir(parents=True, exist_ok=True)
    tag = int(np.random.default_rng().integers(0, 10**9))
    out = base / f"{prefix}_{tag}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def cholesky_sampler(Sigma: np.ndarray, rng: np.random.Generator):
    """Return function to sample N(theta, Sigma) using one Cholesky."""
    L = np.linalg.cholesky(Sigma)
    d = Sigma.shape[0]

    def sample(theta: np.ndarray, size: int | None = None) -> np.ndarray:
        if size is None:
            z = rng.standard_normal(d)
            return theta + L @ z
        z = rng.standard_normal((size, d))
        return theta + z @ L.T

    return sample


def fisher_norm(v: np.ndarray, Sigma_inv: np.ndarray) -> float:
    return float(np.sqrt(v @ Sigma_inv @ v))


def fisher_step_length(delta: np.ndarray, Sigma_inv: np.ndarray) -> float:
    return fisher_norm(delta, Sigma_inv)


def fit_linear_plane(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """OLS: y ~ X @ coef. Returns (coef, R^2)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return coef, r2


def risk_gaussian_mean(theta: np.ndarray, mu: np.ndarray, trSigma: float) -> float:
    """Population MSE: E||X - mu||^2 for X ~ N(theta, Sigma)."""
    return float(trSigma + np.sum((theta - mu) ** 2))


# --------------------------- Core simulation -----------------------------

def simulate_one_run(
    T: int,
    d: int,
    Sigma: np.ndarray,
    regime: str,         # "iid", "exo", "endogenous", or "mixed"
    C_exo: float,        # total *Fisher* exogenous path budget (used if exo enabled)
    gamma: float,        # action-to-environment gain
    policy_k: float,     # u_t = -k * mu_{t-1}  (paper-aligned measurability)
    seed: int,
) -> RunStats:
    """
    Gaussian mean estimation with paper-aligned timing:
      - Deployed predictor at time t: mu_{t-1}
      - Empirical loss at time t: ||mu_{t-1} - x_t||^2
      - Endogenous action u_t is based on mu_{t-1}
      - Drift term V_T uses mu_{t-1} inside both risks:
            |R(theta_{t+1}, mu_{t-1}) - R(theta_t, mu_{t-1})|
    """
    rng = np.random.default_rng(seed)
    sampler = cholesky_sampler(Sigma, rng)
    Sigma_inv = np.linalg.inv(Sigma)
    trSigma = float(np.trace(Sigma))

    theta = np.zeros(d)  # theta_t
    mu = np.zeros(d)     # mu_{t-1} at loop start

    # Precompute exogenous vectors with Fisher-normalized directions so sum_t ||v_exo_t||_F ≈ C_exo.
    exo_vecs = np.zeros((T, d))
    if regime in ("exo", "mixed") and C_exo > 0.0:
        step_F = C_exo / T
        raw = rng.normal(size=(T, d))
        for t in range(T):
            v = raw[t]
            nF = fisher_norm(v, Sigma_inv)
            if nF < EPS:
                v = rng.normal(size=d)
                nF = fisher_norm(v, Sigma_inv) + EPS
            exo_vecs[t] = v * (step_F / (nF + EPS))

    # Storage for theta_t (to define theta_T cleanly)
    theta_pre = np.zeros((T, d))   # theta_t

    # Budgets
    sum_dt = 0.0
    sum_kappa = 0.0
    fisher_path_len = 0.0

    # Risks / losses
    emp_losses: List[float] = []
    pop_pre_losses: List[float] = []    # R(theta_t, mu_{t-1})
    pop_post_losses: List[float] = []   # R(theta_{t+1}, mu_{t-1})  (paper one-step-ahead target)
    V_terms: List[float] = []

    for t in range(1, T + 1):
        mu_prev = mu.copy()     # deployed f_t
        theta_pre[t - 1] = theta

        # sample x_t ~ N(theta_t, Sigma)
        x_t = sampler(theta)

        # empirical loss at time t using deployed predictor mu_{t-1}
        emp_losses.append(float(np.sum((mu_prev - x_t) ** 2)))

        # population risk at theta_t with deployed predictor
        R_pre = risk_gaussian_mean(theta, mu_prev, trSigma)
        pop_pre_losses.append(R_pre)

        # policy action (paper-aligned measurability): based on mu_{t-1}
        u = np.zeros(d)
        if regime in ("endogenous", "mixed") and abs(gamma) > 0.0:
            u = -policy_k * mu_prev

        # exogenous increment
        v_exo = exo_vecs[t - 1] if regime in ("exo", "mixed") else np.zeros(d)

        # environment update
        theta_next = theta + v_exo + gamma * u

        # budgets (Fisher-consistent)
        dt_t = fisher_norm(v_exo, Sigma_inv)
        kappa_t = fisher_norm(gamma * u, Sigma_inv)
        sum_dt += dt_t
        sum_kappa += kappa_t
        fisher_path_len += fisher_step_length(theta_next - theta, Sigma_inv)

        # one-step-ahead population risk with same deployed predictor
        R_post = risk_gaussian_mean(theta_next, mu_prev, trSigma)
        pop_post_losses.append(R_post)

        # drift functional term (paper V_T)
        V_terms.append(abs(R_post - R_pre))

        # learner update AFTER observing x_t (mu_t)
        mu = mu_prev + (x_t - mu_prev) / t

        # advance environment
        theta = theta_next

    # err_T compares mu_T to theta_T (last sampled theta)
    theta_T = theta_pre[-1]
    err_T = float(np.linalg.norm(mu - theta_T))

    V_T = float(np.mean(V_terms)) if V_terms else float("nan")

    traj_emp_risk = float(np.mean(emp_losses)) if emp_losses else float("nan")
    traj_pop_risk_pre = float(np.mean(pop_pre_losses)) if pop_pre_losses else float("nan")
    traj_pop_risk_post = float(np.mean(pop_post_losses)) if pop_post_losses else float("nan")

    traj_samp_gap = float(abs(traj_emp_risk - traj_pop_risk_pre))
    traj_rep_gap = float(abs(traj_emp_risk - traj_pop_risk_post)) 

    pop_mse_initial = float(pop_pre_losses[0]) if pop_pre_losses else float("nan")
    pop_mse_final = float(pop_pre_losses[-1]) if pop_pre_losses else float("nan")

    return RunStats(
        err_T=err_T,
        V_T=V_T,
        traj_samp_gap=traj_samp_gap,
        traj_rep_gap=traj_rep_gap,
        sum_dt=sum_dt,
        sum_kappa=sum_kappa,
        fisher_path_len=fisher_path_len,
        traj_emp_risk=traj_emp_risk,
        traj_pop_risk_pre=traj_pop_risk_pre,
        traj_pop_risk_post=traj_pop_risk_post,
        pop_mse_initial=pop_mse_initial,
        pop_mse_final=pop_mse_final,
    )


# --------------------------- Experiment scaffolding -----------------------------

@dataclass(frozen=True)
class MetricSpec:
    name: str
    row_key: str
    label: str
    tag: str


METRIC_SPECS: Dict[str, MetricSpec] = {
    "V_T": MetricSpec(
        name="V_T",
        row_key="V_T",
        label=r"$V_T=\frac1T\sum_t|R(\theta_{t+1},f_t)-R(\theta_t,f_t)|$",
        tag="V_T",
    ),
    "err": MetricSpec(
        name="err",
        row_key="err",
        label=r"$\mathbb{E}_{\text{seeds}}\!\left[\|\hat{\mu}_T-\theta_T\|_2\right]$",
        tag="err",
    ),
    "samp_gap": MetricSpec(
        name="samp_gap",
        row_key="traj_samp_gap",
        label=r"$|\overline{\hat{R}}-\overline{R(\theta_t,f_t)}|$",
        tag="traj_samp_gap",
    ),
    "rep_gap": MetricSpec(
        name="rep_gap",
        row_key="traj_rep_gap",
        label=r"$|\overline{\hat{R}}-\overline{R(\theta_{t+1},f_t)}|$",
        tag="traj_rep_gap",
    ),
}

DEFAULT_METRIC = "V_T"


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
        plane_fit=results_dir / "plane_fit.txt",
        raw_csv=results_dir / "additivity_raw.csv",
        summary_csv=results_dir / "additivity_summary.csv",
        meta_json=results_dir / "additivity_meta.json",
        fig_prefix=results_dir / "budget_scatter",
    )


def simulate_setting(cfg: ExperimentConfig, C_exo: float, gamma: float, seed: int) -> RunStats:
    reg = regime_name(C_exo, gamma)
    regime_map = {
        "iid": ("iid", 0.0, 0.0),
        "exogenous-only": ("exo", C_exo, 0.0),
        "endogenous-only": ("endogenous", 0.0, gamma),
        "mixed": ("mixed", C_exo, gamma),
    }
    regime_name_str, C_use, gamma_use = regime_map[reg]
    return simulate_one_run(
        T=cfg.T,
        d=cfg.d,
        Sigma=cfg.Sigma,
        regime=regime_name_str,
        C_exo=C_use,
        gamma=gamma_use,
        policy_k=cfg.k_policy,
        seed=seed,
    )


def run_grid(cfg: ExperimentConfig) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    rows: List[Dict[str, float]] = []
    raw_rows: List[Dict[str, float]] = []

    for C_exo in cfg.C_exo_grid:
        for gamma in cfg.gamma_grid:
            reg = regime_name(C_exo, gamma)

            stats_list: List[RunStats] = []
            for seed in cfg.seeds:
                s = simulate_setting(cfg, C_exo, gamma, seed)
                stats_list.append(s)
                raw_rows.append(
                    {
                        "C_exo": float(C_exo),
                        "gamma": float(gamma),
                        "T": int(cfg.T),
                        "seed": int(seed),
                        "regime": reg,
                        "err_T": float(s.err_T),
                        "V_T": float(s.V_T),
                        "traj_samp_gap": float(s.traj_samp_gap),
                        "traj_rep_gap": float(s.traj_rep_gap),
                        "sum_dt": float(s.sum_dt),
                        "sum_kappa": float(s.sum_kappa),
                        "fisher_path_len": float(s.fisher_path_len),
                        "traj_emp_risk": float(s.traj_emp_risk),
                        "traj_pop_risk_pre": float(s.traj_pop_risk_pre),
                        "traj_pop_risk_post": float(s.traj_pop_risk_post),
                        "pop_mse_initial": float(s.pop_mse_initial),
                        "pop_mse_final": float(s.pop_mse_final),
                    }
                )

            def mean(attr: str) -> float:
                return float(np.mean([getattr(s, attr) for s in stats_list]))

            rows.append(
                {
                    "C_exo": float(C_exo),
                    "gamma": float(gamma),
                    "T": int(cfg.T),
                    "regime": reg,
                    "err": mean("err_T"),
                    "V_T": mean("V_T"),
                    "traj_samp_gap": mean("traj_samp_gap"),
                    "traj_rep_gap": mean("traj_rep_gap"),
                    "sum_dt": mean("sum_dt"),
                    "sum_kappa": mean("sum_kappa"),
                    "fisher_path_len": mean("fisher_path_len"),
                }
            )

    return rows, raw_rows


def compute_regression_summary(
    rows: Sequence[Dict[str, float]],
    metric_key: str,
    metric_spec: MetricSpec,
) -> RegressionSummary:
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

    group = defaultdict(lambda: {"T": None, "regime": None, "vals": defaultdict(list)})

    keys_to_store = [
        "err_T", "V_T", "traj_samp_gap", "traj_rep_gap",
        "sum_dt", "sum_kappa", "fisher_path_len",
        "traj_emp_risk", "traj_pop_risk_pre", "traj_pop_risk_post",
        "pop_mse_initial", "pop_mse_final",
    ]

    for r in raw_rows:
        key = (r["C_exo"], r["gamma"])
        g = group[key]
        g["T"] = r["T"]
        g["regime"] = r["regime"]
        for k in keys_to_store:
            g["vals"][k].append(float(r[k]))

    summary_rows: List[Dict[str, float]] = []
    for (C_exo, gamma), g in sorted(group.items()):
        row: Dict[str, float] = {
            "C_exo": float(C_exo),
            "gamma": float(gamma),
            "T": int(g["T"]),
            "regime": str(g["regime"]),
        }
        for k in keys_to_store:
            m, se, n = mean_se(g["vals"][k])
            row[f"mean_{k}"] = m
            row[f"se_{k}"] = se
            row["n"] = int(n)

        m_dt = row["mean_sum_dt"]
        m_kap = row["mean_sum_kappa"]
        Tval = row["T"]
        row["mean_C_over_T(alpha_opt)"] = float((m_dt + alpha_opt * m_kap) / (Tval + EPS))
        summary_rows.append(row)

    return summary_rows


def write_plane_fit(path: Path, stats: RegressionSummary) -> None:
    metric_tag = stats.metric_spec.tag
    b0, b_s, b1, b2 = map(float, stats.plane_coef)
    a0, a1 = stats.collapse_coef
    with open(path, "w") as f:
        f.write(f"Additivity plane: {metric_tag} ~ b0 + b_s*T^{-1/2} + b1*(sum_dt/T) + b2*(sum_kappa/T)\n")
        f.write(f"b0   = {b0:.6f}\n")
        f.write(f"b_s  = {b_s:.6f}\n")
        f.write(f"b1   = {b1:.6f}\n")
        f.write(f"b2   = {b2:.6f}\n")
        f.write(f"R^2  = {stats.plane_r2:.4f}\n")
        f.write("\n")
        f.write(f"Collapse: {metric_tag} ~ a0 + a1*(C_T/T)\n")
        f.write(f"alpha_opt = {stats.alpha_opt:.6f}\n")
        f.write(f"a0 = {a0:.6f}\n")
        f.write(f"a1 = {a1:.6f}\n")
        f.write(f"R^2  = {stats.collapse_r2:.4f}\n")


def save_tables(raw_csv: Path,
                summary_csv: Path,
                meta_json: Path,
                raw_rows: Sequence[Dict[str, float]],
                summary_rows: Sequence[Dict[str, float]],
                meta: Dict[str, object]) -> None:
    raw_fields = list(raw_rows[0].keys()) if raw_rows else []
    with open(raw_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)

    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)


def build_meta(cfg: ExperimentConfig,
               stats: RegressionSummary,
               summary_rows: Sequence[Dict[str, float]],
               raw_rows: Sequence[Dict[str, float]]) -> Dict[str, object]:
    trace_sigma = float(np.trace(cfg.Sigma))
    return {
        "figure": "Gaussian additivity sanity check (paper-aligned V_T)",
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
            "spec": f"{stats.metric_spec.tag} ~ b0 + b_s*T^(-1/2) + b1*(sum_dt/T) + b2*(sum_kappa/T)",
            "b0": float(stats.plane_coef[0]),
            "b_s": float(stats.plane_coef[1]),
            "b1": float(stats.plane_coef[2]),
            "b2": float(stats.plane_coef[3]),
            "R2": float(stats.plane_r2),
        },
        "alpha_opt": float(stats.alpha_opt),
        "collapse_fit": {
            "spec": f"{stats.metric_spec.tag} ~ a0 + a1*(C_T/T)",
            "a0": float(stats.collapse_coef[0]),
            "a1": float(stats.collapse_coef[1]),
            "R2": float(stats.collapse_r2),
        },
        "summary_preview": summary_rows[: min(8, len(summary_rows))],
        "raw_head": raw_rows[: min(10, len(raw_rows))],
    }


def plot_budget_scatter(stats: RegressionSummary, fig_prefix: Path) -> None:
    y = stats.y
    C_over_T = stats.C_over_T
    regime = stats.regime
    a0, a1 = stats.collapse_coef
    r2_line = stats.collapse_r2
    alpha_opt = stats.alpha_opt

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
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
        ls="-",
        color="0.25",
        label="linear fit",
        zorder=2,
    )

    ax.set_xlabel(r"Budget Ratio $C_T/T$", fontsize=10)
    ax.set_ylabel(stats.metric_spec.label, labelpad=2, fontsize=10)
    ax.set_ylim(0.0, (y.max() * 1.15) if y.size else 1.0)

    ax.tick_params(axis="both", which="major", labelsize=6.0)
    ax.tick_params(axis="both", which="minor", labelsize=5.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", which="major", alpha=0.45)

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
    inset.axhline(0, color="0.5", lw=0.6, ls="--")
    inset.scatter(C_over_T, residuals, s=10, color="#4f65a3", alpha=0.8, linewidths=0)
    inset.set_xticks([0.0, round(float(xmax) / 2, 2), round(float(xmax), 2)])
    inset.set_yticks([])
    inset.tick_params(labelsize="x-small", pad=1.2, length=2.0)
    inset.set_title("Residuals", fontsize="small", pad=1.5)
    inset.spines["right"].set_visible(False)
    inset.spines["top"].set_visible(False)

    legend_handles = [
        Line2D([], [], marker="o", markersize=4.5, markerfacecolor=palette["iid"], markeredgecolor="none",
               linestyle="none", label="iid"),
        Line2D([], [], marker="s", markersize=4.5, markerfacecolor=palette["exogenous-only"], markeredgecolor="none",
               linestyle="none", label="exogenous only"),
        Line2D([], [], marker="d", markersize=4.5, markerfacecolor=palette["endogenous-only"], markeredgecolor="none",
               linestyle="none", label="endogenous only"),
        Line2D([], [], marker="^", markersize=4.5, markerfacecolor=palette["mixed"], markeredgecolor="none",
               linestyle="none", label="mixed"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
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
        color="0.25",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )
    ax.text(
        0.8,
        0.6,
        fr"$R^2 = {r2_line:.2f}$",
        fontsize=8,
        color="0.25",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    fig.savefig(fig_prefix.with_suffix(".pdf"), **SAVEFIG_KW)
    fig.savefig(fig_prefix.with_suffix(".svg"), **SAVEFIG_KW)
    fig.savefig(fig_prefix.with_suffix(".png"), dpi=600, **SAVEFIG_KW)
    plt.close(fig)


# ------------------------------ CLI / main ------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    choice_list = ", ".join(f"{name}→{spec.label}" for name, spec in METRIC_SPECS.items())
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_SPECS.keys()),
        default=DEFAULT_METRIC,
        help=f"Metric to target (options: {choice_list}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metric_spec = METRIC_SPECS[args.metric]

    cfg = ExperimentConfig(
        d=5,
        Sigma=np.diag([1.0] * 5),
        T=2000,
        seeds=tuple(range(12)),
        k_policy=0.25,
        C_exo_grid=(0.0, 2.0, 4.0, 8.0, 16.0, 32.0),
        gamma_grid=(0.0, 0.01, 0.02, 0.04, 0.08),
    )

    results_dir = create_results_dir(f"gaussian_additivity_{metric_spec.name}")
    paths = build_export_paths(results_dir)

    rows, raw_rows = run_grid(cfg)

    metric_key = metric_spec.row_key
    if not rows or metric_key not in rows[0]:
        raise KeyError(f"Metric row_key='{metric_key}' not found in aggregated rows.")

    stats = compute_regression_summary(rows, metric_key, metric_spec)
    summary_rows = build_summary_rows(raw_rows, stats.alpha_opt)

    write_plane_fit(paths.plane_fit, stats)
    meta = build_meta(cfg, stats, summary_rows, raw_rows)
    save_tables(paths.raw_csv, paths.summary_csv, paths.meta_json, raw_rows, summary_rows, meta)
    plot_budget_scatter(stats, paths.fig_prefix)

    print(f"[ok] wrote results to: {results_dir}")
    print(f"[ok] metric: {metric_spec.name} (row_key={metric_spec.row_key})")
    print(f"[ok] plane fit:   {paths.plane_fit}")
    print(f"[ok] raw csv:     {paths.raw_csv}")
    print(f"[ok] summary csv: {paths.summary_csv}")
    print(f"[ok] meta json:   {paths.meta_json}")
    print(f"[ok] figures:     {paths.fig_prefix}.(pdf|svg|png)")


if __name__ == "__main__":
    main()
