#!/usr/bin/env python3
"""Gaussian T-sweep saturation experiment (paper-aligned).

Shows:
1) vanishing drift (delta_F=0): Delta_rep ~ Delta_sam -> 0 (sampling dominated)
2) bounded persistent drift (delta_F>0): Delta_rep decreases then plateaus (drift floor)

Paper-aligned quantities (f_t measurable w.r.t. F_{t-1}):
- deployed predictor at time t: f_t = mu_{t-1}
- empirical risk:   Rhat_T   = (1/T) sum ||x_t - mu_{t-1}||^2
- pop risk:         R_T      = (1/T) sum [tr(Sigma) + ||theta_t   - mu_{t-1}||^2]
- one-step-ahead:   R_T^+    = (1/T) sum [tr(Sigma) + ||theta_{t+1}- mu_{t-1}||^2]
- gaps: Delta_sam=|Rhat_T-R_T|, Delta_rep=|Rhat_T-R_T^+|
- drift term: V_T = (1/T) sum |R(theta_{t+1},mu_{t-1}) - R(theta_t,mu_{t-1})|

Outputs:
- raw.csv, summary.csv
- saturation_Delta_rep.(pdf|svg|png)
- saturation_Delta_sam.(pdf|svg|png)
- saturation_V_T.(pdf|svg|png)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter, MaxNLocator, AutoMinorLocator

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

from matplotlib.ticker import MaxNLocator, AutoMinorLocator

EPS = 1e-12
FIGSIZE = (3.4, 2.4)
SAVEFIG_KW = {}


@dataclass
class RunStats:
    T: int
    delta_F: float
    seed: int
    Delta_rep: float
    Delta_sam: float
    V_T: float
    Rhat_T: float
    R_T: float
    R_T_plus: float
    mean_fisher_step: float
    mean_abs_theta: float


def create_results_dir(prefix: str) -> Path:
    base = Path("results")
    base.mkdir(parents=True, exist_ok=True)
    tag = int(np.random.default_rng().integers(0, 10**9))
    out = base / f"{prefix}_{tag}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def cholesky_sampler(Sigma: np.ndarray, rng: np.random.Generator):
    L = np.linalg.cholesky(Sigma)
    d = Sigma.shape[0]

    def sample(theta_vec: np.ndarray) -> np.ndarray:
        z = rng.standard_normal(d)
        return theta_vec + L @ z

    return sample


def fisher_norm(v: np.ndarray, Sigma_inv: np.ndarray) -> float:
    return float(np.sqrt(v @ Sigma_inv @ v))


def risk_gaussian_mean(theta_vec: np.ndarray, mu: np.ndarray, trSigma: float) -> float:
    return float(trSigma + np.sum((theta_vec - mu) ** 2))


def mean_se(vals: Iterable[float]) -> Tuple[float, float, int]:
    vals = list(vals)
    n = len(vals)
    m = float(np.mean(vals)) if n else float("nan")
    se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return m, se, n


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_saturation(summary_rows: Sequence[Dict[str, float]], fig_path: Path, which: str) -> None:
    deltas = sorted({float(r["delta_F"]) for r in summary_rows})
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    fig.set_size_inches(*FIGSIZE)

    for delta in deltas:
        rows = [r for r in summary_rows if float(r["delta_F"]) == float(delta)]
        rows = sorted(rows, key=lambda r: int(r["T"]))
        Tvals = np.array([int(r["T"]) for r in rows], dtype=float)
        y = np.array([float(r[f"mean_{which}"]) for r in rows], dtype=float)
        yse = np.array([float(r[f"se_{which}"]) for r in rows], dtype=float)

        label = r"$\delta_F$=" + f"{delta:g}"
        ax.plot(Tvals, y, marker="o", markersize=3.0, lw=0.9, label=label)
        ax.fill_between(Tvals, y - yse, y + yse, alpha=0.15, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel(r"Time horizon $T$", fontsize=10)

    if which == "Delta_rep":
        ax.set_ylabel(r"$\Delta_T^{\mathrm{rep}}$", fontsize=10)
        ax.set_title(r"Repro gap under bounded persistent drift", fontsize=10, pad=2)
    elif which == "Delta_sam":
        ax.set_ylabel(r"$\Delta_T^{\mathrm{sam}}$", fontsize=10)
        ax.set_title(r"Sampling term (falls with $T$)", fontsize=10, pad=2)
    else:
        ax.set_ylabel(r"$V_T$", fontsize=10)
        ax.set_title(r"Drift term (sets the floor)", fontsize=10, pad=2)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", which="major", alpha=0.45)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")

    fig.savefig(fig_path.with_suffix(".pdf"), **SAVEFIG_KW)
    fig.savefig(fig_path.with_suffix(".svg"), **SAVEFIG_KW)
    fig.savefig(fig_path.with_suffix(".png"), dpi=600, **SAVEFIG_KW)
    plt.close(fig)

def plot_components_overlay(summary_rows: Sequence[Dict[str, float]],
                            fig_path: Path,
                            delta_show: float,
                            include_iid: bool = True) -> None:
    """
    One plot with TWO curves vs T for a single delta_F:
      - Delta_sam (falls with T)
      - V_T (roughly flat)
    Optionally overlays iid (delta_F=0) as faint reference.
    """

    def curve(delta: float):
        rows = [r for r in summary_rows if abs(float(r["delta_F"]) - delta) < 1e-12]
        rows = sorted(rows, key=lambda r: int(r["T"]))
        T = np.array([int(r["T"]) for r in rows], dtype=float)

        sam = np.array([float(r["mean_Delta_sam"]) for r in rows], dtype=float)
        sam_se = np.array([float(r["se_Delta_sam"]) for r in rows], dtype=float)

        V = np.array([float(r["mean_V_T"]) for r in rows], dtype=float)
        V_se = np.array([float(r["se_V_T"]) for r in rows], dtype=float)
        return T, sam, sam_se, V, V_se

    T, sam, sam_se, V, V_se = curve(delta_show)

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    fig.set_size_inches(*FIGSIZE)

    # Curves
    ax.plot(T, sam, marker="o", markersize=3.0, lw=0.9,
            label=rf"$\Delta_T^{{\mathrm{{sam}}}}$ ($\delta_F={delta_show:g}$)")
    ax.fill_between(T, sam - sam_se, sam + sam_se, alpha=0.15, linewidth=0)

    ax.plot(T, V, marker="s", markersize=3.0, lw=0.9,
            label=rf"$V_T$ ($\delta_F={delta_show:g}$)")
    ax.fill_between(T, V - V_se, V + V_se, alpha=0.15, linewidth=0)

    if include_iid and abs(delta_show) > 1e-12:
        T0, sam0, sam0_se, V0, V0_se = curve(0.0)
        ax.plot(T0, sam0, lw=0.9, alpha=0.5, label=r"$\Delta_T^{\mathrm{sam}}$ (iid)")
        ax.plot(T0, V0, lw=0.9, alpha=0.5, label=r"$V_T$ (iid)")

    ax.set_xscale("log")
    ax.set_xlabel(r"Horizon $T$", fontsize=9)
    ax.set_ylabel(r"Magnitude", fontsize=9)

   
    ax.tick_params(axis="both", which="major", labelsize=8, length=3)
    ax.tick_params(axis="both", which="minor", labelsize=8, length=2)

    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.6)

    ax.legend(frameon=False, fontsize=7, ncol=1, loc="center right")

    fig.savefig(fig_path.with_suffix(".pdf"), **SAVEFIG_KW)
    fig.savefig(fig_path.with_suffix(".svg"), **SAVEFIG_KW)
    fig.savefig(fig_path.with_suffix(".png"), dpi=600, **SAVEFIG_KW)
    plt.close(fig)


def reflect_step(theta: float, step: float, direction: float, R: float) -> Tuple[float, float]:
    """
    Reflecting dynamics in [-R, R] with constant step magnitude.
    Returns (theta_next, direction_next).
    """
    proposed = theta + direction * step
    if proposed > R:
        # reflect across +R
        theta_next = 2 * R - proposed
        direction_next = -direction
    elif proposed < -R:
        # reflect across -R
        theta_next = -2 * R - proposed
        direction_next = -direction
    else:
        theta_next = proposed
        direction_next = direction
    return float(theta_next), float(direction_next)


def simulate_one_run_reflect(
    T: int,
    d: int,
    Sigma: np.ndarray,
    Sigma_inv: np.ndarray,
    delta_F: float,
    R: float,
    seed: int,
) -> RunStats:
    rng = np.random.default_rng(seed)
    sampler = cholesky_sampler(Sigma, rng)
    trSigma = float(np.trace(Sigma))

    # Sigma is sigma_scale * I; step_euc is calibrated so Fisher step = delta_F.
    sigma = float(np.sqrt(Sigma[0, 0]))
    step_euc = delta_F * sigma  # so Fisher step = step/sigma = delta_F

    theta = 0.0
    direction = +1.0  # bounce back and forth
    mu = np.zeros(d)

    emp_losses: List[float] = []
    pop_at_t: List[float] = []
    pop_at_t_plus: List[float] = []
    V_terms: List[float] = []

    fisher_steps: List[float] = []
    abs_thetas: List[float] = []

    for t in range(1, T + 1):
        mu_prev = mu.copy()

        theta_vec = np.zeros(d)
        theta_vec[0] = theta

        x_t = sampler(theta_vec)
        emp_losses.append(float(np.sum((x_t - mu_prev) ** 2)))

        R_t = risk_gaussian_mean(theta_vec, mu_prev, trSigma)
        pop_at_t.append(R_t)

        theta_next, direction = reflect_step(theta, step_euc, direction, R)

        theta_vec_next = np.zeros(d)
        theta_vec_next[0] = theta_next

        step_vec = theta_vec_next - theta_vec
        fisher_steps.append(fisher_norm(step_vec, Sigma_inv))
        abs_thetas.append(abs(theta))

        R_t_plus = risk_gaussian_mean(theta_vec_next, mu_prev, trSigma)
        pop_at_t_plus.append(R_t_plus)
        V_terms.append(abs(R_t_plus - R_t))

        mu = mu_prev + (x_t - mu_prev) / t
        theta = theta_next

    Rhat_T = float(np.mean(emp_losses))
    R_T = float(np.mean(pop_at_t))
    R_T_plus = float(np.mean(pop_at_t_plus))

    Delta_sam = float(abs(Rhat_T - R_T))
    Delta_rep = float(abs(Rhat_T - R_T_plus))
    V_T = float(np.mean(V_terms))

    return RunStats(
        T=int(T),
        delta_F=float(delta_F),
        seed=int(seed),
        Delta_rep=Delta_rep,
        Delta_sam=Delta_sam,
        V_T=V_T,
        Rhat_T=Rhat_T,
        R_T=R_T,
        R_T_plus=R_T_plus,
        mean_fisher_step=float(np.mean(fisher_steps) if fisher_steps else 0.0),
        mean_abs_theta=float(np.mean(abs_thetas) if abs_thetas else 0.0),
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sigma_scale", type=float, default=1.0,
                   help="Sigma = sigma_scale * I (default 1.0).")
    p.add_argument("--radius", type=float, default=1.0,
                   help="Reflecting bounds [-R, R] (default 1.0).")
    p.add_argument("--seed_count", type=int, default=40,
                   help="Number of seeds (default 40).")
    return p.parse_args()


def main():
    args = parse_args()

    d = 5
    Sigma = args.sigma_scale * np.eye(d)
    Sigma_inv = np.linalg.inv(Sigma)

    seeds = tuple(range(args.seed_count))

    # log-spaced T grid
    #T_grid = (100, 200, 400, 800, 1600, 3200, 6400)
    T_grid = (200, 400, 800, 1600, 3200, 6400, 12800)

    # per-step Fisher motion levels (include 0 baseline)
    
    delta_grid = (0.00, 0.1, 0.200)

    outdir = create_results_dir("gaussian_T_sweep_reflect")
    raw_path = outdir / "raw.csv"
    summary_path = outdir / "summary.csv"

    raw_rows: List[Dict[str, object]] = []
    for delta in delta_grid:
        for T in T_grid:
            for seed in seeds:
                s = simulate_one_run_reflect(
                    T=T, d=d, Sigma=Sigma, Sigma_inv=Sigma_inv,
                    delta_F=delta, R=args.radius, seed=seed
                )
                raw_rows.append({
                    "T": s.T,
                    "delta_F": s.delta_F,
                    "seed": s.seed,
                    "Delta_rep": s.Delta_rep,
                    "Delta_sam": s.Delta_sam,
                    "V_T": s.V_T,
                    "Rhat_T": s.Rhat_T,
                    "R_T": s.R_T,
                    "R_T_plus": s.R_T_plus,
                    "mean_fisher_step": s.mean_fisher_step,
                    "mean_abs_theta": s.mean_abs_theta,
                })

    # summarize by (delta_F, T)
    from collections import defaultdict
    grp = defaultdict(list)
    for r in raw_rows:
        grp[(float(r["delta_F"]), int(r["T"]))].append(r)

    summary_rows: List[Dict[str, float]] = []
    for (delta, T), rows in sorted(grp.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        def col(k: str) -> List[float]:
            return [float(rr[k]) for rr in rows]

        m_rep, se_rep, n = mean_se(col("Delta_rep"))
        m_sam, se_sam, _ = mean_se(col("Delta_sam"))
        m_V, se_V, _ = mean_se(col("V_T"))

        summary_rows.append({
            "delta_F": float(delta),
            "T": int(T),
            "n": int(n),
            "mean_Delta_rep": float(m_rep),
            "se_Delta_rep": float(se_rep),
            "mean_Delta_sam": float(m_sam),
            "se_Delta_sam": float(se_sam),
            "mean_V_T": float(m_V),
            "se_V_T": float(se_V),
            "mean_fisher_step": float(np.mean(col("mean_fisher_step"))),
            "mean_abs_theta": float(np.mean(col("mean_abs_theta"))),
        })

    write_csv(raw_path, raw_rows)
    write_csv(summary_path, summary_rows)

    plot_saturation(summary_rows, outdir / "saturation_Delta_rep", which="Delta_rep")
    plot_saturation(summary_rows, outdir / "saturation_Delta_sam", which="Delta_sam")
    plot_saturation(summary_rows, outdir / "saturation_V_T", which="V_T")
    plot_components_overlay(
        summary_rows,
        outdir / "components_overlay",
        delta_show=0.1,     
        include_iid=True
    )

    print(f"[ok] wrote results to: {outdir}")
    print(f"[ok] Sigma = {args.sigma_scale} * I, reflecting bounds R = {args.radius}")
    print(f"[ok] raw:     {raw_path}")
    print(f"[ok] summary: {summary_path}")
    print(f"[ok] figures: saturation_(Delta_rep|Delta_sam|V_T).(pdf|svg|png)")
    print(f"[note] summary.csv: mean_fisher_step should be ~ delta_F (sanity check).")


if __name__ == "__main__":
    main()
