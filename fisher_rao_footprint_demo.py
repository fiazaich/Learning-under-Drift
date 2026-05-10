#!/usr/bin/env python3
"""fisher_rao_footprint_demo.py

Demonstrate Fisher--Rao contraction under a fixed Markov kernel K in the
Gaussian-location model.

Environment:
  P_t = N(theta_t, Sigma) with drift--feedback dynamics similar in spirit to
  gaussian_additivity_experiment.py (exo Fisher-budgeted increments + endogenous feedback).

Observable channel (fixed Markov kernel):
  K: u = B x + xi,   xi ~ N(0, sigmaK^2 I_k)

Then Q_t := K_# P_t is Gaussian with mean B theta_t and covariance
S = B Sigma B^T + sigmaK^2 I_k.

Outputs (written under results/fr_footprint_<rand>/):
  - fr_footprint_rate_demo.pdf/png : local Fisher rate and footprints.
  - fr_footprint_contraction_demo.pdf/png : stepwise contraction scatter.
  - fr_footprint_rate_scatter.pdf/png : optional cross-seed terminal-rate
    contraction scatter (when --multi-seed > 0).

Example:
  python fisher_rao_footprint_demo.py \
    --T 4000 --d 5 --regime mixed --C-exo 2.0 --gamma 0.01 --k 0.25 \
    --kdim 2 --sigmaK 0.2 --extra-kernels \
    --burst --burst-period 400 --burst-hi 4.0 \
    --rate-window 60 --multi-seed 12
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

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

PALETTE = ["#1f77b4", "#ff7f0e"] + [plt.get_cmap("tab10")(i) for i in range(2, 10)]
MARKERS = ["o", "s", "^", "d", "v", "<", ">", "P", "X", "*"]
SCATTER_SMALL = dict(s=16, linewidths=0.9, alpha=0.7)
from gaussian_additivity_experiment import (
    cholesky_sampler,
    fisher_norm,
    create_results_dir,
    EPS,
)


@dataclass
class Trajectory:
    theta: np.ndarray          # (T+1, d)
    mu_prev: np.ndarray        # (T, d) predictor used at step t (mu_{t-1})
    dt: np.ndarray             # (T,) exo Fisher step lengths
    kappa: np.ndarray          # (T,) endo Fisher step lengths
    l_intrinsic: np.ndarray    # (T,) ||theta_{t+1}-theta_t||_{Sigma^{-1}}


def make_burst_weights_contiguous(
    T: int,
    period: int,
    hi: float,
    rng: np.random.Generator,
    frac_hi: float = 0.5,
) -> np.ndarray:
    """Contiguous bursts with mean exactly 1.

    Each period is split into a contiguous high segment (fraction frac_hi) and
    a contiguous low segment, with a random phase shift per period.

    We choose lo so that the *period average* is 1, then renormalize globally
    to mean 1.
    """
    period = max(2, int(period))
    hi = float(hi)
    frac_hi = float(frac_hi)
    lo = (1.0 - frac_hi * hi) / max(1e-12, (1.0 - frac_hi))
    if lo <= 0:
        lo = 0.05

    w = np.empty(T, dtype=float)
    idx = 0
    while idx < T:
        end = min(T, idx + period)
        L = end - idx
        m = max(1, int(round(frac_hi * L)))
        # contiguous high block of length m with random start within the period
        start = int(rng.integers(low=0, high=max(1, L - m + 1)))
        block = np.full(L, lo, dtype=float)
        block[start : start + m] = hi
        w[idx:end] = block
        idx = end

    # global renormalization to mean 1
    w /= (np.mean(w) + 1e-12)
    return w


def trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing-window mean with the same length as x.

    For t < window, uses mean over x[:t+1].
    """
    window = max(1, int(window))
    out = np.empty_like(x, dtype=float)
    c = np.cumsum(x, dtype=float)
    for t in range(len(x)):
        a = max(0, t - window + 1)
        s = c[t] - (c[a - 1] if a > 0 else 0.0)
        out[t] = s / (t - a + 1)
    return out


def kernel_label_sort_key(label: str) -> Tuple[int, str]:
    match = re.search(r"k=(\d+)", label)
    if match:
        return int(match.group(1)), label
    return 10**9, label


def set_rate_legend_ordered_by_k(ax: plt.Axes, **legend_kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    intrinsic = [(h, lab) for h, lab in zip(handles, labels) if "intrinsic rate" in lab]
    induced = [(h, lab) for h, lab in zip(handles, labels) if "intrinsic rate" not in lab]
    induced.sort(key=lambda item: kernel_label_sort_key(item[1]), reverse=True)

    ordered = intrinsic + induced
    ax.legend(
        [h for h, _lab in ordered],
        [lab for _h, lab in ordered],
        **legend_kwargs,
    )


def simulate_with_trajectory(
    *,
    T: int,
    d: int,
    Sigma: np.ndarray,
    regime: str,
    C_exo: float,
    gamma: float,
    policy_k: float,
    seed: int,
    burst: bool,
    burst_period: int,
    burst_hi: float,
) -> Trajectory:
    rng = np.random.default_rng(seed)
    sampler = cholesky_sampler(Sigma, rng)
    Sigma_inv = np.linalg.inv(Sigma)

    theta = np.zeros(d)
    mu = np.zeros(d)

    # Exogenous vectors with Fisher-normalized directions so sum_t ||v_exo_t||_F ≈ C_exo.
    exo_vecs = np.zeros((T, d))
    if regime in ("exo", "mixed") and C_exo > 0.0:
        base_step = C_exo / T
        w = np.ones(T)
        if burst:
            w = make_burst_weights_contiguous(T, burst_period, burst_hi, rng)
        raw = rng.normal(size=(T, d))
        for t in range(T):
            v = raw[t]
            nF = fisher_norm(v, Sigma_inv)
            if nF < EPS:
                v = rng.normal(size=d)
                nF = fisher_norm(v, Sigma_inv) + EPS
            step_F = base_step * w[t]
            exo_vecs[t] = v * (step_F / (nF + EPS))

    theta_hist = np.zeros((T + 1, d))
    mu_prev_hist = np.zeros((T, d))
    dt_hist = np.zeros(T)
    kappa_hist = np.zeros(T)
    l_hist = np.zeros(T)

    theta_hist[0] = theta

    for t in range(1, T + 1):
        mu_prev = mu.copy()
        mu_prev_hist[t - 1] = mu_prev

        x_t = sampler(theta)

        u = np.zeros(d)
        if regime in ("endogenous", "mixed") and gamma != 0.0:
            u = -policy_k * mu_prev

        v_end = gamma * u

        v_exo = np.zeros(d)
        if regime in ("exo", "mixed"):
            v_exo = exo_vecs[t - 1]

        dt = fisher_norm(v_exo, Sigma_inv) if (regime in ("exo", "mixed")) else 0.0
        kappa = fisher_norm(v_end, Sigma_inv) if (regime in ("endogenous", "mixed")) else 0.0
        dt_hist[t - 1] = float(dt)
        kappa_hist[t - 1] = float(kappa)

        theta_next = theta + v_exo + v_end

        l = fisher_norm(theta_next - theta, Sigma_inv)
        l_hist[t - 1] = float(l)

        mu = ((t - 1) * mu + x_t) / t

        theta = theta_next
        theta_hist[t] = theta

    return Trajectory(
        theta=theta_hist,
        mu_prev=mu_prev_hist,
        dt=dt_hist,
        kappa=kappa_hist,
        l_intrinsic=l_hist,
    )


def footprint_lengths_linear_kernel(theta: np.ndarray, Sigma: np.ndarray, B: np.ndarray, sigmaK: float) -> np.ndarray:
    """Per-step FR lengths after K: u = Bx + xi, xi ~ N(0, sigmaK^2 I)."""
    S = B @ Sigma @ B.T + (sigmaK**2) * np.eye(B.shape[0])
    S_inv = np.linalg.inv(S)
    T = theta.shape[0] - 1
    lK = np.zeros(T)
    for t in range(T):
        dmu = B @ (theta[t + 1] - theta[t])
        lK[t] = float(np.sqrt(dmu.T @ (S_inv @ dmu)))
    return lK


def build_kernels(rng: np.random.Generator, d: int, kdim: int, sigmaK: float, extra: bool) -> List[Tuple[str, np.ndarray, float]]:
    def make_B(k: int) -> np.ndarray:
        B = rng.normal(size=(k, d))
        B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return B

    kernels: List[Tuple[str, np.ndarray, float]] = []
    kernels.append((f"k={kdim}, $\\sigma_K$={sigmaK:g}", make_B(kdim), sigmaK))

    if extra:
        k1 = min(d, max(kdim + 1, kdim))
        kernels.append((f"k={k1}, $\\sigma_K$={max(0.05, sigmaK/2):g}", make_B(k1), max(0.05, sigmaK / 2)))
        k2 = max(1, kdim - 1)
        kernels.append((f"k={k2}, $\\sigma_K$={sigmaK*2:g}", make_B(k2), sigmaK * 2))

    return kernels


def plot_main(
    *,
    outdir,
    l: np.ndarray,
    lKs: Dict[str, np.ndarray],
    rate_window: int,
) -> None:
    T = len(l)
    tgrid = np.arange(1, T + 1)

    rate = trailing_mean(l, rate_window)
    rateKs = {lab: trailing_mean(v, rate_window) for lab, v in lKs.items()}

    fig1 = plt.figure(figsize=(3.6, 2.6))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(tgrid, rate, label="intrinsic rate $\\bar a_t(P)$", linewidth=1.6, color=PALETTE[0])
    for idx, (lab, rK) in enumerate(rateKs.items()):
        ax1.plot(
            tgrid,
            rK,
            label=("induced rate $\\bar a_t(K_{\\sharp}P)$ (" + lab + ")"),
            linewidth=1.3,
            color=PALETTE[(idx + 1) % len(PALETTE)],
            linestyle="-",
        )
    ax1.set_xlabel("time $t$")
    ax1.set_ylabel("local average rate")
    ax1.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax1.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax1.grid(True, alpha=0.25)
    set_rate_legend_ordered_by_k(ax1, loc="upper right", frameon=False)

    pdf1 = outdir / "fr_footprint_rate_demo.pdf"
    plt.savefig(pdf1, dpi=250, bbox_inches="tight")
    plt.savefig(pdf1.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2 = plt.figure(figsize=(3.6, 2.6))
    ax2 = fig2.add_subplot(1, 1, 1)
    m = float(max(np.max(l), max(np.max(v) for v in lKs.values())) + 1e-12)
    ax2.plot([0, m], [0, m], linewidth=1.0, linestyle="--", color="0.35", label="$y=x$")
    for idx, (lab, v) in enumerate(lKs.items()):
        ax2.scatter(
            l,
            v,
            color=PALETTE[(idx + 1) % len(PALETTE)],
            marker=MARKERS[idx % len(MARKERS)],
            label=lab,
            **SCATTER_SMALL,
        )
    ax2.set_xlabel("intrinsic step length $d_F(P_{t+1},P_t)$")
    ax2.set_ylabel("induced step length $d_F((K_{\\sharp}P)_{t+1},(K_{\\sharp}P)_t)$")
    ax2.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax2.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", frameon=False)

    pdf2 = outdir / "fr_footprint_contraction_demo.pdf"
    plt.savefig(pdf2, dpi=250, bbox_inches="tight")
    plt.savefig(pdf2.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig2)


def plot_multi_seed_scatter(
    *,
    outdir,
    rates_P: np.ndarray,
    rates_Q: Dict[str, np.ndarray],
) -> None:
    fig = plt.figure(figsize=(3.4, 2.6))
    ax = fig.add_subplot(1, 1, 1)

    mx = float(max(np.max(rates_P), max(np.max(v) for v in rates_Q.values())) + 1e-12)
    ax.plot([0, mx], [0, mx], linestyle="--", linewidth=1.0, color="0.35")

    for idx, (lab, y) in enumerate(rates_Q.items()):
        ax.scatter(
            rates_P,
            y,
            color=PALETTE[(idx + 1) % len(PALETTE)],
            marker=MARKERS[idx % len(MARKERS)],
            label=lab,
            **SCATTER_SMALL,
        )

    ax.set_xlabel("intrinsic rate $A_T(P)/T$")
    ax.set_ylabel("observed rate $A_T(K_{\\sharp}P)/T$")
    ax.set_title("Terminal rate contraction")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    pdf = outdir / "fr_footprint_rate_scatter.pdf"
    plt.savefig(pdf, dpi=250, bbox_inches="tight")
    plt.savefig(pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=4000)
    p.add_argument("--d", type=int, default=5)
    p.add_argument("--regime", type=str, default="mixed", choices=["exo", "endogenous", "mixed"])
    p.add_argument("--C-exo", dest="C_exo", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--k", type=float, default=0.25, help="feedback gain (policy scale)")
    p.add_argument("--seed", type=int, default=0)

    # kernel params
    p.add_argument("--kdim", type=int, default=2)
    p.add_argument("--sigmaK", type=float, default=0.2)
    p.add_argument("--extra-kernels", action="store_true")

    # bursty exogenous allocation
    p.add_argument("--burst", action="store_true")
    p.add_argument("--burst-period", type=int, default=400)
    p.add_argument("--burst-hi", type=float, default=4.0)

    # rate visualization
    p.add_argument("--rate-window", type=int, default=60, help="trailing window for local rate")

    # multi-seed
    p.add_argument("--multi-seed", type=int, default=0, help="if >0, run this many seeds and plot terminal rates")

    args = p.parse_args()

    Sigma = np.eye(args.d)
    rng_kernel = np.random.default_rng(12345)
    kernels = build_kernels(rng_kernel, args.d, args.kdim, args.sigmaK, args.extra_kernels)

    outdir = create_results_dir("fr_footprint")

    # single-run trajectory
    traj = simulate_with_trajectory(
        T=args.T,
        d=args.d,
        Sigma=Sigma,
        regime=args.regime,
        C_exo=args.C_exo,
        gamma=args.gamma,
        policy_k=args.k,
        seed=args.seed,
        burst=args.burst,
        burst_period=args.burst_period,
        burst_hi=args.burst_hi,
    )

    l = traj.l_intrinsic
    lKs: Dict[str, np.ndarray] = {}
    for lab, B, sig in kernels:
        lKs[lab] = footprint_lengths_linear_kernel(traj.theta, Sigma, B, sig)

    # sanity: stepwise contraction (numerical tolerance)
    max_viol = max(float(np.max(v - l)) for v in lKs.values())

    plot_main(outdir=outdir, l=l, lKs=lKs, rate_window=args.rate_window)

    # optional multi-seed scatter
    if args.multi_seed and args.multi_seed > 0:
        seeds = list(range(args.multi_seed))
        rates_P = np.zeros(len(seeds))
        rates_Q: Dict[str, np.ndarray] = {lab: np.zeros(len(seeds)) for lab, _B, _sig in kernels}

        for i, sd in enumerate(seeds):
            tr = simulate_with_trajectory(
                T=args.T,
                d=args.d,
                Sigma=Sigma,
                regime=args.regime,
                C_exo=args.C_exo,
                gamma=args.gamma,
                policy_k=args.k,
                seed=sd,
                burst=args.burst,
                burst_period=args.burst_period,
                burst_hi=args.burst_hi,
            )
            l_i = tr.l_intrinsic
            rates_P[i] = float(np.sum(l_i) / args.T)
            for lab, B, sig in kernels:
                lK_i = footprint_lengths_linear_kernel(tr.theta, Sigma, B, sig)
                rates_Q[lab][i] = float(np.sum(lK_i) / args.T)

        plot_multi_seed_scatter(outdir=outdir, rates_P=rates_P, rates_Q=rates_Q)

    # write summary
    summary = []
    summary.append(f"max_step_violation (lK - l) = {max_viol:.3e}")
    summary.append(f"A_T(P)/T   = {float(np.sum(l)/args.T):.6g}")
    for lab, v in lKs.items():
        summary.append(f"A_T(Q)/T   ({lab}) = {float(np.sum(v)/args.T):.6g}  (ratio {(np.sum(v)/np.sum(l)):.3f})")
    (outdir / "summary_rate.txt").write_text("\n".join(summary))

    print(f"[ok] wrote {outdir / 'fr_footprint_rate_demo.pdf'}")
    print(f"[ok] wrote {outdir / 'fr_footprint_contraction_demo.pdf'}")
    if args.multi_seed and args.multi_seed > 0:
        print(f"[ok] wrote {outdir / 'fr_footprint_rate_scatter.pdf'}")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
