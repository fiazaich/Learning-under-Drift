"""Modular plotting utility for the NN speed-limit fits."""
from __future__ import annotations

"""
Plotting strategy:
1. Fit the regression plane using only the per-setting means (one point per (T, ratio, gamma)).
2. Plot those mean points prominently (these are the regression inputs).
3. Overlay the individual per-seed observations with low-alpha, tiny markers after adding small
   horizontal jitter in (C_T/T, T^{-1/2}) to avoid lattice artifacts (dependent variable is untouched).
4. Keep the fitted plane clean/light so it remains readable.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import json
import numpy as np
import numpy.linalg as npl
import pandas as pd
# Use Agg backend for non-interactive plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots

from out_utils import create_results_dir

plt.style.use(['science', 'ieee'])


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class PlotInputs:
    y: np.ndarray
    y_label: str
    target_kind: str
    T: np.ndarray
    gamma: np.ndarray
    ratio: np.ndarray
    sum_dt_over_T: np.ndarray
    sum_kappa_over_T: np.ndarray
    regime: np.ndarray
    meta: Dict[str, object]


@dataclass
class FitResults:
    alpha_theory: float
    c0: float
    c1: float
    c2: float
    r2_full: float
    r2_noT: float
    r2_noC: float
    a0_res: float
    a1_res: float
    r2_res: float
    C_over_T: np.ndarray
    C_over_T_plot: np.ndarray
    y_res: np.ndarray
    inv_sqrt_T: np.ndarray
    y: np.ndarray
    y_label: str
    regime: np.ndarray
    mask_eval: np.ndarray
    r2_full_eval: float
    r2_noT_eval: float
    r2_noC_eval: float
    a0_eval_collapse: float
    a1_eval_collapse: float
    r2_eval_collapse: float


@dataclass
class FigurePaths:
    residual: Path
    ablation: Path
    plane: Path


@dataclass
class RawScatter:
    C_over_T: np.ndarray
    inv_sqrt_T: np.ndarray
    gaps: np.ndarray
    seeds: np.ndarray
    is_hold: np.ndarray


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #
def load_inputs(csv_path: Path, meta_path: Path) -> PlotInputs:
    df = pd.read_csv(csv_path)
    meta: Dict[str, object] = {}
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except FileNotFoundError:
        meta = {}

    if "mean_gen_gap_traj" in df.columns:
        y = df["mean_gen_gap_traj"].to_numpy(float)
        y_label = r"Risk gap $|\widehat R_T - R_T|$"
        target_kind = "gap"
    elif "mean_gen_gap_T" in df.columns:
        y = df["mean_gen_gap_T"].to_numpy(float)
        y_label = r"Risk gap $|\widehat R_T - R_T|$"
        target_kind = "gap"
    else:
        y = df["mean_err_T"].to_numpy(float)
        y_label = r"$\mathbb{E}[E_T]$ (parameter error)"
        target_kind = "err"

    T = df["T"].to_numpy(float)
    gamma = df["gamma"].to_numpy(float) if "gamma" in df.columns else np.zeros_like(T)
    ratio = df["C_exo_ratio"].to_numpy(float) if "C_exo_ratio" in df.columns else np.zeros_like(T)
    sum_dt_over_T = (df["mean_sum_dt"] / df["T"]).to_numpy(float)
    sum_kappa_over_T = (df["mean_sum_kappa"] / df["T"]).to_numpy(float)
    regime = df.get("regime", pd.Series(["mixed"] * len(df))).to_numpy(str)

    return PlotInputs(
        y=y,
        y_label=y_label,
        target_kind=target_kind,
        T=T,
        gamma=gamma,
        ratio=ratio,
        sum_dt_over_T=sum_dt_over_T,
        sum_kappa_over_T=sum_kappa_over_T,
        regime=regime,
        meta=meta,
    )


def split_fit_eval(inputs: PlotInputs):
    hold = inputs.meta.get("holdout", {})
    strategy = hold.get("strategy")
    n = inputs.T.shape[0]
    if strategy == "random_uniform_20pct":
        seed = int(hold.get("seed", 0))
        frac = float(hold.get("fraction", hold.get("frac", 0.2)))
        rng = np.random.default_rng(seed)
        eval_mask = rng.uniform(size=n) < frac
        if not np.any(eval_mask):
            eval_mask[rng.integers(0, n)] = True
        fit_mask = ~eval_mask
        if not np.any(fit_mask):
            idx = rng.integers(0, n)
            fit_mask[idx] = True
            eval_mask[idx] = False
        return fit_mask, eval_mask

    T_hold = hold.get("T")
    if T_hold is None:
        mask_fit = np.ones_like(inputs.T, bool)
    else:
        mask_fit = (inputs.T != T_hold)
    return mask_fit, ~mask_fit


def select_alpha(meta: Dict[str, object], default: float = 1.0) -> float:
    alpha_candidates: list[float] = []
    rg = meta.get("risk_gap_plane", {})
    if rg:
        b1g = float(rg.get("b1", 0.0))
        b2g = float(rg.get("b2", 0.0))
        if abs(b1g) > 1e-12:
            alpha_candidates.append(b2g / b1g)
    alpha_meta = meta.get("alpha_abs_gap")
    if alpha_meta is not None:
        alpha_candidates.append(float(alpha_meta))
    alpha_speed = meta.get("speed_limit_full", {}).get("alpha_theory")
    if alpha_speed is not None:
        alpha_candidates.append(float(alpha_speed))
    alpha_candidates = [a for a in alpha_candidates if np.isfinite(a)]
    if alpha_candidates:
        return max(0.0, float(alpha_candidates[0]))
    return default


def linear_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    assert np.isfinite(y).all() and np.isfinite(X).all(), "Non-finite in data/matrix"
    try:
        beta, *_ = npl.lstsq(X, y, rcond=None)
    except Exception:
        beta = npl.pinv(X) @ y
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2) + 1e-20)
    r2 = 1.0 - ss_res / ss_tot
    return beta, r2, yhat


def compute_fit(inputs: PlotInputs) -> FitResults:
    mask_fit, mask_eval = split_fit_eval(inputs)
    inv_sqrt_T = inputs.T ** -0.5
    alpha = select_alpha(inputs.meta, default=1.0)
    if inputs.target_kind == "gap":
        alpha = float(inputs.meta.get("speed_limit_full", {}).get("alpha_theory", alpha))
    C_over_T = inputs.sum_dt_over_T + alpha * inputs.sum_kappa_over_T
    rng_jitter = np.random.RandomState(0)
    C_over_T_plot = C_over_T + 1e-9 * rng_jitter.randn(*C_over_T.shape)

    def r2_from_coef(X, y, coef):
        if X.size == 0:
            return float("nan")
        yhat = X @ coef
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-20
        return 1.0 - ss_res / ss_tot

    def fit_plane(X, y):
        coef, r2, _ = linear_fit(X, y)
        return coef, r2

    use_meta_gap = (inputs.target_kind == "gap" and "speed_limit_full" in inputs.meta)

    if use_meta_gap:
        plane_meta = inputs.meta["speed_limit_full"]
        c0 = float(plane_meta["c0"])
        c1 = float(plane_meta["c1"])
        c2 = float(plane_meta["c2"])
        r2_full_fit = float(plane_meta.get("R2", float("nan")))
        r2_full_eval = float("nan")

        ab_drop_T = inputs.meta.get("ablation_drop_T", {})
        r2_noT_fit = float(ab_drop_T.get("R2", float("nan")))
        r2_noT_eval = float("nan")

        ab_drop_C = inputs.meta.get("ablation_drop_C", {})
        r2_noC_fit = float(ab_drop_C.get("R2", float("nan")))
        r2_noC_eval = float("nan")

        partial_meta = inputs.meta.get("partial_residual", {})
        a0_res = float(partial_meta.get("a0", float("nan")))
        a1_res = float(partial_meta.get("a1", float("nan")))
        r2_res = float(partial_meta.get("R2", float("nan")))

        collapse_key = inputs.meta.get("eval_collapse_gap", {})
        a0_eval_collapse = float(collapse_key.get("a0", float("nan")))
        a1_eval_collapse = float(collapse_key.get("a1", float("nan")))
        r2_eval_collapse = float(collapse_key.get("R2", float("nan")))
    else:
        # Fall back to local refit (original behavior)
        C_eval = C_over_T[mask_eval]
        y_eval = inputs.y[mask_eval]
        if C_eval.size:
            X_eval_1d = np.c_[np.ones_like(C_eval), C_eval]
            beta_eval, r2_eval_collapse, _ = linear_fit(X_eval_1d, y_eval)
            a0_eval_collapse, a1_eval_collapse = map(float, beta_eval)
        else:
            a0_eval_collapse = float("nan")
            a1_eval_collapse = float("nan")
            r2_eval_collapse = float("nan")

        ones = np.ones_like(inputs.y)
        X_all = np.c_[ones, inv_sqrt_T, C_over_T]
        X_fit = X_all[mask_fit]
        X_eval = X_all[~mask_fit]
        y_fit = inputs.y[mask_fit]
        y_eval = inputs.y[~mask_fit]

        b_full, r2_full_fit = fit_plane(X_fit if X_fit.size else X_all, y_fit if y_fit.size else inputs.y)
        c0, c1, c2 = map(float, b_full)
        r2_full_eval = r2_from_coef(X_eval, y_eval, b_full)

        X_noT = np.c_[ones, C_over_T]
        X_noT_fit = X_noT[mask_fit]
        X_noT_eval = X_noT[~mask_fit]
        b_noT, r2_noT_fit = fit_plane(X_noT_fit if X_noT_fit.size else X_noT,
                                      y_fit if y_fit.size else inputs.y)
        r2_noT_eval = r2_from_coef(X_noT_eval, y_eval, b_noT)

        X_noC = np.c_[ones, inv_sqrt_T]
        X_noC_fit = X_noC[mask_fit]
        X_noC_eval = X_noC[~mask_fit]
        b_noC, r2_noC_fit = fit_plane(X_noC_fit if X_noC_fit.size else X_noC,
                                      y_fit if y_fit.size else inputs.y)
        r2_noC_eval = r2_from_coef(X_noC_eval, y_eval, b_noC)

        y_res = inputs.y - (c0 + c1 * inv_sqrt_T)
        X_res = np.c_[np.ones_like(C_over_T), C_over_T]
        b_res, r2_res, _ = linear_fit(X_res, y_res)
        a0_res, a1_res = map(float, b_res)

        return FitResults(
            alpha_theory=float(alpha),
            c0=c0,
            c1=c1,
            c2=c2,
            r2_full=float(r2_full_fit),
            r2_noT=float(r2_noT_fit),
            r2_noC=float(r2_noC_fit),
            r2_full_eval=float(r2_full_eval),
            r2_noT_eval=float(r2_noT_eval),
            r2_noC_eval=float(r2_noC_eval),
            a0_res=a0_res,
            a1_res=a1_res,
            r2_res=float(r2_res),
            C_over_T=C_over_T,
            C_over_T_plot=C_over_T_plot,
            y_res=y_res,
            inv_sqrt_T=inv_sqrt_T,
            y=inputs.y,
            y_label=inputs.y_label,
            regime=inputs.regime,
            mask_eval=mask_eval,
            a0_eval_collapse=a0_eval_collapse,
            a1_eval_collapse=a1_eval_collapse,
            r2_eval_collapse=float(r2_eval_collapse),
        )

    # If we reach here, we used metadata for the plane and collapse numbers.
    y_res = inputs.y - (c0 + c1 * inv_sqrt_T)

    print(
        "[Speed-limit fit: meta]"
        f" alpha={alpha:.3f}"
        f"  R2_fit={r2_full_fit:.3f}"
        f"  R2_noT={r2_noT_fit:.3f}"
        f"  R2_noC={r2_noC_fit:.3f}"
        f"  R2_res={r2_res:.3f}"
    )

    return FitResults(
        alpha_theory=float(alpha),
        c0=c0,
        c1=c1,
        c2=c2,
        r2_full=float(r2_full_fit),
        r2_noT=float(r2_noT_fit),
        r2_noC=float(r2_noC_fit),
        r2_full_eval=float(r2_full_eval),
        r2_noT_eval=float(r2_noT_eval),
        r2_noC_eval=float(r2_noC_eval),
        a0_res=float(a0_res),
        a1_res=float(a1_res),
        r2_res=float(r2_res),
        C_over_T=C_over_T,
        C_over_T_plot=C_over_T_plot,
        y_res=y_res,
        inv_sqrt_T=inv_sqrt_T,
        y=inputs.y,
        y_label=inputs.y_label,
        regime=inputs.regime,
        mask_eval=mask_eval,
        a0_eval_collapse=a0_eval_collapse,
        a1_eval_collapse=a1_eval_collapse,
        r2_eval_collapse=float(r2_eval_collapse),
    )


def build_paths(results_dir: Path) -> FigurePaths:
    return FigurePaths(
        residual=results_dir / "fig_speed_limit_partial_residual",
        ablation=results_dir / "fig_speed_limit_ablation",
        plane=results_dir / "fig_speed_limit_plane_3D",
    )


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_partial_residual(fits: FitResults, path_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.3), layout='constrained')
    ax.scatter(fits.C_over_T_plot, fits.y_res, s=18, alpha=0.85, color="#4f65a3")
    xg = np.linspace(0.0, float(fits.C_over_T.max()) * 1.05, 200)
    ax.plot(
        xg,
        fits.a0_res + fits.a1_res * xg,
        color='0.25',
        lw=1.0,
        label=fr'fit ($R^2={fits.r2_res:.2f}$)',
    )
    ax.set_xlabel(r'$C_T/T$ (fixed $\alpha$)', fontsize=10)
    ax.set_ylabel(r'$y - \hat c_0 - \hat c_1 T^{-1/2}$', fontsize=10)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.35, axis='y')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.savefig(path_prefix.with_suffix(".pdf"))
    fig.savefig(path_prefix.with_suffix(".png"), dpi=600)


def plot_ablation_bars(fits: FitResults, path_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.6, 2.1), layout='constrained')
    labels = ["Full", "No $T^{-1/2}$", "No $C_T/T$"]
    r2_vals = [fits.r2_full, fits.r2_noT, fits.r2_noC]
    cols = ["#4f65a3", "#8aa29e", "#c76d5a"]
    bars = ax.bar(labels, r2_vals, color=cols, alpha=0.9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r"$R^2$", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', alpha=0.3)
    for b, r in zip(bars, r2_vals):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.03, f"{r:.2f}",
                ha='center', va='bottom', fontsize=8)
    ax.text(0.28, -0.15, fr"$\Delta R^2_{{-T^{-1/2}}} = {fits.r2_full - fits.r2_noT:.2f}$",
            transform=ax.transAxes, ha='center', va='top', fontsize=8)
    ax.text(0.72, -0.15, fr"$\Delta R^2_{{-C_T/T}} = {fits.r2_full - fits.r2_noC:.2f}$",
            transform=ax.transAxes, ha='center', va='top', fontsize=8)
    fig.savefig(path_prefix.with_suffix(".pdf"))
    fig.savefig(path_prefix.with_suffix(".png"), dpi=600)


def load_raw_scatter(raw_csv: Optional[Path], alpha: float,
                     hold_keys: Optional[set[Tuple[float, float, float]]]) -> Optional[RawScatter]:
    if raw_csv is None or not raw_csv.exists():
        return None
    df = pd.read_csv(raw_csv)
    req_candidates = [
        {"sum_dt", "sum_kappa", "T", "gen_gap_traj"},
        {"sum_dt", "sum_kappa", "T", "gen_gap_T"},
    ]
    req = next((req for req in req_candidates if req.issubset(df.columns)), None)
    if req is None:
        return None
    dt = df["sum_dt"].to_numpy(float)
    kap = df["sum_kappa"].to_numpy(float)
    T = df["T"].to_numpy(float)
    ratios = df["C_exo_ratio"].to_numpy(float) if "C_exo_ratio" in df.columns else np.zeros_like(T)
    gammas = df["gamma"].to_numpy(float) if "gamma" in df.columns else np.zeros_like(T)
    C_over_T = (dt + alpha * kap) / T
    inv_sqrt_T = T ** -0.5
    gap_col = "gen_gap_traj" if "gen_gap_traj" in df.columns else "gen_gap_T"
    gaps = df[gap_col].to_numpy(float)
    seed_idx = df["seed"] if "seed" in df.columns else pd.Series(np.zeros_like(C_over_T))
    if hold_keys:
        combos = list(zip(T.tolist(), ratios.tolist(), gammas.tolist()))
        is_hold = np.array([tuple(map(float, combo)) in hold_keys for combo in combos], dtype=bool)
    else:
        is_hold = np.zeros_like(T, dtype=bool)
    return RawScatter(C_over_T=C_over_T,
                      inv_sqrt_T=inv_sqrt_T,
                      gaps=gaps,
                      seeds=seed_idx.to_numpy(int),
                      is_hold=is_hold)


def plot_plane(fits: FitResults, path_prefix: Path, raw: Optional[RawScatter] = None) -> None:
    fig = plt.figure(figsize=(3.6, 2.8))
    ax = fig.add_subplot(111, projection='3d')

    # Fitted plane over the whole domain (coeffs from fit subset).
    xg, yg = np.meshgrid(
        np.linspace(fits.C_over_T.min(), fits.C_over_T.max(), 32),
        np.linspace(fits.inv_sqrt_T.min(), fits.inv_sqrt_T.max(), 32),
    )
    zg = fits.c0 + fits.c1 * yg + fits.c2 * xg
    ax.plot_surface(
        xg,
        yg,
        zg,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        edgecolor='none',
        shade=False,
        alpha=0.65,
        color="0.8",
    )

    # Light grid, no pane fill.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 0.1
        axis._axinfo["grid"]["color"] = "0.75"

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Fit vs held-out means.
    fit_mask = ~fits.mask_eval
    ax.scatter(
        fits.C_over_T[fit_mask],
        fits.inv_sqrt_T[fit_mask],
        fits.y[fit_mask],
        s=26,
        color="#4f65a3",
        linewidths=0.4,
        edgecolors="white",
        alpha=0.95,
        marker='o',
        label="Fit means",
        zorder=5,
    )
    if fits.mask_eval.any():
        ax.scatter(
            fits.C_over_T[fits.mask_eval],
            fits.inv_sqrt_T[fits.mask_eval],
            fits.y[fits.mask_eval],
            s=40,
            facecolors="#d95f0e",
            edgecolors="white",
            linewidths=0.5,
            alpha=0.98,
            marker='^',
            label="Held-out means",
            zorder=5,
        )

    # 1-D collapse line: held-out only, using eval parameters.
    y_collapse_line = None
    r2_collapse_to_print = float("nan")
    if fits.mask_eval.any() and np.isfinite(fits.a0_eval_collapse):
        C_eval = fits.C_over_T[fits.mask_eval]
        invT_eval = fits.inv_sqrt_T[fits.mask_eval]

        # Only span the C_T/T range of the held-out points.
        x_line = np.linspace(C_eval.min(), C_eval.max(), 200)
        # Place the line at the typical T^{-1/2} of the held-out set.
        invT_line = np.full_like(x_line, invT_eval.mean())

        # gap ≈ a0 + a1 * (C_T/T)  (held-out regression)
        y_line = fits.a0_eval_collapse + fits.a1_eval_collapse * x_line

        y_collapse_line = y_line
        collapse_line, = ax.plot(
            x_line,
            invT_line,
            y_line,
            color="0.2",
            lw=1.0,
            zorder=6,
        )

        r2_collapse_to_print = fits.r2_eval_collapse

    # Axes, ticks, view.
    ax.set_xlabel(r'$C_T/T$', labelpad=6, fontsize=8)
    ax.set_ylabel(r'$T^{-1/2}$', labelpad=6, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='z', labelsize=7)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("")
    fig.text(
        0.14,
        0.52,
        r"Risk gap $|\hat{R}_T - R_T|$",
        rotation=90,
        fontsize=8,
        color='0.25',
        ha='left',
        va='center',
    )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.92)

    x_min, x_max = 0.0, float(fits.C_over_T.max()) * 1.05
    y_min = float(fits.inv_sqrt_T.min()) * 0.95
    y_max = float(fits.inv_sqrt_T.max()) * 1.05
    xticks = nice_ticks(x_min, x_max, n=4)
    yticks = nice_ticks(y_min, y_max, n=4)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Z axis: prioritize the held-out collapse line.
    collapse_max = float(np.max(y_collapse_line)) if y_collapse_line is not None else 0.0
    z_cap_means = float(fits.y.max())
    z_cap = max(z_cap_means, collapse_max)
    z_min = 0.0
    z_max = 1.35 * max(z_cap, 1e-8)
    ax.set_zlim(z_min, z_max)
    zticks = nice_ticks(z_min, z_max, n=5)
    ax.set_zticks(zticks)

    ax.set_xticklabels([f"{t:.3f}" for t in xticks], fontsize=8)
    ax.set_yticklabels([f"{t:.3f}" for t in yticks], fontsize=8)
    ax.set_zticklabels([f"{t:.3f}" for t in zticks], fontsize=8)
    ax.set_box_aspect((1, 1, 0.65))
    ax.view_init(elev=22, azim=133)

    # Inline label for collapse line after the limits are set to avoid clipping.
    if fits.mask_eval.any() and y_collapse_line is not None:
        mid = len(y_collapse_line) // 2
        ax.text(
            x_line[mid],
            invT_line[mid],
            y_collapse_line[mid] + 0.0001,
            "Collapse fit",
            fontsize=7,
            color="0.25",
        )

    # Text: plane-fit R² (fit subset) and collapse R² (held-out).
    hud_y = 0.92
    ax.text2D(
        0.16,
        hud_y,
        fr"$R^2_\text{{plane}} = {fits.r2_full:.2f}$" "\n"
        fr"$R^2_\text{{collapse}} = {r2_collapse_to_print:.2f}$",
        transform=fig.transFigure,
        fontsize=9,
        color='0.25',
        va='top',
    )

    if fits.mask_eval.any():
        handles, labels = ax.get_legend_handles_labels()
        if y_collapse_line is not None and collapse_line in handles:
            idx = handles.index(collapse_line)
            handles.pop(idx)
            labels.pop(idx)
        ax.legend(
            handles,
            labels,
            loc='upper right',
            bbox_to_anchor=(0.95, hud_y),
            bbox_transform=fig.transFigure,
            fontsize=8,
            frameon=False,
            borderaxespad=0.0,
        )

    fig.savefig(path_prefix.with_suffix(".pdf"))
    fig.savefig(path_prefix.with_suffix(".png"), dpi=600)

def nice_ticks(vmin: float, vmax: float, n: int = 6) -> np.ndarray:
    if vmax <= vmin or not np.isfinite([vmin, vmax]).all():
        return np.array([vmin, vmax])
    raw = (vmax - vmin) / max(1, n - 1)
    exp = np.floor(np.log10(raw))
    base = raw / (10.0 ** exp)
    if base < 1.5:
        m = 1
    elif base < 3.5:
        m = 2
    elif base < 7.5:
        m = 5
    else:
        m = 10
    step = m * (10.0 ** exp)
    start = np.floor(vmin / step) * step
    ticks = np.arange(start, vmax + step * 0.5, step)
    ticks = ticks[(ticks >= vmin - 1e-9) & (ticks <= vmax + 1e-9)]
    if ticks.size < 2:
        ticks = np.array([vmin, vmax])
    return ticks


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def main(csv_path: str = "figNN_additivity_summary.csv",
         meta_path: str = "figNN_additivity_meta.json",
         raw_path: Optional[str] = None) -> None:
    inputs = load_inputs(Path(csv_path), Path(meta_path))
    fits = compute_fit(inputs)
    hold_keys: Optional[set[Tuple[float, float, float]]] = None
    if raw_path and fits.mask_eval.any():
        hold_keys = {
            (float(t), float(r), float(g))
            for t, r, g in zip(inputs.T[fits.mask_eval],
                               inputs.ratio[fits.mask_eval],
                               inputs.gamma[fits.mask_eval])
        }
    raw = load_raw_scatter(Path(raw_path), fits.alpha_theory, hold_keys) if raw_path else None
    results_dir = create_results_dir("nn_plotting")
    paths = build_paths(results_dir)
    plot_partial_residual(fits, paths.residual)
    plot_ablation_bars(fits, paths.ablation)
    plot_plane(fits, paths.plane, raw=raw)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NN plotting utility")
    parser.add_argument("--summary", type=str, default="figNN_additivity_summary.csv",
                        help="Path to figNN_additivity_summary.csv")
    parser.add_argument("--meta", type=str, default="figNN_additivity_meta.json",
                        help="Path to figNN_additivity_meta.json")
    parser.add_argument("--raw", type=str, default=None,
                        help="Optional path to figNN_additivity_raw.csv for per-seed scatter")
    args = parser.parse_args()
    main(args.summary, args.meta, args.raw)
