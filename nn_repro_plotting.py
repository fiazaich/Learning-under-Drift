"""Plot NN drift results with held-out T protocol."""
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

PALETTE = {"train": "#1f77b4", "holdout": "#ff7f0e"}
MARKERS = {"train": "o", "holdout": "x"}
SCATTER_KW = dict(s=26, linewidths=1.0, alpha=0.9)

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "ieee"])
except Exception:
    pass

mpl.rcParams.update(
    {
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
    }
)

# ----------------------------- IO helpers -----------------------------
def _load_csv(outdir: Path, name: str) -> pd.DataFrame:
    p = outdir / name
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return pd.read_csv(p)


def _load_meta(outdir: Path) -> Dict:
    p = outdir / "figNN_additivity_meta.json"
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return json.load(f)


def _ensure_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Available: {list(df.columns)}")


def _savefig_both(path_pdf: Path, dpi_png=300):
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path_pdf, dpi=250, bbox_inches="tight")
    plt.savefig(path_pdf.with_suffix(".png"), dpi=dpi_png, bbox_inches="tight")
    plt.close()


# ----------------------------- math helpers -----------------------------
def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-20
    return 1.0 - ss_res / ss_tot


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit y ≈ a0 + a1 x, return (a0,a1,R2).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.c_[np.ones_like(x), x]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, a1 = float(coef[0]), float(coef[1])
    yhat = a0 + a1 * x
    return a0, a1, r2_score(y, yhat)


def fit_plane(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit y ≈ X @ beta (X already includes intercept column if desired).
    Return (beta, R2).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    return beta, r2_score(y, yhat)


# ----------------------------- target mapping -----------------------------
def _target_spec(target: str) -> Tuple[str, str, str]:
    """
    Returns (raw_column, summary_column, y_label)
    """
    if target == "delta_rep":
        return "delta_rep", "mean_delta_rep", r"$\Delta_T^{\mathrm{rep}} = |\widehat R_T - R_T^+|$"
    if target == "V_T":
        return "V_T", "mean_V_T", r"$V_T = \frac{1}{T}\sum_t |R(\theta_{t+1},f_t)-R(\theta_t,f_t)|$"
    if target == "err_T":
        return "err_T", "mean_err_T", r"Terminal mismatch $\mathbb{E}|f_T(x)-g_{\theta_T}(x)|$"
    if target == "legacy_gap":
        return "gen_gap_traj_legacy", "mean_gen_gap_traj_legacy", r"Legacy proxy gap"
    raise ValueError("target must be one of: delta_rep, V_T, err_T, legacy_gap")


def pick_holdout_T(T_values: np.ndarray, user_T_hold: Optional[int]) -> int:
    uniq = sorted(set(int(t) for t in T_values))
    if user_T_hold is not None:
        if user_T_hold not in uniq:
            raise ValueError(f"--T-hold={user_T_hold} not in available Ts: {uniq}")
        return int(user_T_hold)
    return int(uniq[-1])  # default: largest T


# ----------------------------- core derived columns -----------------------------
def add_budget_columns_summary(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    df = df.copy()
    _ensure_cols(df, ["T", "mean_sum_dt", "mean_sum_kappa"])
    T = df["T"].astype(float).to_numpy()
    df["C_over_T"] = (df["mean_sum_dt"] + alpha * df["mean_sum_kappa"]) / T
    df["inv_sqrt_T"] = T ** (-0.5)
    return df


def add_budget_columns_raw(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    df = df.copy()
    _ensure_cols(df, ["T", "sum_dt", "sum_kappa"])
    T = df["T"].astype(float).to_numpy()
    df["C_over_T"] = (df["sum_dt"] + alpha * df["sum_kappa"]) / T
    df["inv_sqrt_T"] = T ** (-0.5)
    return df


# ----------------------------- plots -----------------------------
def plot_ablation(summary: pd.DataFrame, outdir: Path, y_sum_col: str, ylab: str, T_hold: int):
    _ensure_cols(summary, ["T", "C_exo_ratio", "gamma", y_sum_col])
    Ts = sorted(summary["T"].unique())
    gammas = sorted(summary["gamma"].unique())

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(gammas))]
    markers = ["o", "s", "^", "d", "v", "<", ">", "P", "X", "*"]

    nT = len(Ts)
    fig_h = 2.0 + 1.25 * nT
    fig = plt.figure(figsize=(4.6, fig_h))

    for i, T in enumerate(Ts, start=1):
        ax = plt.subplot(nT, 1, i)
        dfT = summary[summary["T"] == T].copy()

        for idx, g in enumerate(gammas):
            dfg = dfT[dfT["gamma"] == g].sort_values("C_exo_ratio")
            ax.plot(
                dfg["C_exo_ratio"].to_numpy(),
                dfg[y_sum_col].to_numpy(),
                marker=markers[idx % len(markers)],
                color=colors[idx],
                linewidth=1.2,
                label=rf"$\gamma={g:g}$",
            )

        ax.set_xscale("log")
        title = rf"$T={int(T)}$"
        if int(T) == int(T_hold):
            title += " (holdout)"
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.yaxis.set_major_locator(MaxNLocator(4))

        if i == nT:
            ax.set_xlabel(r"Exogenous budget ratio $C_{\mathrm{exo}}/T$")
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                loc="upper left",
                frameon=True,
                ncol=min(len(gammas), 3),
            )

    _savefig_both(outdir / f"figNN_ablation_{y_sum_col}.pdf")


def plot_collapse(summary: pd.DataFrame, outdir: Path, y_sum_col: str, ylab: str, alpha: float, T_hold: int, mode: str):
    """
    mode:
      - "byT": stacked panels per T (each with its own within-T R^2)
      - "residual": single plot of y - c1*T^{-1/2} vs C/T, using c1 fit on train
    """
    _ensure_cols(summary, ["T", "mean_sum_dt", "mean_sum_kappa", y_sum_col])

    df = add_budget_columns_summary(summary, alpha=alpha)
    T = df["T"].astype(int).to_numpy()
    is_hold = (T == int(T_hold))

    if mode == "byT":
        Ts = sorted(df["T"].unique())
        nT = len(Ts)
        fig_h = 2.0 + 1.25 * nT
        plt.figure(figsize=(4.6, fig_h))

        for i, Tv in enumerate(Ts, start=1):
            ax = plt.subplot(nT, 1, i)
            dfi = df[df["T"] == Tv].copy()
            x = dfi["C_over_T"].to_numpy()
            y = dfi[y_sum_col].to_numpy()

            a0, a1, r2i = fit_line(x, y)
            xs = np.linspace(x.min(), x.max(), 200)

            # points
            ax.scatter(x, y, color="#4f65a3", s=28, linewidths=0.6, alpha=0.9)
            ax.plot(xs, a0 + a1 * xs, linewidth=1.2)

            title = rf"$T={int(Tv)}$"
            if int(Tv) == int(T_hold):
                title += " (holdout)"
            ax.set_title(title)
            ax.set_ylabel(ylab)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_locator(MaxNLocator(4))

            ax.text(
                0.02,
                0.92,
                rf"$\alpha={alpha:.3g}$  |  $R^2={r2i:.3f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
            )

            if i == nT:
                ax.set_xlabel(r"$C_T/T = (\sum d_t + \alpha \sum \kappa_t)/T$")

        _savefig_both(outdir / f"figNN_collapse_byT_{y_sum_col}.pdf")
        return

    if mode != "residual":
        raise ValueError("collapse mode must be 'byT' or 'residual'")

    # residualized collapse: fit c1 on train: y ≈ c0 + c1*T^{-1/2} + c2*(C/T)
    train = df[~is_hold].copy()
    test = df[is_hold].copy()

    x_train = train["C_over_T"].to_numpy()
    t_train = train["inv_sqrt_T"].to_numpy()
    y_train = train[y_sum_col].to_numpy()

    X_train = np.c_[np.ones_like(y_train), t_train, x_train]
    beta, _ = fit_plane(X_train, y_train)
    c0, c1, c2 = map(float, beta)

    # residualize
    y_train_res = y_train - c1 * t_train
    a0, a1, r2_train_line = fit_line(x_train, y_train_res)

    # test residuals + test R2 w.r.t. the same line
    x_test = test["C_over_T"].to_numpy()
    t_test = test["inv_sqrt_T"].to_numpy()
    y_test = test[y_sum_col].to_numpy()
    y_test_res = y_test - c1 * t_test

    yhat_train_res = a0 + a1 * x_train
    yhat_test_res = a0 + a1 * x_test
    r2_train = r2_score(y_train_res, yhat_train_res) if len(y_train_res) > 1 else float("nan")
    r2_test = r2_score(y_test_res, yhat_test_res) if len(y_test_res) > 1 else float("nan")

    plt.figure(figsize=(4.0, 2.9))
    # train points
    plt.scatter(
        x_train,
        y_train_res,
        color=PALETTE["train"],
        marker=MARKERS["train"],
        label=r"train ($T\ne T_{\mathrm{hold}}$)",
        **SCATTER_KW,
    )
    # holdout points 
    plt.scatter(
        x_test,
        y_test_res,
        color=PALETTE["holdout"],
        marker=MARKERS["holdout"],
        s=SCATTER_KW["s"] * 1.1,
        linewidths=1.2,
        label=f"holdout (T={T_hold})",
        alpha=0.9,
    )

    xs = np.linspace(df["C_over_T"].min(), df["C_over_T"].max(), 250)
    plt.plot(xs, a0 + a1 * xs, linewidth=1.4, color="0.25")

    plt.xlabel(r"$C_T/T$")
    plt.ylabel(ylab + r" (residualized)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(5))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))
    plt.legend(loc="best", frameon=True)

    plt.text(
        0.02,
        0.98,
        rf"$\alpha={alpha:.3g}$,  $c_1={c1:.3g}$" "\n"
        rf"$R^2_{{train}}={r2_train:.3f}$,  $R^2_{{hold}}={r2_test:.3f}$",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
    )

    _savefig_both(outdir / f"figNN_collapse_residual_{y_sum_col}.pdf")


def plot_obs_vs_pred(summary: pd.DataFrame, outdir: Path, y_sum_col: str, ylab: str, T_hold: int):
    """
    Single-figure additivity diagnostic:
      Fit plane y ≈ b0 + b_s*T^{-1/2} + b1*(sum_dt/T) + b2*(sum_kappa/T)
      on train (T != hold), then plot observed vs predicted for train+holdout.
    """
    _ensure_cols(summary, ["T", "mean_sum_dt", "mean_sum_kappa", y_sum_col])

    df = summary.copy()
    T = df["T"].astype(float).to_numpy()
    inv_sqrt_T = T ** (-0.5)
    x1 = (df["mean_sum_dt"] / T).to_numpy()
    x2 = (df["mean_sum_kappa"] / T).to_numpy()
    y = df[y_sum_col].to_numpy()

    T_int = df["T"].astype(int).to_numpy()
    is_hold = (T_int == int(T_hold))
    train = ~is_hold

    X = np.c_[np.ones_like(y), inv_sqrt_T, x1, x2]
    beta, _ = fit_plane(X[train], y[train])
    yhat_train = X[train] @ beta
    yhat_hold = X[is_hold] @ beta

    r2_train = r2_score(y[train], yhat_train) if np.sum(train) > 1 else float("nan")
    r2_hold = r2_score(y[is_hold], yhat_hold) if np.sum(is_hold) > 1 else float("nan")

    plt.figure(figsize=(3.6, 3.2))
    obs_pred_kw = dict(SCATTER_KW)
    obs_pred_kw["s"] = 20
    obs_pred_kw["linewidths"] = 0.9
    plt.scatter(
        yhat_train,
        y[train],
        color=PALETTE["train"],
        marker=MARKERS["train"],
        label=r"train ($T\ne T_{\mathrm{hold}}$)",
        **obs_pred_kw,
    )
    plt.scatter(
        yhat_hold,
        y[is_hold],
        color=PALETTE["holdout"],
        marker=MARKERS["holdout"],
        label=f"holdout (T={T_hold})",
        **obs_pred_kw,
    )

    # y=x
    all_yhat = np.concatenate([yhat_train, yhat_hold]) if np.sum(is_hold) else yhat_train
    lo = float(np.min(all_yhat))
    hi = float(np.max(all_yhat))
    pad = 0.03 * (hi - lo + 1e-12)
    xs = np.linspace(lo - pad, hi + pad, 200)
    plt.plot(xs, xs, linewidth=1.1, color="0.25", zorder=1.5)

    plt.xlabel(r"Predicted (plane)")
    plt.ylabel(r"Observed " + ylab)
    plt.gca().xaxis.set_major_locator(MaxNLocator(5))
    plt.gca().yaxis.set_major_locator(MaxNLocator(5))
    plt.legend(loc="best", frameon=True)

    b0, bs, b1, b2 = map(float, beta)
    plt.text(
        0.02,
        0.98,
        rf"$R^2_{{train}}={r2_train:.3f}$, $R^2_{{hold}}={r2_hold:.3f}$" "\n"
        rf"$b_s={bs:.3g},\; b_1={b1:.3g},\; b_2={b2:.3g}$",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
    )

    _savefig_both(outdir / f"figNN_obs_vs_pred_{y_sum_col}.pdf")


def fit_alpha_holdout(summary: pd.DataFrame, y_sum_col: str, T_hold: int) -> Tuple[float, Dict[str, float]]:
    """
    Fit alpha = b2/b1 for the plane:
      y ≈ b0 + b_s*T^{-1/2} + b1*(sum_dt/T) + b2*(sum_kappa/T)
    on train (T != hold). Returns (alpha, diagnostics).
    """
    _ensure_cols(summary, ["T", "mean_sum_dt", "mean_sum_kappa", y_sum_col])

    df = summary.copy()
    T = df["T"].astype(float).to_numpy()
    inv_sqrt_T = T ** (-0.5)
    x1 = (df["mean_sum_dt"] / T).to_numpy()
    x2 = (df["mean_sum_kappa"] / T).to_numpy()
    y = df[y_sum_col].to_numpy()

    T_int = df["T"].astype(int).to_numpy()
    train = (T_int != int(T_hold))

    X = np.c_[np.ones_like(y), inv_sqrt_T, x1, x2]
    beta, r2_train = fit_plane(X[train], y[train])
    b0, bs, b1, b2 = map(float, beta)

    alpha = 1.0
    if abs(b1) > 1e-12:
        alpha = max(0.0, b2 / b1)

    diagnostics = {
        "b0": b0,
        "b_s": bs,
        "b1": b1,
        "b2": b2,
        "alpha": float(alpha),
        "R2_train_plane": float(r2_train),
    }
    return float(alpha), diagnostics


def plot_3d_single(raw: pd.DataFrame, outdir: Path, y_raw_col: str, ylab: str, T_hold: int, alpha: float):
    """
    Optional: single 3D plot (still busy). We DO NOT color by T.
    Color by gamma, marker by train/holdout.
    """
    _ensure_cols(raw, ["T", "gamma", "sum_dt", "sum_kappa", y_raw_col])
    df = add_budget_columns_raw(raw, alpha=alpha)

    T = df["T"].astype(int).to_numpy()
    is_hold = (T == int(T_hold))

    x1 = (df["sum_dt"] / df["T"]).to_numpy()
    x2 = (df["sum_kappa"] / df["T"]).to_numpy()
    z = df[y_raw_col].to_numpy()
    g = df["gamma"].to_numpy()
    ug = np.unique(g)
    g_idx = np.searchsorted(ug, g)

    fig = plt.figure(figsize=(4.6, 3.6))
    ax = fig.add_subplot(111, projection="3d")  

    ax.scatter(x1[~is_hold], x2[~is_hold], z[~is_hold], c=g_idx[~is_hold], s=18, depthshade=True, alpha=0.4)
    ax.scatter(
        x1[is_hold],
        x2[is_hold],
        z[is_hold],
        c=g_idx[is_hold],
        s=26,
        marker=MARKERS["holdout"],
        linewidths=1.2,
        depthshade=True,
    )

    ax.set_xlabel(r"$\sum_t d_t/T$")
    ax.set_ylabel(r"$\sum_t \kappa_t/T$")
    ax.set_zlabel(ylab)
    ax.set_title(rf"3D (holdout T={T_hold} marked)")

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=max(1, len(ug) - 1))),
        ax=ax,
        pad=0.08,
        shrink=0.7,
    )
    cbar.set_ticks(np.arange(len(ug)))
    cbar.set_ticklabels([f"{val:g}" for val in ug])
    cbar.set_label(r"$\gamma$")

    ax.view_init(elev=22, azim=-55)
    _savefig_both(outdir / f"figNN_additivity_3d_{y_raw_col}_heldout.pdf")


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot balanced NN results with held-out T protocol.")
    ap.add_argument("--outdir", required=True, type=str)
    ap.add_argument("--target", default="delta_rep", choices=["delta_rep", "V_T", "err_T", "legacy_gap"])
    ap.add_argument("--T-hold", type=int, default=None, help="Horizon to hold out. Default: largest T.")
    ap.add_argument("--collapse-mode", default="byT", choices=["byT", "residual"])
    ap.add_argument("--do-3d", action="store_true", help="Also write a single 3D plot with heldout markers.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    raw = _load_csv(outdir, "figNN_additivity_raw.csv")
    summary = _load_csv(outdir, "figNN_additivity_summary.csv")

    y_raw_col, y_sum_col, ylab = _target_spec(args.target)
    T_hold = pick_holdout_T(summary["T"].to_numpy(), args.T_hold)

    # 1) Fit alpha on train Ts
    alpha, diag = fit_alpha_holdout(summary, y_sum_col=y_sum_col, T_hold=T_hold)

    # 2) Plots
    plot_ablation(summary, outdir, y_sum_col=y_sum_col, ylab=ylab, T_hold=T_hold)
    plot_collapse(summary, outdir, y_sum_col=y_sum_col, ylab=ylab, alpha=alpha, T_hold=T_hold, mode=args.collapse_mode)
    plot_obs_vs_pred(summary, outdir, y_sum_col=y_sum_col, ylab=ylab, T_hold=T_hold)

    if args.do_3d:
        plot_3d_single(raw, outdir, y_raw_col=y_raw_col, ylab=ylab, T_hold=T_hold, alpha=alpha)

    print("Held-out protocol diagnostics:")
    print(f"  T_hold = {T_hold}")
    print(f"  alpha (fit on T != hold) = {diag['alpha']:.6g}")
    print(f"  plane-fit R^2 on train    = {diag['R2_train_plane']:.4f}")
    print(f"  b_s={diag['b_s']:.6g}, b1={diag['b1']:.6g}, b2={diag['b2']:.6g}")
    print("Wrote (PDF+PNG):")
    print(f"  figNN_ablation_{y_sum_col}.*")
    print(f"  figNN_collapse_{args.collapse_mode}_{y_sum_col}.*")
    print(f"  figNN_obs_vs_pred_{y_sum_col}.*")
    if args.do_3d:
        print(f"  figNN_additivity_3d_{y_raw_col}_heldout.*")


if __name__ == "__main__":
    main()
