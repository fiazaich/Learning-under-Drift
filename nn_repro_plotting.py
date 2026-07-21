import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PALETTE = {"train": "#1f77b4", "holdout": "#ff7f0e"}
MARKERS = {"train": "o", "holdout": "x"}
SCATTER_KW = dict(s=26, linewidths=1.0, alpha=0.9)

try:
    import scienceplots

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


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-20
    return 1.0 - ss_res / ss_tot


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.c_[np.ones_like(x), x]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, a1 = float(coef[0]), float(coef[1])
    yhat = a0 + a1 * x
    return a0, a1, r2_score(y, yhat)


def fit_plane(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    return beta, r2_score(y, yhat)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(method="average").to_numpy()
    if np.std(x_rank) <= 1e-12 or np.std(y_rank) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _target_spec(target: str) -> Tuple[str, str, str]:
    if target == "delta_rep":
        return "delta_rep", "mean_delta_rep", r"$\Delta_T^{\mathrm{preq}} = |\widehat R_T - R_T^+|$"
    if target == "delta_sam":
        return "delta_sam", "mean_delta_sam", r"$\Delta_T^{\mathrm{sam}} = |\widehat R_T - R_T|$"
    if target == "V_T":
        return "V_T", "mean_V_T", r"$V_T = \frac{1}{T}\sum_t |R(\theta_{t+1},f_t)-R(\theta_t,f_t)|$"
    if target == "err_T":
        return "err_T", "mean_err_T", r"Terminal mismatch $\mathbb{E}|f_T(x)-g_{\theta_T}(x)|$"
    if target == "legacy_gap":
        return "gen_gap_traj_legacy", "mean_gen_gap_traj_legacy", r"Legacy proxy gap"
    raise ValueError("target must be one of: delta_rep, delta_sam, V_T, err_T, legacy_gap")


def pick_holdout_T(T_values: np.ndarray, user_T_hold: Optional[int]) -> int:
    uniq = sorted(set(int(t) for t in T_values))
    if user_T_hold is not None:
        if user_T_hold not in uniq:
            raise ValueError(f"--T-hold={user_T_hold} not in available Ts: {uniq}")
        return int(user_T_hold)
    return int(uniq[-1])


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

    train = df[~is_hold].copy()
    test = df[is_hold].copy()

    x_train = train["C_over_T"].to_numpy()
    t_train = train["inv_sqrt_T"].to_numpy()
    y_train = train[y_sum_col].to_numpy()

    X_train = np.c_[np.ones_like(y_train), t_train, x_train]
    beta, _ = fit_plane(X_train, y_train)
    c0, c1, c2 = map(float, beta)

    y_train_res = y_train - c1 * t_train
    a0, a1, r2_train_line = fit_line(x_train, y_train_res)

    x_test = test["C_over_T"].to_numpy()
    t_test = test["inv_sqrt_T"].to_numpy()
    y_test = test[y_sum_col].to_numpy()
    y_test_res = y_test - c1 * t_test

    yhat_train_res = a0 + a1 * x_train
    yhat_test_res = a0 + a1 * x_test
    r2_train = r2_score(y_train_res, yhat_train_res) if len(y_train_res) > 1 else float("nan")
    r2_test = r2_score(y_test_res, yhat_test_res) if len(y_test_res) > 1 else float("nan")

    plt.figure(figsize=(4.0, 2.9))
    plt.scatter(
        x_train,
        y_train_res,
        color=PALETTE["train"],
        marker=MARKERS["train"],
        label=r"train ($T\ne T_{\mathrm{hold}}$)",
        **SCATTER_KW,
    )
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


def budget_proxy_fit(summary: pd.DataFrame, y_sum_col: str, alpha: float, T_hold: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
    _ensure_cols(summary, ["T", "mean_sum_dt", "mean_sum_kappa", y_sum_col])
    df = add_budget_columns_summary(summary, alpha=alpha)
    T = df["T"].astype(float).to_numpy()
    train = df["T"].astype(int).to_numpy() != int(T_hold)
    y = df[y_sum_col].to_numpy(dtype=float)
    X = np.c_[np.ones(len(df)), T ** (-0.5), df["C_over_T"].to_numpy(dtype=float)]
    beta, r2_train = fit_plane(X[train], y[train])
    pred = X @ beta
    df["budget_proxy_pred"] = pred
    df["budget_proxy_residual"] = y - pred
    df["budget_proxy_ratio"] = y / np.maximum(np.abs(pred), 1e-12)
    df["is_holdout"] = ~train
    r2_hold = r2_score(y[~train], pred[~train]) if np.sum(~train) > 1 else float("nan")
    diag = {
        "proxy_b0": float(beta[0]),
        "proxy_b_sampling": float(beta[1]),
        "proxy_b_budget": float(beta[2]),
        "proxy_R2_train": float(r2_train),
        "proxy_R2_hold": float(r2_hold),
        "proxy_spearman_all": spearman_corr(df["C_over_T"].to_numpy(), y),
        "proxy_spearman_train": spearman_corr(df.loc[train, "C_over_T"].to_numpy(), y[train]),
        "proxy_spearman_hold": spearman_corr(df.loc[~train, "C_over_T"].to_numpy(), y[~train]),
    }
    return df, diag


def plot_budget_proxy_diagnostic(proxy_df: pd.DataFrame, plotdir: Path, y_sum_col: str, ylab: str, diag: Dict[str, float]) -> None:
    train = ~proxy_df["is_holdout"].to_numpy(dtype=bool)
    y = proxy_df[y_sum_col].to_numpy(dtype=float)
    pred = proxy_df["budget_proxy_pred"].to_numpy(dtype=float)
    resid = proxy_df["budget_proxy_residual"].to_numpy(dtype=float)
    x = proxy_df["C_over_T"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    for mask, label, color, marker in [
        (train, r"train ($T\ne T_{\mathrm{hold}}$)", PALETTE["train"], MARKERS["train"]),
        (~train, "holdout", PALETTE["holdout"], MARKERS["holdout"]),
    ]:
        axes[0].scatter(pred[mask], y[mask], color=color, marker=marker, label=label, **SCATTER_KW)
        axes[1].scatter(x[mask], resid[mask], color=color, marker=marker, label=label, **SCATTER_KW)

    lo = float(min(np.min(pred), np.min(y)))
    hi = float(max(np.max(pred), np.max(y)))
    pad = 0.03 * (hi - lo + 1e-12)
    xs = np.linspace(lo - pad, hi + pad, 200)
    axes[0].plot(xs, xs, color="0.25", linewidth=1.1)
    display_ylab = r"$V_T$" if y_sum_col == "mean_V_T" else ylab
    axes[0].set_xlabel("Fitted budget proxy")
    axes[0].set_ylabel(r"Observed " + display_ylab)
    axes[0].text(
        0.02,
        0.98,
        rf"$R^2_{{train}}={diag['proxy_R2_train']:.3f}$, $R^2_{{hold}}={diag['proxy_R2_hold']:.3f}$",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
    )

    axes[1].axhline(0.0, color="0.25", linewidth=1.0)
    axes[1].set_xlabel(r"$C_T/T = (\sum d_t + \alpha \sum \kappa_t)/T$")
    axes[1].set_ylabel("Residual")
    axes[1].text(
        0.02,
        0.98,
        rf"Spearman $\rho$={diag['proxy_spearman_all']:.3f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
    )

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
    axes[0].legend(loc="best", frameon=True)
    _savefig_both(plotdir / f"figNN_budget_proxy_diagnostic_{y_sum_col}.pdf")


def write_structural_diagnostics(raw: pd.DataFrame, summary: pd.DataFrame, plotdir: Path, y_sum_col: str, alpha: float, T_hold: int) -> None:
    y_raw_col, _, ylab = _target_spec("V_T" if y_sum_col == "mean_V_T" else "delta_rep")
    proxy_df, diag = budget_proxy_fit(summary, y_sum_col=y_sum_col, alpha=alpha, T_hold=T_hold)
    plot_budget_proxy_diagnostic(proxy_df, plotdir, y_sum_col=y_sum_col, ylab=ylab, diag=diag)

    proxy_cols = [
        "T",
        "C_exo_ratio",
        "gamma",
        y_sum_col,
        "C_over_T",
        "budget_proxy_pred",
        "budget_proxy_residual",
        "budget_proxy_ratio",
        "is_holdout",
    ]
    proxy_df[proxy_cols].to_csv(plotdir / f"tableNN_budget_proxy_diagnostics_{y_sum_col}.csv", index=False)

    lines = [
        "NN structural mechanism diagnostics",
        f"target = {y_sum_col}",
        f"T_hold = {T_hold}",
        f"alpha_fit_train = {alpha:.10g}",
        "",
        "Budget proxy fit on training horizons:",
        f"  y = b0 + b_sampling*T^(-1/2) + b_budget*C_T/T",
        f"  b0 = {diag['proxy_b0']:.10g}",
        f"  b_sampling = {diag['proxy_b_sampling']:.10g}",
        f"  b_budget = {diag['proxy_b_budget']:.10g}",
        f"  R2_train = {diag['proxy_R2_train']:.6g}",
        f"  R2_hold = {diag['proxy_R2_hold']:.6g}",
        "",
        "Rank association between target and C_T/T:",
        f"  Spearman_all = {diag['proxy_spearman_all']:.6g}",
        f"  Spearman_train = {diag['proxy_spearman_train']:.6g}",
        f"  Spearman_hold = {diag['proxy_spearman_hold']:.6g}",
    ]

    if {"delta_rep", "delta_sam", "V_T"}.issubset(raw.columns):
        slack = raw["delta_sam"].to_numpy(dtype=float) + raw["V_T"].to_numpy(dtype=float) - raw["delta_rep"].to_numpy(dtype=float)
        q = pd.Series(slack).quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
        lines.extend(
            [
                "",
                "Exact per-seed decomposition check: delta_rep <= delta_sam + V_T",
                f"  min_slack = {float(np.min(slack)):.10g}",
                f"  median_slack = {float(np.median(slack)):.10g}",
                f"  max_slack = {float(np.max(slack)):.10g}",
                f"  fraction_slack_below_1e-4 = {float(np.mean(slack < 1e-4)):.6g}",
                "  slack_quantiles:",
            ]
        )
        for idx, value in q.items():
            lines.append(f"    q{idx:g} = {float(value):.10g}")

    if {"mean_delta_rep", "mean_delta_sam", "mean_V_T"}.issubset(summary.columns):
        summary_slack = summary["mean_delta_sam"].to_numpy(dtype=float) + summary["mean_V_T"].to_numpy(dtype=float) - summary["mean_delta_rep"].to_numpy(dtype=float)
        lines.extend(
            [
                "",
                "Summary-level mean diagnostic:",
                f"  min(mean_delta_sam + mean_V_T - mean_delta_rep) = {float(np.min(summary_slack)):.10g}",
                f"  corr(mean_delta_rep, mean_delta_sam) = {float(summary['mean_delta_rep'].corr(summary['mean_delta_sam'])):.6g}",
                f"  corr(mean_delta_rep, mean_delta_sam + mean_V_T) = {float(summary['mean_delta_rep'].corr(summary['mean_delta_sam'] + summary['mean_V_T'])):.6g}",
            ]
        )

    (plotdir / f"diagnosticsNN_structural_{y_sum_col}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_3d_single(raw: pd.DataFrame, outdir: Path, y_raw_col: str, ylab: str, T_hold: int, alpha: float):
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


def main():
    ap = argparse.ArgumentParser(description="Plot balanced NN results with held-out T protocol.")
    ap.add_argument("--outdir", required=True, type=str)
    ap.add_argument("--plotdir", default=None, type=str, help="Directory for analysis outputs. Default: write next to CSVs in --outdir.")
    ap.add_argument("--target", default="delta_rep", choices=["delta_rep", "delta_sam", "V_T", "err_T", "legacy_gap"])
    ap.add_argument("--T-hold", type=int, default=None, help="Horizon to hold out. Default: largest T.")
    ap.add_argument("--collapse-mode", default="byT", choices=["byT", "residual"])
    ap.add_argument("--do-3d", action="store_true", help="Also write a single 3D plot with heldout markers.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    plotdir = Path(args.plotdir) if args.plotdir is not None else outdir
    plotdir.mkdir(parents=True, exist_ok=True)
    raw = _load_csv(outdir, "figNN_additivity_raw.csv")
    summary = _load_csv(outdir, "figNN_additivity_summary.csv")

    y_raw_col, y_sum_col, ylab = _target_spec(args.target)
    T_hold = pick_holdout_T(summary["T"].to_numpy(), args.T_hold)

    alpha, diag = fit_alpha_holdout(summary, y_sum_col=y_sum_col, T_hold=T_hold)

    plot_ablation(summary, plotdir, y_sum_col=y_sum_col, ylab=ylab, T_hold=T_hold)
    plot_collapse(summary, plotdir, y_sum_col=y_sum_col, ylab=ylab, alpha=alpha, T_hold=T_hold, mode=args.collapse_mode)
    plot_obs_vs_pred(summary, plotdir, y_sum_col=y_sum_col, ylab=ylab, T_hold=T_hold)
    write_structural_diagnostics(raw, summary, plotdir, y_sum_col=y_sum_col, alpha=alpha, T_hold=T_hold)

    if args.do_3d:
        plot_3d_single(raw, plotdir, y_raw_col=y_raw_col, ylab=ylab, T_hold=T_hold, alpha=alpha)

    print("Held-out protocol diagnostics:")
    print(f"  T_hold = {T_hold}")
    print(f"  alpha (fit on T != hold) = {diag['alpha']:.6g}")
    print(f"  plane-fit R^2 on train    = {diag['R2_train_plane']:.4f}")
    print(f"  b_s={diag['b_s']:.6g}, b1={diag['b1']:.6g}, b2={diag['b2']:.6g}")
    print(f"  output directory = {plotdir}")
    print("Wrote (PDF+PNG):")
    print(f"  figNN_ablation_{y_sum_col}.*")
    print(f"  figNN_collapse_{args.collapse_mode}_{y_sum_col}.*")
    print(f"  figNN_obs_vs_pred_{y_sum_col}.*")
    print(f"  figNN_budget_proxy_diagnostic_{y_sum_col}.*")
    print(f"  diagnosticsNN_structural_{y_sum_col}.txt")
    print(f"  tableNN_budget_proxy_diagnostics_{y_sum_col}.csv")
    if args.do_3d:
        print(f"  figNN_additivity_3d_{y_raw_col}_heldout.*")


if __name__ == "__main__":
    main()
