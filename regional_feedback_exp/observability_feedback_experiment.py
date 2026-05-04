from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from dataclasses import dataclass

_MPLCONFIGDIR = Path("regional_feedback_outputs") / "matplotlib_cache"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Config
# -----------------------------

DATA_PATH = "US_Regional_Sales_Data.csv"
OUTPUT_DIR = Path("regional_feedback_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Unit Price"
QUANTITY_COL = "Order Quantity"
COST_COL = "Unit Cost"
GROUP_COL = "Sales Channel"

N_ROUNDS = 20
N_SEEDS = 10
TEST_SIZE = 0.2
MU_GRID = [0.00, 0.05, 0.10, 0.20]
RIDGE_ALPHA = 1.0
RIDGE_ALPHA_GRID = [0.1, 1.0, 10.0, 100.0]

RHO = 0.5
B_C = 0.4
B_Q = 0.3

NOISE_SCALE_Q = 0.02
NOISE_SCALE_C = 0.02
NOISE_SCALE_Y = 0.02

COARSE_SCORE_BINS = 5
TASK_SCORE_BINS = 4
TASK_QTY_BINS = 4
PSEUDOCOUNT = 1e-6
MIN_POSITIVE_VALUE = 1e-3
EPS = 1e-8


# -----------------------------
# Utilities
# -----------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fisher_rao_categorical(p: np.ndarray, q: np.ndarray) -> float:
    inner = float(np.sum(np.sqrt(p * q)))
    inner = float(np.clip(inner, -1.0, 1.0))
    return 2.0 * math.acos(inner)


def compute_quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    # Use unique quantiles; if duplicates collapse, bins may be fewer than requested.
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, q)
    edges = np.unique(edges)
    if len(edges) < 2:
        v = float(np.mean(values))
        edges = np.array([v - 1.0, v + 1.0], dtype=float)
    return edges


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # np.digitize returns 1..len(edges); subtract 1 for 0-indexed bins.
    # Keep final edge inclusive by clipping.
    bins = np.digitize(values, edges[1:-1], right=False)
    return np.asarray(bins, dtype=int)


def empirical_categorical_distribution(state_ids: np.ndarray, n_states: int, pseudocount: float) -> np.ndarray:
    counts = np.bincount(state_ids, minlength=n_states).astype(float)
    counts += pseudocount
    probs = counts / counts.sum()
    return probs


def make_group_indicator(series: pd.Series) -> np.ndarray:
    mode_value = series.mode(dropna=True)
    ref = mode_value.iloc[0] if len(mode_value) else series.dropna().iloc[0]
    return (series.astype(str).to_numpy() != str(ref)).astype(int)


# -----------------------------
# Data prep
# -----------------------------


def load_and_prepare_base_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Some CSV exports store numeric values as text with thousands separators,
    # e.g. "1,963.10". Convert only columns that are fully numeric after cleanup.
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            cleaned = df[col].astype("string").str.replace(",", "", regex=False).str.strip()
            numeric = pd.to_numeric(cleaned, errors="coerce")
            non_missing = df[col].notna()
            if non_missing.any() and numeric[non_missing].notna().all():
                df[col] = numeric

    # Drop obvious date / ID-like columns if present.
    drop_candidates = []
    for col in df.columns:
        col_low = col.lower()
        if col_low == TARGET_COL.lower():
            continue
        if any(tok in col_low for tok in ["id", "ordernumber", "ordernumber", "invoice", "transaction"]):
            drop_candidates.append(col)
    # Keep WarehouseCode even though it looks identifier-like; it can carry context.
    drop_candidates = [c for c in drop_candidates if c.lower() != "warehousecode"]

    df = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore").copy()

    # Parse dates into simple features if present, then drop raw date columns.
    for col in list(df.columns):
        if "date" in col.lower():
            try:
                dt = pd.to_datetime(df[col], format="%d/%m/%y", errors="coerce")
                if dt.notna().any():
                    df[f"{col}_year"] = dt.dt.year
                    df[f"{col}_month"] = dt.dt.month
                    df[f"{col}_dayofweek"] = dt.dt.dayofweek
                    df = df.drop(columns=[col])
            except Exception:
                pass

    # Basic imputations on the raw dataframe.
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Missing"
            df[col] = df[col].fillna(fill)

    # Ensure required columns exist.
    required = {TARGET_COL, QUANTITY_COL, COST_COL, GROUP_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Clip core positives now to avoid issues.
    for col in [TARGET_COL, QUANTITY_COL, COST_COL]:
        df[col] = np.clip(df[col].astype(float), MIN_POSITIVE_VALUE, None)

    return df


def build_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


# -----------------------------
# Model helpers
# -----------------------------


def fit_model(df_train: pd.DataFrame, alpha: float = RIDGE_ALPHA) -> Pipeline:
    preprocessor, _, _ = build_preprocessor(df_train)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL].to_numpy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        model.fit(X_train, y_train)
    return model


def choose_ridge_alpha(df0: pd.DataFrame) -> float:
    train_df, val_df = train_test_split(df0, test_size=TEST_SIZE, random_state=1729)
    y_val = val_df[TARGET_COL].to_numpy(dtype=float)

    scores = []
    for alpha in RIDGE_ALPHA_GRID:
        model = fit_model(train_df, alpha=alpha)
        preds = predict_model(model, val_df)
        scores.append((mean_squared_error(y_val, preds), alpha))

    return float(min(scores)[1])


def predict_model(model: Pipeline, df_eval: pd.DataFrame) -> np.ndarray:
    X_eval = df_eval.drop(columns=[TARGET_COL])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        return np.asarray(model.predict(X_eval), dtype=float)


# -----------------------------
# Channels
# -----------------------------


@dataclass
class ChannelArtifacts:
    coarse_score_edges: np.ndarray
    task_score_edges: np.ndarray
    task_qty_edges: np.ndarray
    subgroup_ref: str


def fit_channel_artifacts(df0: pd.DataFrame, preds0: np.ndarray) -> ChannelArtifacts:
    coarse_score_edges = compute_quantile_edges(preds0, COARSE_SCORE_BINS)
    task_score_edges = compute_quantile_edges(preds0, TASK_SCORE_BINS)
    task_qty_edges = compute_quantile_edges(df0[QUANTITY_COL].to_numpy(), TASK_QTY_BINS)

    mode_value = df0[GROUP_COL].mode(dropna=True)
    subgroup_ref = str(mode_value.iloc[0] if len(mode_value) else df0[GROUP_COL].iloc[0])

    return ChannelArtifacts(
        coarse_score_edges=coarse_score_edges,
        task_score_edges=task_score_edges,
        task_qty_edges=task_qty_edges,
        subgroup_ref=subgroup_ref,
    )


def coarse_channel_probs(preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    state_ids = assign_bins(preds, artifacts.coarse_score_edges)
    n_states = len(artifacts.coarse_score_edges) - 1
    return empirical_categorical_distribution(state_ids, n_states, PSEUDOCOUNT)


def task_channel_probs(df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    score_bin = assign_bins(preds, artifacts.task_score_edges)
    qty_bin = assign_bins(df[QUANTITY_COL].to_numpy(), artifacts.task_qty_edges)
    group_bin = (df[GROUP_COL].astype(str).to_numpy() != artifacts.subgroup_ref).astype(int)

    n_score = len(artifacts.task_score_edges) - 1
    n_qty = len(artifacts.task_qty_edges) - 1
    n_group = 2

    state_ids = score_bin * (n_qty * n_group) + qty_bin * n_group + group_bin
    n_states = n_score * n_qty * n_group
    return empirical_categorical_distribution(state_ids, n_states, PSEUDOCOUNT)


# -----------------------------
# Feedback dynamics
# -----------------------------


@dataclass
class FeedbackScales:
    q_sd: float
    c_sd: float
    y_sd: float
    sigma_q: float
    sigma_c: float
    sigma_y: float


def estimate_noise_scales(df0: pd.DataFrame) -> tuple[float, float, float]:
    sigma_q = NOISE_SCALE_Q * float(df0[QUANTITY_COL].std())
    sigma_c = NOISE_SCALE_C * float(df0[COST_COL].std())
    sigma_y = NOISE_SCALE_Y * float(df0[TARGET_COL].std())
    return sigma_q, sigma_c, sigma_y


def estimate_feedback_scales(df0: pd.DataFrame) -> FeedbackScales:
    q_sd = float(df0[QUANTITY_COL].std())
    c_sd = float(df0[COST_COL].std())
    y_sd = float(df0[TARGET_COL].std())
    sigma_q, sigma_c, sigma_y = estimate_noise_scales(df0)
    return FeedbackScales(
        q_sd=max(q_sd, EPS),
        c_sd=max(c_sd, EPS),
        y_sd=max(y_sd, EPS),
        sigma_q=sigma_q,
        sigma_c=sigma_c,
        sigma_y=sigma_y,
    )


def apply_feedback_one_round(
    df_t: pd.DataFrame,
    preds_t: np.ndarray,
    mu: float,
    scales: FeedbackScales,
    rng: np.random.Generator,
) -> pd.DataFrame:
    df_next = df_t.copy()

    r = preds_t - float(np.mean(preds_t))
    r_std = float(np.std(r)) + EPS
    r_tilde = r / r_std
    response = np.tanh(r_tilde)

    g = make_group_indicator(df_t[GROUP_COL])

    noise_q = rng.normal(0.0, scales.sigma_q, size=len(df_t))
    noise_c = rng.normal(0.0, scales.sigma_c, size=len(df_t))
    noise_y = rng.normal(0.0, scales.sigma_y, size=len(df_t))

    delta_q = -mu * scales.q_sd * response + noise_q
    delta_c = mu * scales.c_sd * (1.0 + RHO * g) * response + noise_c
    delta_y_std = (
        mu * response
        + B_C * (delta_c / scales.c_sd)
        - B_Q * (delta_q / scales.q_sd)
    )
    delta_y = scales.y_sd * delta_y_std + noise_y

    q_next = np.clip(df_t[QUANTITY_COL].to_numpy(dtype=float) + delta_q, MIN_POSITIVE_VALUE, None)
    c_next = np.clip(df_t[COST_COL].to_numpy(dtype=float) + delta_c, MIN_POSITIVE_VALUE, None)
    y_next = np.clip(df_t[TARGET_COL].to_numpy(dtype=float) + delta_y, MIN_POSITIVE_VALUE, None)

    df_next[QUANTITY_COL] = q_next
    df_next[COST_COL] = c_next
    df_next[TARGET_COL] = y_next

    return df_next


# -----------------------------
# Simulation
# -----------------------------


def run_single_condition(
    df0: pd.DataFrame,
    artifacts: ChannelArtifacts,
    mu: float,
    seed: int,
    scales: FeedbackScales,
    ridge_alpha: float,
    round_records_outpath: Path | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    df_current = df0.copy()

    round_records: list[dict] = []

    for t in range(N_ROUNDS - 1):
        train_df, eval_df = train_test_split(df_current, test_size=TEST_SIZE, random_state=seed + 1000 * t)
        train_df = train_df.reset_index(drop=False).rename(columns={"index": "_row_id"})
        eval_df = eval_df.reset_index(drop=False).rename(columns={"index": "_row_id"})

        train_model_df = train_df.drop(columns=["_row_id"])
        eval_model_df = eval_df.drop(columns=["_row_id"])

        model = fit_model(train_model_df, alpha=ridge_alpha)

        preds_current_all = predict_model(model, df_current)
        preds_train_current = predict_model(model, train_model_df)
        preds_eval_current = predict_model(model, eval_model_df)

        # Build next round from full current data.
        df_next = apply_feedback_one_round(
            df_t=df_current,
            preds_t=preds_current_all,
            mu=mu,
            scales=scales,
            rng=rng,
        )

        # Evaluate same rows in next round using row identities.
        next_eval_df = df_next.iloc[eval_df["_row_id"].to_numpy()].copy()
        preds_eval_next = predict_model(model, next_eval_df)

        y_current_all = df_current[TARGET_COL].to_numpy(dtype=float)
        y_train_current = train_df[TARGET_COL].to_numpy(dtype=float)
        y_eval_current = eval_df[TARGET_COL].to_numpy(dtype=float)
        y_eval_next = next_eval_df[TARGET_COL].to_numpy(dtype=float)

        train_mse = float(mean_squared_error(y_train_current, preds_train_current))
        current_all_mse = float(mean_squared_error(y_current_all, preds_current_all))
        eval_mse_current = float(mean_squared_error(y_eval_current, preds_eval_current))
        eval_mse_next = float(mean_squared_error(y_eval_next, preds_eval_next))
        current_rmse = float(np.sqrt(eval_mse_current))
        next_rmse = float(np.sqrt(eval_mse_next))

        delta_rep = abs(eval_mse_current - eval_mse_next)
        v_t = abs(eval_mse_current - eval_mse_next)
        # Diagnostic only, not a validated sampling term:
        # sampling_candidate = |MSE_D_t_all(f_t) - MSE_D_t_eval(f_t)|.
        # This is same-round, but MSE_D_t_all includes train rows used to fit f_t.
        sampling_candidate = abs(current_all_mse - eval_mse_current)

        # Observable channel probabilities and FR step lengths.
        coarse_p = coarse_channel_probs(preds_current_all, artifacts)
        task_p = task_channel_probs(df_current, preds_current_all, artifacts)

        preds_next_all_same_model = predict_model(model, df_next)
        coarse_q = coarse_channel_probs(preds_next_all_same_model, artifacts)
        task_q = task_channel_probs(df_next, preds_next_all_same_model, artifacts)

        fr_step_coarse = fisher_rao_categorical(coarse_p, coarse_q)
        fr_step_task = fisher_rao_categorical(task_p, task_q)

        round_records.append(
            {
                "round": t,
                "mu": mu,
                "seed": seed,
                "eval_mse_current": eval_mse_current,
                "current_all_mse": current_all_mse,
                "eval_mse_next": eval_mse_next,
                "train_mse": train_mse,
                "current_rmse": current_rmse,
                "next_rmse": next_rmse,
                "delta_rep": delta_rep,
                "v_t": v_t,
                "sampling_candidate": sampling_candidate,
                "fr_step_coarse": fr_step_coarse,
                "fr_step_task": fr_step_task,
            }
        )

        df_current = df_next

    # Aggregate
    rr = pd.DataFrame(round_records)
    if round_records_outpath is not None:
        round_records_outpath.parent.mkdir(parents=True, exist_ok=True)
        round_cols = [
            "round",
            "mu",
            "seed",
            "train_mse",
            "eval_mse_current",
            "eval_mse_next",
            "delta_rep",
            "v_t",
            "sampling_candidate",
        ]
        rr[round_cols].to_csv(round_records_outpath, index=False)

    return {
        "mu": mu,
        "seed": seed,
        "Delta_rep_T": float(rr["delta_rep"].mean()),
        "V_T": float(rr["v_t"].mean()),
        "sampling_candidate_T": float(rr["sampling_candidate"].mean()),
        "mean_current_rmse": float(rr["current_rmse"].mean()),
        "mean_next_rmse": float(rr["next_rmse"].mean()),
        "A_T_coarse": float(rr["fr_step_coarse"].sum()),
        "A_T_task": float(rr["fr_step_task"].sum()),
        "A_rate_coarse": float(rr["fr_step_coarse"].mean()),
        "A_rate_task": float(rr["fr_step_task"].mean()),
    }


# -----------------------------
# Plotting
# -----------------------------


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = results_df.groupby("mu").agg(
        Delta_rep_T_mean=("Delta_rep_T", "mean"),
        Delta_rep_T_std=("Delta_rep_T", "std"),
        V_T_mean=("V_T", "mean"),
        V_T_std=("V_T", "std"),
        sampling_candidate_T_mean=("sampling_candidate_T", "mean"),
        sampling_candidate_T_std=("sampling_candidate_T", "std"),
        A_rate_coarse_mean=("A_rate_coarse", "mean"),
        A_rate_coarse_std=("A_rate_coarse", "std"),
        A_rate_task_mean=("A_rate_task", "mean"),
        A_rate_task_std=("A_rate_task", "std"),
    ).reset_index()
    return summary


def plot_errorbar(x, y, yerr, xlabel, ylabel, title, outpath, label=None):
    plt.figure(figsize=(5.2, 3.6))
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if label is not None:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def make_plots(results_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    plot_errorbar(
        summary_df["mu"],
        summary_df["Delta_rep_T_mean"],
        summary_df["Delta_rep_T_std"],
        xlabel="Feedback strength $\\mu$",
        ylabel=r"$\Delta_T^{rep}$",
        title="Prequential gap vs feedback strength",
        outpath=OUTPUT_DIR / "figure_1_gap_vs_mu.png",
    )

    plt.figure(figsize=(5.2, 3.6))
    plt.errorbar(
        summary_df["mu"], summary_df["V_T_mean"], yerr=summary_df["V_T_std"],
        marker="o", capsize=4, label="Drift term"
    )
    plt.errorbar(
        summary_df["mu"],
        summary_df["sampling_candidate_T_mean"],
        yerr=summary_df["sampling_candidate_T_std"],
        marker="s",
        capsize=4,
        label="Sampling candidate"
    )
    plt.xlabel("Feedback strength $\\mu$")
    plt.ylabel("Aggregate MSE gap")
    plt.title("Drift term and sampling candidate vs feedback strength")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_2_drift_sampling_candidate_vs_mu.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5.2, 3.6))
    plt.errorbar(
        summary_df["mu"], summary_df["A_rate_coarse_mean"], yerr=summary_df["A_rate_coarse_std"],
        marker="o", capsize=4, label="Coarse channel"
    )
    plt.errorbar(
        summary_df["mu"], summary_df["A_rate_task_mean"], yerr=summary_df["A_rate_task_std"],
        marker="o", capsize=4, label="Task-aligned channel"
    )
    plt.xlabel("Feedback strength $\\mu$")
    plt.ylabel("Average observable FR rate")
    plt.title("Observable Fisher rate vs feedback strength")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_3_fr_rate_vs_mu.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5.2, 3.6))
    plt.scatter(results_df["A_rate_coarse"], results_df["Delta_rep_T"], alpha=0.75, label="Coarse channel")
    plt.scatter(results_df["A_rate_task"], results_df["Delta_rep_T"], alpha=0.75, label="Task-aligned channel")
    plt.xlabel("Average observable FR rate")
    plt.ylabel(r"$\Delta_T^{rep}$")
    plt.title("Observable FR rate vs prequential gap")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_4_fr_vs_gap_scatter.png", dpi=200)
    plt.close()


def write_dataset_profile(df: pd.DataFrame) -> None:
    profile_lines = [
        f"Rows: {len(df)}",
        "",
        "Columns:",
        *[f"- {col}: {df[col].dtype}" for col in df.columns],
        "",
        "Core variable ranges:",
    ]
    for col in [TARGET_COL, COST_COL, QUANTITY_COL]:
        desc = df[col].describe()
        profile_lines.append(
            f"- {col}: min={desc['min']:.4f}, median={desc['50%']:.4f}, "
            f"mean={desc['mean']:.4f}, max={desc['max']:.4f}, sd={desc['std']:.4f}"
        )
    profile_lines.extend(
        [
            "",
            f"{GROUP_COL} values:",
            *[f"- {value}" for value in sorted(df[GROUP_COL].astype(str).unique())],
        ]
    )
    (OUTPUT_DIR / "dataset_profile.txt").write_text("\n".join(profile_lines), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    df0 = load_and_prepare_base_dataframe(DATA_PATH)
    write_dataset_profile(df0)
    print(f"Loaded dataframe with shape: {df0.shape}")
    print("Columns:")
    for c in df0.columns:
        print(f"  - {c}")

    ridge_alpha = choose_ridge_alpha(df0)
    print(f"Chosen Ridge alpha: {ridge_alpha}")
    print(
        "Sampling candidate formula (diagnostic, not validated): "
        "abs(current_all_mse - eval_mse_current), where "
        "current_all_mse = MSE(Y_t on all rows, f_t(X_t) on all rows) and "
        "eval_mse_current = MSE(Y_t on eval rows, f_t(X_t) on eval rows)."
    )

    # Fit baseline model and fixed channels from round 0.
    baseline_model = fit_model(df0, alpha=ridge_alpha)
    baseline_preds = predict_model(baseline_model, df0)
    artifacts = fit_channel_artifacts(df0, baseline_preds)

    scales = estimate_feedback_scales(df0)
    print(
        "Feedback scales: "
        f"q_sd={scales.q_sd:.4f}, c_sd={scales.c_sd:.4f}, y_sd={scales.y_sd:.4f}; "
        f"sigma_q={scales.sigma_q:.4f}, sigma_c={scales.sigma_c:.4f}, sigma_y={scales.sigma_y:.4f}"
    )

    results: list[dict] = []
    for mu in MU_GRID:
        for seed in range(N_SEEDS):
            print(f"Running mu={mu:.3f}, seed={seed}")
            result = run_single_condition(
                df0=df0,
                artifacts=artifacts,
                mu=mu,
                seed=seed,
                scales=scales,
                ridge_alpha=ridge_alpha,
            )
            results.append(result)

    results_df = pd.DataFrame(results)
    summary_df = summarize_results(results_df)

    results_df.to_csv(OUTPUT_DIR / "regional_feedback_results_by_seed.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "regional_feedback_summary.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "table_1_summary.csv", index=False)

    # Simple correlation table.
    corr_rows = []
    for ch_name in ["coarse", "task"]:
        x = results_df[f"A_rate_{ch_name}"]
        y_gap = results_df["Delta_rep_T"]
        y_vt = results_df["V_T"]
        corr_rows.append(
            {
                "channel": ch_name,
                "pearson_with_gap": float(pd.Series(x).corr(pd.Series(y_gap), method="pearson")),
                "spearman_with_gap": float(pd.Series(x).corr(pd.Series(y_gap), method="spearman")),
                "pearson_with_vt": float(pd.Series(x).corr(pd.Series(y_vt), method="pearson")),
                "spearman_with_vt": float(pd.Series(x).corr(pd.Series(y_vt), method="spearman")),
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUTPUT_DIR / "regional_feedback_correlations.csv", index=False)
    corr_df.to_csv(OUTPUT_DIR / "table_2_correlations.csv", index=False)

    make_plots(results_df, summary_df)

    print("\nSaved outputs to:", OUTPUT_DIR)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print("\nChannel correlations:")
    print(corr_df.to_string(index=False))


if __name__ == "__main__":
    main()
