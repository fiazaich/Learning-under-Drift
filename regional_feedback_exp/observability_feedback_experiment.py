from __future__ import annotations

import math
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

_MPLCONFIGDIR = Path(os.environ.get("TMPDIR", "/tmp")) / "regional_feedback_matplotlib_cache"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import scienceplots  # noqa: F401

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


# Config

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
BURN_IN_RANDOM_OFFSET = 100_000
RIDGE_ALPHA = 1.0
RIDGE_ALPHA_GRID = [0.1, 1.0, 10.0, 100.0]

RHO = 0.5
B_C = 0.4
B_Q = 0.3

NOISE_SCALE_Q = 0.02
NOISE_SCALE_C = 0.02
NOISE_SCALE_Y = 0.02
RESPONSE_NOISE_SCALE = 0.15
COMMON_FEEDBACK_SHOCK_SCALE = 0.20

COARSE_SCORE_BINS = 5
NULL_RANDOM_BUCKETS = 8
TASK_QTY_BINS = 4
TASK_COST_BINS = 4
PSEUDOCOUNT = 1e-6
MIN_POSITIVE_VALUE = 1e-3
EPS = 1e-8
SAVEFIG_KW = {}

CHANNEL_ORDER = ["null", "coarse", "task_no_qty", "task_no_cost", "task_no_group", "task"]
CHANNEL_LABELS = {
    "null": "Null blind",
    "coarse": "Coarse score",
    "task_no_qty": "Task minus quantity",
    "task_no_cost": "Task minus cost",
    "task_no_group": "Task minus subgroup",
    "task": "Task-aligned",
}


# Utilities

def fisher_rao_categorical(p: np.ndarray, q: np.ndarray) -> float:
    inner = float(np.sum(np.sqrt(p * q)))
    inner = float(np.clip(inner, -1.0, 1.0))
    return 2.0 * math.acos(inner)


def finite_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total <= EPS:
        return 0.0
    residual = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - residual / total)


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return 0.0
    value = pd.Series(x).corr(pd.Series(y), method=method)
    return 0.0 if pd.isna(value) else float(value)


def standard_error(std_values: pd.Series, n: int) -> pd.Series:
    return std_values / math.sqrt(n)


def save_figure_all_formats(fig, fig_path: Path) -> None:
    fig.savefig(fig_path.with_suffix(".pdf"), **SAVEFIG_KW)
    fig.savefig(fig_path.with_suffix(".svg"), **SAVEFIG_KW)
    fig.savefig(fig_path.with_suffix(".png"), dpi=600, **SAVEFIG_KW)
    plt.close(fig)


def style_axis(ax) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.6)


def channel_label(channel: str) -> str:
    return CHANNEL_LABELS.get(str(channel), str(channel).replace("_", " "))


def compute_quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        v = float(np.mean(values))
        edges = np.array([v - 1.0, v + 1.0], dtype=float)
    return edges


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.asarray(np.digitize(values, edges[1:-1], right=False), dtype=int)


def empirical_categorical_distribution(state_ids: np.ndarray, n_states: int, pseudocount: float) -> np.ndarray:
    counts = np.bincount(state_ids, minlength=n_states).astype(float)
    counts += pseudocount
    return counts / counts.sum()


def make_group_indicator(series: pd.Series) -> np.ndarray:
    mode_value = series.mode(dropna=True)
    ref = mode_value.iloc[0] if len(mode_value) else series.dropna().iloc[0]
    return (series.astype(str).to_numpy() != str(ref)).astype(int)


# Data and model

def load_and_prepare_base_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            cleaned = df[col].astype("string").str.replace(",", "", regex=False).str.strip()
            numeric = pd.to_numeric(cleaned, errors="coerce")
            non_missing = df[col].notna()
            if non_missing.any() and numeric[non_missing].notna().all():
                df[col] = numeric

    drop_candidates = []
    for col in df.columns:
        col_low = col.lower()
        if col_low == TARGET_COL.lower():
            continue
        if any(tok in col_low for tok in ["id", "ordernumber", "invoice", "transaction"]):
            drop_candidates.append(col)
    drop_candidates = [c for c in drop_candidates if c.lower() != "warehousecode"]
    df = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore").copy()

    for col in list(df.columns):
        if "date" in col.lower():
            dt = pd.to_datetime(df[col], format="%d/%m/%y", errors="coerce")
            if dt.notna().any():
                df[f"{col}_year"] = dt.dt.year
                df[f"{col}_month"] = dt.dt.month
                df[f"{col}_dayofweek"] = dt.dt.dayofweek
                df = df.drop(columns=[col])

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "Missing")

    missing = {TARGET_COL, QUANTITY_COL, COST_COL, GROUP_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in [TARGET_COL, QUANTITY_COL, COST_COL]:
        df[col] = np.clip(df[col].astype(float), MIN_POSITIVE_VALUE, None)
    return df


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
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
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def fit_model(df_train: pd.DataFrame, alpha: float = RIDGE_ALPHA) -> Pipeline:
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(df_train)),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        model.fit(df_train.drop(columns=[TARGET_COL]), df_train[TARGET_COL].to_numpy())
    return model


def predict_model(model: Pipeline, df_eval: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
        return np.asarray(model.predict(df_eval.drop(columns=[TARGET_COL])), dtype=float)


def choose_ridge_alpha(df0: pd.DataFrame) -> float:
    train_df, val_df = train_test_split(df0, test_size=TEST_SIZE, random_state=1729)
    y_val = val_df[TARGET_COL].to_numpy(dtype=float)
    scores = []
    for alpha in RIDGE_ALPHA_GRID:
        model = fit_model(train_df, alpha=alpha)
        preds = predict_model(model, val_df)
        scores.append((mean_squared_error(y_val, preds), alpha))
    return float(min(scores)[1])


# Monitoring channels

@dataclass
class ChannelArtifacts:
    coarse_score_edges: np.ndarray
    task_qty_edges: np.ndarray
    task_cost_edges: np.ndarray
    subgroup_ref: str


def fit_channel_artifacts(df0: pd.DataFrame, preds0: np.ndarray) -> ChannelArtifacts:
    mode_value = df0[GROUP_COL].mode(dropna=True)
    return ChannelArtifacts(
        coarse_score_edges=compute_quantile_edges(preds0, COARSE_SCORE_BINS),
        task_qty_edges=compute_quantile_edges(df0[QUANTITY_COL].to_numpy(), TASK_QTY_BINS),
        task_cost_edges=compute_quantile_edges(df0[COST_COL].to_numpy(), TASK_COST_BINS),
        subgroup_ref=str(mode_value.iloc[0] if len(mode_value) else df0[GROUP_COL].iloc[0]),
    )


def fixed_random_buckets(n_rows: int, n_buckets: int) -> np.ndarray:
    row_ids = np.arange(n_rows, dtype=np.uint64)
    hashed = row_ids * np.uint64(11400714819323198485) + np.uint64(0x9E3779B97F4A7C15)
    return np.asarray(hashed % np.uint64(n_buckets), dtype=int)


def null_channel_probs(df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    del preds, artifacts
    state_ids = fixed_random_buckets(len(df), NULL_RANDOM_BUCKETS)
    return empirical_categorical_distribution(state_ids, NULL_RANDOM_BUCKETS, PSEUDOCOUNT)


def coarse_channel_probs(df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    del df
    state_ids = assign_bins(preds, artifacts.coarse_score_edges)
    return empirical_categorical_distribution(state_ids, len(artifacts.coarse_score_edges) - 1, PSEUDOCOUNT)


def _task_bins(df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score_bin = assign_bins(preds, artifacts.coarse_score_edges)
    qty_bin = assign_bins(df[QUANTITY_COL].to_numpy(), artifacts.task_qty_edges)
    cost_bin = assign_bins(df[COST_COL].to_numpy(), artifacts.task_cost_edges)
    group_bin = (df[GROUP_COL].astype(str).to_numpy() != artifacts.subgroup_ref).astype(int)
    return score_bin, qty_bin, cost_bin, group_bin


def task_channel_probs(df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    score_bin, qty_bin, cost_bin, group_bin = _task_bins(df, preds, artifacts)
    n_score = len(artifacts.coarse_score_edges) - 1
    n_qty = len(artifacts.task_qty_edges) - 1
    n_cost = len(artifacts.task_cost_edges) - 1
    state_ids = score_bin * (n_qty * n_cost * 2) + qty_bin * (n_cost * 2) + cost_bin * 2 + group_bin
    return empirical_categorical_distribution(state_ids, n_score * n_qty * n_cost * 2, PSEUDOCOUNT)


def task_no_qty_channel_probs(df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    score_bin, _, cost_bin, group_bin = _task_bins(df, preds, artifacts)
    n_score = len(artifacts.coarse_score_edges) - 1
    n_cost = len(artifacts.task_cost_edges) - 1
    state_ids = score_bin * (n_cost * 2) + cost_bin * 2 + group_bin
    return empirical_categorical_distribution(state_ids, n_score * n_cost * 2, PSEUDOCOUNT)


def task_no_cost_channel_probs(df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    score_bin, qty_bin, _, group_bin = _task_bins(df, preds, artifacts)
    n_score = len(artifacts.coarse_score_edges) - 1
    n_qty = len(artifacts.task_qty_edges) - 1
    state_ids = score_bin * (n_qty * 2) + qty_bin * 2 + group_bin
    return empirical_categorical_distribution(state_ids, n_score * n_qty * 2, PSEUDOCOUNT)


def task_no_group_channel_probs(df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    score_bin, qty_bin, cost_bin, _ = _task_bins(df, preds, artifacts)
    n_score = len(artifacts.coarse_score_edges) - 1
    n_qty = len(artifacts.task_qty_edges) - 1
    n_cost = len(artifacts.task_cost_edges) - 1
    state_ids = score_bin * (n_qty * n_cost) + qty_bin * n_cost + cost_bin
    return empirical_categorical_distribution(state_ids, n_score * n_qty * n_cost, PSEUDOCOUNT)


CHANNEL_PROB_FUNCTIONS = {
    "null": null_channel_probs,
    "coarse": coarse_channel_probs,
    "task_no_qty": task_no_qty_channel_probs,
    "task_no_cost": task_no_cost_channel_probs,
    "task_no_group": task_no_group_channel_probs,
    "task": task_channel_probs,
}


# Feedback dynamics

@dataclass
class FeedbackScales:
    q_sd: float
    c_sd: float
    y_sd: float
    sigma_q: float
    sigma_c: float
    sigma_y: float


def estimate_feedback_scales(df0: pd.DataFrame) -> FeedbackScales:
    q_sd = max(float(df0[QUANTITY_COL].std()), EPS)
    c_sd = max(float(df0[COST_COL].std()), EPS)
    y_sd = max(float(df0[TARGET_COL].std()), EPS)
    return FeedbackScales(
        q_sd=q_sd,
        c_sd=c_sd,
        y_sd=y_sd,
        sigma_q=NOISE_SCALE_Q * q_sd,
        sigma_c=NOISE_SCALE_C * c_sd,
        sigma_y=NOISE_SCALE_Y * y_sd,
    )


def apply_feedback_one_round(
    df_t: pd.DataFrame,
    preds_t: np.ndarray,
    mu: float,
    scales: FeedbackScales,
    rng: np.random.Generator,
) -> pd.DataFrame:
    df_t = df_t.reset_index(drop=True)
    df_next = df_t.copy()

    r = preds_t - float(np.mean(preds_t))
    response = np.tanh(r / (float(np.std(r)) + EPS) + rng.normal(0.0, RESPONSE_NOISE_SCALE, size=len(df_t)))
    g = make_group_indicator(df_t[GROUP_COL])

    feedback_multiplier = max(0.0, 1.0 + rng.normal(0.0, COMMON_FEEDBACK_SHOCK_SCALE))
    common_q = rng.normal(0.0, mu * COMMON_FEEDBACK_SHOCK_SCALE * scales.q_sd)
    common_c = rng.normal(0.0, mu * COMMON_FEEDBACK_SHOCK_SCALE * scales.c_sd)
    common_y = rng.normal(0.0, mu * COMMON_FEEDBACK_SHOCK_SCALE * scales.y_sd)

    noise_q = rng.normal(0.0, scales.sigma_q, size=len(df_t))
    noise_c = rng.normal(0.0, scales.sigma_c, size=len(df_t))
    noise_y = rng.normal(0.0, scales.sigma_y, size=len(df_t))

    delta_q = -mu * feedback_multiplier * scales.q_sd * response + common_q + noise_q
    delta_c = mu * feedback_multiplier * scales.c_sd * (1.0 + RHO * g) * response + common_c + noise_c
    delta_y_std = mu * feedback_multiplier * response + B_C * (delta_c / scales.c_sd) - B_Q * (delta_q / scales.q_sd)
    delta_y = scales.y_sd * delta_y_std + common_y + noise_y

    df_next[QUANTITY_COL] = np.clip(df_t[QUANTITY_COL].to_numpy(dtype=float) + delta_q, MIN_POSITIVE_VALUE, None)
    df_next[COST_COL] = np.clip(df_t[COST_COL].to_numpy(dtype=float) + delta_c, MIN_POSITIVE_VALUE, None)
    df_next[TARGET_COL] = np.clip(df_t[TARGET_COL].to_numpy(dtype=float) + delta_y, MIN_POSITIVE_VALUE, None)
    return df_next.reset_index(drop=True)


# Simulation and main metrics

def split_with_row_ids(df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_df = df.reset_index(drop=True).copy()
    split_df["_row_id"] = np.arange(len(split_df), dtype=int)
    train_df, eval_df = train_test_split(split_df, test_size=TEST_SIZE, random_state=random_state)
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    return train_df, eval_df, train_df.drop(columns=["_row_id"]), eval_df.drop(columns=["_row_id"])


def burn_in_learner(df0: pd.DataFrame, seed: int, ridge_alpha: float) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, eval_df, train_model_df, eval_model_df = split_with_row_ids(df0, random_state=seed + BURN_IN_RANDOM_OFFSET)
    return fit_model(train_model_df, alpha=ridge_alpha), train_df, eval_df, eval_model_df


def aggregate_round_metrics(rr: pd.DataFrame) -> dict:
    rhat_t = float(rr["empirical_loss"].mean())
    r_t = float(rr["population_risk"].mean())
    rplus_t = float(rr["preq_target"].mean())
    return {
        "Rhat_T": rhat_t,
        "R_T": r_t,
        "Rplus_T": rplus_t,
        "Delta_rep_T": abs(rhat_t - rplus_t),
        "V_T": float(rr["v_t"].mean()),
    }


def run_single_condition(
    df0: pd.DataFrame,
    artifacts: ChannelArtifacts,
    mu: float,
    seed: int,
    scales: FeedbackScales,
    ridge_alpha: float,
    round_records_sink: list[dict],
) -> dict:
    rng = np.random.default_rng(seed)
    df_current = df0.reset_index(drop=True).copy()
    model, train_df, eval_df, eval_model_df = burn_in_learner(df0, seed, ridge_alpha)
    round_records: list[dict] = []

    for t in range(N_ROUNDS - 1):
        preds_current_all = predict_model(model, df_current)
        preds_eval_current = predict_model(model, eval_model_df)

        df_next = apply_feedback_one_round(df_current, preds_current_all, mu, scales, rng)
        next_eval_df = df_next.iloc[eval_df["_row_id"].to_numpy()].copy()
        preds_next_all_same_model = predict_model(model, df_next)

        y_current_all = df_current[TARGET_COL].to_numpy(dtype=float)
        y_next_all = df_next[TARGET_COL].to_numpy(dtype=float)
        y_eval_current = eval_df[TARGET_COL].to_numpy(dtype=float)

        loss_current_all = (y_current_all - preds_current_all) ** 2
        loss_next_all = (y_next_all - preds_next_all_same_model) ** 2
        current_all_mse = float(np.mean(loss_current_all))
        next_all_mse = float(np.mean(loss_next_all))
        empirical_loss = float(mean_squared_error(y_eval_current, preds_eval_current))
        v_t = abs(current_all_mse - next_all_mse)

        fr_steps = {}
        for channel in CHANNEL_ORDER:
            p = CHANNEL_PROB_FUNCTIONS[channel](df_current, preds_current_all, artifacts)
            q = CHANNEL_PROB_FUNCTIONS[channel](df_next, preds_next_all_same_model, artifacts)
            fr_steps[channel] = fisher_rao_categorical(p, q)

        record = {
            "round": t,
            "mu": mu,
            "seed": seed,
            "empirical_loss": empirical_loss,
            "population_risk": current_all_mse,
            "preq_target": next_all_mse,
            "v_t": v_t,
            **{f"fr_step_{channel}": fr_steps[channel] for channel in CHANNEL_ORDER},
        }
        round_records.append(record)
        round_records_sink.append(record)

        df_current = df_next
        if t < N_ROUNDS - 2:
            train_df, eval_df, train_model_df, eval_model_df = split_with_row_ids(
                df_current,
                random_state=seed + 1000 * (t + 1),
            )
            model = fit_model(train_model_df, alpha=ridge_alpha)

    rr = pd.DataFrame(round_records)
    return {
        "mu": mu,
        "seed": seed,
        **aggregate_round_metrics(rr),
        **{f"A_T_{channel}": float(rr[f"fr_step_{channel}"].sum()) for channel in CHANNEL_ORDER},
        **{f"A_rate_{channel}": float(rr[f"fr_step_{channel}"].mean()) for channel in CHANNEL_ORDER},
    }


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "Delta_rep_T_mean": ("Delta_rep_T", "mean"),
        "Delta_rep_T_std": ("Delta_rep_T", "std"),
        "V_T_mean": ("V_T", "mean"),
        "V_T_std": ("V_T", "std"),
    }
    for channel in CHANNEL_ORDER:
        agg_spec[f"A_rate_{channel}_mean"] = (f"A_rate_{channel}", "mean")
        agg_spec[f"A_rate_{channel}_std"] = (f"A_rate_{channel}", "std")
    return results_df.groupby("mu").agg(**agg_spec).reset_index()


def build_step_diagnostics_table(round_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_positive = round_df[round_df["mu"] > 0.0]
    for channel in CHANNEL_ORDER:
        predictor = f"fr_step_{channel}"
        model_df = all_positive[["v_t", predictor]].dropna()
        rows.append(
            {
                "mu": "all_positive",
                "target": "v_t",
                "channel": channel,
                "predictor": predictor,
                "n_steps": int(len(model_df)),
                "pearson": safe_corr(model_df[predictor], model_df["v_t"], method="pearson"),
                "spearman": safe_corr(model_df[predictor], model_df["v_t"], method="spearman"),
            }
        )
    return pd.DataFrame(rows)


def fit_raw_rate_regression_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["V_T", "Delta_rep_T"]:
        for channel in CHANNEL_ORDER:
            predictor = f"A_rate_{channel}"
            model_df = results_df[[target, predictor]].dropna()
            x = model_df[[predictor]].to_numpy(dtype=float)
            y = model_df[target].to_numpy(dtype=float)
            model = LinearRegression()
            model.fit(x, y)
            preds = model.predict(x)
            rows.append(
                {
                    "target": target,
                    "channel": channel,
                    "predictor": predictor,
                    "n": int(len(model_df)),
                    "pearson": safe_corr(model_df[predictor], model_df[target], method="pearson"),
                    "spearman": safe_corr(model_df[predictor], model_df[target], method="spearman"),
                    "slope": float(model.coef_[0]),
                    "intercept": float(model.intercept_),
                    "r2_in_sample": finite_r2(y, preds),
                }
            )
    return pd.DataFrame(rows)


def build_channel_association_summary_table(
    raw_rate_regression_df: pd.DataFrame,
    step_diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    run_labels = {"V_T": r"$V_T$", "Delta_rep_T": r"$\Delta_T^{\mathrm{rep}}$"}
    for channel in CHANNEL_ORDER:
        for target, label in run_labels.items():
            row = raw_rate_regression_df[
                (raw_rate_regression_df["channel"] == channel) & (raw_rate_regression_df["target"] == target)
            ].iloc[0]
            rows.append(
                {
                    "scale": "run_level",
                    "channel": channel,
                    "target": target,
                    "target_label": label,
                    "metric": "raw_rate_r2",
                    "value": float(row["r2_in_sample"]),
                }
            )
        step_row = step_diagnostics_df[
            (step_diagnostics_df["channel"] == channel) & (step_diagnostics_df["target"] == "v_t")
        ].iloc[0]
        rows.append(
            {
                "scale": "transition_level",
                "channel": channel,
                "target": "v_t",
                "target_label": r"$v_t$",
                "metric": "spearman_rho",
                "value": float(step_row["spearman"]),
            }
        )
    return pd.DataFrame(rows)


# Main figures

def plot_main_channel_association_summary(
    raw_rate_regression_df: pd.DataFrame,
    step_diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    association_df = build_channel_association_summary_table(raw_rate_regression_df, step_diagnostics_df)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), layout="constrained")
    x = np.arange(len(CHANNEL_ORDER))
    bar_width = 0.36

    for idx, (target, label) in enumerate([("V_T", r"$V_T$"), ("Delta_rep_T", r"$\Delta_T^{\mathrm{rep}}$")]):
        values = [
            float(
                association_df[
                    (association_df["scale"] == "run_level")
                    & (association_df["channel"] == channel)
                    & (association_df["target"] == target)
                ]["value"].iloc[0]
            )
            for channel in CHANNEL_ORDER
        ]
        axes[0].bar(x + (idx - 0.5) * bar_width, values, width=bar_width, label=label, color=PALETTE[idx])
    axes[0].set_ylabel(r"Raw-rate regression $R^2$")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([channel_label(channel) for channel in CHANNEL_ORDER], rotation=25, ha="right", fontsize=7.0)
    axes[0].legend(frameon=False, fontsize=7.0, loc="upper left")

    values = [
        float(
            association_df[
                (association_df["scale"] == "transition_level")
                & (association_df["channel"] == channel)
                & (association_df["target"] == "v_t")
            ]["value"].iloc[0]
        )
        for channel in CHANNEL_ORDER
    ]
    axes[1].bar(x, values, width=0.58, label=r"$v_t$", color=PALETTE[0])
    axes[1].set_ylabel(r"Spearman $\rho$")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([channel_label(channel) for channel in CHANNEL_ORDER], rotation=25, ha="right", fontsize=7.0)
    axes[1].legend(frameon=False, fontsize=7.0, loc="upper left")

    for label, ax in zip(["(a)", "(b)"], axes):
        ax.axhline(0.0, color="black", lw=0.7)
        ax.text(0.98, 0.96, label, transform=ax.transAxes, ha="right", va="top", fontsize=8.0)
        style_axis(ax)
    save_figure_all_formats(fig, OUTPUT_DIR / "figure_main_channel_association_summary.png")
    return association_df


def plot_main_step_fr_association(step_diagnostics_df: pd.DataFrame) -> pd.DataFrame:
    step_vt = step_diagnostics_df[
        (step_diagnostics_df["mu"].astype(str) == "all_positive") & (step_diagnostics_df["target"] == "v_t")
    ].sort_values("spearman", ascending=False)

    fig, ax = plt.subplots(figsize=(7.2, 2.7), layout="constrained")
    x = np.arange(len(step_vt))
    bars = ax.bar(x, step_vt["spearman"].to_numpy(dtype=float), color=PALETTE[0], width=0.68)
    ax.set_ylabel(r"Spearman $\rho(\mathrm{FR\ step}, v_t)$")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([channel_label(channel) for channel in step_vt["channel"]], rotation=25, ha="right", fontsize=7.0)
    for bar, value in zip(bars, step_vt["spearman"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.015, f"{float(value):.2f}", ha="center", va="bottom", fontsize=7.0)
    ax.text(0.98, 0.94, f"n={int(step_vt['n_steps'].iloc[0])} positive-$\\mu$ transitions", transform=ax.transAxes, ha="right", va="top", fontsize=7.0)
    style_axis(ax)
    save_figure_all_formats(fig, OUTPUT_DIR / "figure_main_step_fr_association.png")
    return step_vt


def plot_main_feedback_manipulation_check(summary_df: pd.DataFrame, n_seeds: int) -> None:
    mu = summary_df["mu"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.55), layout="constrained")

    for target, label, color, marker in [
        ("V_T", r"$V_T$", PALETTE[0], "o"),
        ("Delta_rep_T", r"$\Delta_T^{\mathrm{rep}}$", PALETTE[1], "s"),
    ]:
        y = summary_df[f"{target}_mean"].to_numpy(dtype=float)
        yerr = standard_error(summary_df[f"{target}_std"], n_seeds).to_numpy(dtype=float)
        axes[0].errorbar(mu, y, yerr=yerr, marker=marker, markersize=3.2, lw=0.9, color=color, capsize=2.0, label=label)
    axes[0].set_xlabel(r"Feedback strength $\mu$")
    axes[0].set_ylabel("Risk quantity")
    axes[0].legend(frameon=False, fontsize=7.0, loc="upper left", bbox_to_anchor=(0.10, 0.98))
    style_axis(axes[0])

    for channel, marker, color in [("null", "o", PALETTE[0]), ("coarse", "s", PALETTE[1]), ("task", "^", PALETTE[2])]:
        y = summary_df[f"A_rate_{channel}_mean"].to_numpy(dtype=float)
        yerr = standard_error(summary_df[f"A_rate_{channel}_std"], n_seeds).to_numpy(dtype=float)
        axes[1].errorbar(mu, y, yerr=yerr, marker=marker, markersize=3.2, lw=0.9, color=color, capsize=2.0, label=channel_label(channel))
    axes[1].set_xlabel(r"Feedback strength $\mu$")
    axes[1].set_ylabel("Raw observed FR rate")
    axes[1].legend(frameon=False, fontsize=7.0, loc="upper left", bbox_to_anchor=(0.10, 0.98))
    style_axis(axes[1])

    for label, ax in zip(["(a)", "(b)"], axes):
        ax.text(0.02, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=8.0)
    save_figure_all_formats(fig, OUTPUT_DIR / "figure_main_feedback_manipulation_check.png")


def write_main_figure_values_summary(
    association_df: pd.DataFrame,
    step_vt_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    lines = [
        "Main figure values for regional feedback observability experiment",
        "",
        "Figure A: channel association summary",
    ]
    for _, row in association_df.sort_values(["scale", "target", "channel"]).iterrows():
        lines.append(f"{row['scale']}, {channel_label(row['channel'])}, {row['target_label']}: {row['metric']}={float(row['value']):.6g}")
    lines.extend(["", "Figure B: per-transition association"])
    for _, row in step_vt_df.iterrows():
        lines.append(f"{channel_label(row['channel'])}: Spearman rho={float(row['spearman']):.6g}, n={int(row['n_steps'])}")
    lines.extend(["", "Figure C: manipulation check condition means"])
    for _, row in summary_df.sort_values("mu").iterrows():
        lines.append(f"$\\mu$={float(row['mu']):.6g}:")
        for col, label in [
            ("V_T_mean", "$V_T$ mean"),
            ("Delta_rep_T_mean", "$\\Delta_T^{\\mathrm{rep}}$ mean"),
            ("A_rate_null_mean", "Null blind raw observed FR mean"),
            ("A_rate_coarse_mean", "Coarse score raw observed FR mean"),
            ("A_rate_task_mean", "Task-aligned raw observed FR mean"),
        ]:
            lines.append(f"  {label}: {float(row[col]):.10g}")
        lines.append("")
    (OUTPUT_DIR / "main_figure_values_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def cleanup_output_dir() -> None:
    keep_names = {
        "regional_feedback_results_by_seed.csv",
        "regional_feedback_rounds.csv",
        "regional_feedback_summary.csv",
        "regional_feedback_step_diagnostics.csv",
        "regional_feedback_raw_rate_regressions.csv",
        "table_channel_association_summary.csv",
        "main_figure_values_summary.txt",
        "figure_main_channel_association_summary.pdf",
        "figure_main_channel_association_summary.png",
        "figure_main_channel_association_summary.svg",
        "figure_main_step_fr_association.pdf",
        "figure_main_step_fr_association.png",
        "figure_main_step_fr_association.svg",
        "figure_main_feedback_manipulation_check.pdf",
        "figure_main_feedback_manipulation_check.png",
        "figure_main_feedback_manipulation_check.svg",
    }
    if (OUTPUT_DIR / "appendix_diagnostics").exists():
        shutil.rmtree(OUTPUT_DIR / "appendix_diagnostics")
    for path in OUTPUT_DIR.iterdir():
        if path.is_file() and path.name not in keep_names:
            path.unlink()


# Main

def main() -> None:
    df0 = load_and_prepare_base_dataframe(DATA_PATH)
    print(f"Loaded dataframe with shape: {df0.shape}")
    ridge_alpha = choose_ridge_alpha(df0)
    print(f"Chosen Ridge alpha: {ridge_alpha}")

    baseline_model = fit_model(df0, alpha=ridge_alpha)
    artifacts = fit_channel_artifacts(df0, predict_model(baseline_model, df0))
    scales = estimate_feedback_scales(df0)
    print(
        "Feedback scales: "
        f"q_sd={scales.q_sd:.4f}, c_sd={scales.c_sd:.4f}, y_sd={scales.y_sd:.4f}; "
        f"sigma_q={scales.sigma_q:.4f}, sigma_c={scales.sigma_c:.4f}, sigma_y={scales.sigma_y:.4f}"
    )

    results = []
    round_records = []
    for mu in MU_GRID:
        for seed in range(N_SEEDS):
            print(f"Running mu={mu:.3f}, seed={seed}")
            results.append(
                run_single_condition(
                    df0=df0,
                    artifacts=artifacts,
                    mu=mu,
                    seed=seed,
                    scales=scales,
                    ridge_alpha=ridge_alpha,
                    round_records_sink=round_records,
                )
            )

    results_df = pd.DataFrame(results)
    round_df = pd.DataFrame(round_records)
    summary_df = summarize_results(results_df)
    step_diagnostics_df = build_step_diagnostics_table(round_df)
    raw_rate_regression_df = fit_raw_rate_regression_table(results_df)

    results_df.to_csv(OUTPUT_DIR / "regional_feedback_results_by_seed.csv", index=False)
    round_df.to_csv(OUTPUT_DIR / "regional_feedback_rounds.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "regional_feedback_summary.csv", index=False)
    step_diagnostics_df.to_csv(OUTPUT_DIR / "regional_feedback_step_diagnostics.csv", index=False)
    raw_rate_regression_df.to_csv(OUTPUT_DIR / "regional_feedback_raw_rate_regressions.csv", index=False)

    association_df = plot_main_channel_association_summary(raw_rate_regression_df, step_diagnostics_df)
    association_df.to_csv(OUTPUT_DIR / "table_channel_association_summary.csv", index=False)
    step_vt_df = plot_main_step_fr_association(step_diagnostics_df)
    plot_main_feedback_manipulation_check(summary_df, results_df["seed"].nunique())
    write_main_figure_values_summary(association_df, step_vt_df, summary_df)
    cleanup_output_dir()

    print("\nSaved main-figure outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
