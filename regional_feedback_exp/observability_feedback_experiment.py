from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from dataclasses import dataclass

_MPLCONFIGDIR = Path(os.environ.get("TMPDIR", "/tmp")) / "regional_feedback_matplotlib_cache"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

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

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
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
APPENDIX_OUTPUT_DIR = OUTPUT_DIR / "appendix_diagnostics"
APPENDIX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Unit Price"
QUANTITY_COL = "Order Quantity"
COST_COL = "Unit Cost"
GROUP_COL = "Sales Channel"

N_ROUNDS = 20
N_SEEDS = 10
TEST_SIZE = 0.2
MU_GRID = [0.00, 0.05, 0.10, 0.20]
BURN_IN_RANDOM_OFFSET = 100_000
EXCLUDE_INITIAL_FEEDBACK_ROUNDS = 3
RIDGE_ALPHA = 1.0
RIDGE_ALPHA_GRID = [0.1, 1.0, 10.0, 100.0]

RHO = 0.5
B_C = 0.4
B_Q = 0.3

NOISE_SCALE_Q = 0.02
NOISE_SCALE_C = 0.02
NOISE_SCALE_Y = 0.02

COARSE_SCORE_BINS = 5
NULL_RANDOM_BUCKETS = 8
TASK_SCORE_BINS = 4
TASK_QTY_BINS = 4
TASK_COST_BINS = 4
PSEUDOCOUNT = 1e-6
MIN_POSITIVE_VALUE = 1e-3
EPS = 1e-8
FIGSIZE = (3.4, 2.4)
SAVEFIG_KW = {}
N_BOOTSTRAP = 10000
PAIRWISE_TEST_SEED = 20260505

CHANNEL_ORDER = ["null", "coarse", "task_no_qty", "task_no_cost", "task_no_group", "task"]
CHANNEL_LABELS = {
    "null": "Null blind",
    "coarse": "Coarse score",
    "task_no_qty": "Task minus quantity",
    "task_no_cost": "Task minus cost",
    "task_no_group": "Task minus subgroup",
    "task": "Task-aligned",
}
TARGET_LABELS = {
    "V_T": r"$V_T$",
    "Delta_rep_T": r"$\Delta_T^{\mathrm{rep}}$",
    "Delta_sam_T": r"$\Delta_T^{\mathrm{sam}}$",
    "v_t": r"$v_t$",
    "Eval_within_step_drift_T": "held-out step shift",
    "eval_within_step_drift": "held-out step shift",
    "Mean_abs_delta_loss_T": "pointwise loss motion",
    "mean_abs_delta_loss": "pointwise loss motion",
    "Mean_abs_delta_error_T": "pointwise error motion",
    "mean_abs_delta_error": "pointwise error motion",
    "Mean_abs_delta_y_T": "target motion",
    "mean_abs_delta_y": "target motion",
    "Mean_abs_delta_pred_T": "prediction motion",
    "mean_abs_delta_pred": "prediction motion",
}
CHANNEL_DEFINITIONS = {
    "null": "Fixed deterministic row/hash buckets only; no predictions or task variables.",
    "coarse": "Prediction-score bins only, with bin edges fixed from the baseline D_0 predictions.",
    "task_no_qty": "Prediction-score bins x Unit Cost bins x Sales Channel subgroup; omits Order Quantity.",
    "task_no_cost": "Prediction-score bins x Order Quantity bins x Sales Channel subgroup; omits Unit Cost.",
    "task_no_group": "Prediction-score bins x Order Quantity bins x Unit Cost bins; omits Sales Channel subgroup.",
    "task": "Prediction-score bins x Order Quantity bins x Unit Cost bins x Sales Channel subgroup.",
}


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
    task_cost_edges: np.ndarray
    subgroup_ref: str
    sales_channel_categories: list[str]


def fit_channel_artifacts(df0: pd.DataFrame, preds0: np.ndarray) -> ChannelArtifacts:
    coarse_score_edges = compute_quantile_edges(preds0, COARSE_SCORE_BINS)
    task_score_edges = compute_quantile_edges(preds0, TASK_SCORE_BINS)
    task_qty_edges = compute_quantile_edges(df0[QUANTITY_COL].to_numpy(), TASK_QTY_BINS)
    task_cost_edges = compute_quantile_edges(df0[COST_COL].to_numpy(), TASK_COST_BINS)

    mode_value = df0[GROUP_COL].mode(dropna=True)
    subgroup_ref = str(mode_value.iloc[0] if len(mode_value) else df0[GROUP_COL].iloc[0])
    sales_channel_categories = sorted(df0[GROUP_COL].astype(str).unique())

    return ChannelArtifacts(
        coarse_score_edges=coarse_score_edges,
        task_score_edges=task_score_edges,
        task_qty_edges=task_qty_edges,
        task_cost_edges=task_cost_edges,
        subgroup_ref=subgroup_ref,
        sales_channel_categories=sales_channel_categories,
    )


def fixed_random_buckets(n_rows: int, n_buckets: int) -> np.ndarray:
    row_ids = np.arange(n_rows, dtype=np.uint64)
    hashed = row_ids * np.uint64(11400714819323198485) + np.uint64(0x9E3779B97F4A7C15)
    return np.asarray(hashed % np.uint64(n_buckets), dtype=int)


def null_channel_probs(df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    del preds, artifacts
    state_ids = fixed_random_buckets(len(df), NULL_RANDOM_BUCKETS)
    return empirical_categorical_distribution(state_ids, NULL_RANDOM_BUCKETS, PSEUDOCOUNT)


def coarse_channel_probs(preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    state_ids = assign_bins(preds, artifacts.coarse_score_edges)
    n_states = len(artifacts.coarse_score_edges) - 1
    return empirical_categorical_distribution(state_ids, n_states, PSEUDOCOUNT)


def task_channel_probs(df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    score_bin = assign_bins(preds, artifacts.coarse_score_edges)
    qty_bin = assign_bins(df[QUANTITY_COL].to_numpy(), artifacts.task_qty_edges)
    cost_bin = assign_bins(df[COST_COL].to_numpy(), artifacts.task_cost_edges)
    group_bin = (df[GROUP_COL].astype(str).to_numpy() != artifacts.subgroup_ref).astype(int)

    n_score = len(artifacts.coarse_score_edges) - 1
    n_qty = len(artifacts.task_qty_edges) - 1
    n_cost = len(artifacts.task_cost_edges) - 1
    n_group = 2

    state_ids = (
        score_bin * (n_qty * n_cost * n_group)
        + qty_bin * (n_cost * n_group)
        + cost_bin * n_group
        + group_bin
    )
    n_states = n_score * n_qty * n_cost * n_group
    return empirical_categorical_distribution(state_ids, n_states, PSEUDOCOUNT)


def task_no_qty_channel_probs(df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    score_bin = assign_bins(preds, artifacts.coarse_score_edges)
    cost_bin = assign_bins(df[COST_COL].to_numpy(), artifacts.task_cost_edges)
    group_bin = (df[GROUP_COL].astype(str).to_numpy() != artifacts.subgroup_ref).astype(int)

    n_score = len(artifacts.coarse_score_edges) - 1
    n_cost = len(artifacts.task_cost_edges) - 1
    n_group = 2

    state_ids = score_bin * (n_cost * n_group) + cost_bin * n_group + group_bin
    n_states = n_score * n_cost * n_group
    return empirical_categorical_distribution(state_ids, n_states, PSEUDOCOUNT)


def task_no_cost_channel_probs(df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    score_bin = assign_bins(preds, artifacts.coarse_score_edges)
    qty_bin = assign_bins(df[QUANTITY_COL].to_numpy(), artifacts.task_qty_edges)
    group_bin = (df[GROUP_COL].astype(str).to_numpy() != artifacts.subgroup_ref).astype(int)

    n_score = len(artifacts.coarse_score_edges) - 1
    n_qty = len(artifacts.task_qty_edges) - 1
    n_group = 2

    state_ids = score_bin * (n_qty * n_group) + qty_bin * n_group + group_bin
    n_states = n_score * n_qty * n_group
    return empirical_categorical_distribution(state_ids, n_states, PSEUDOCOUNT)


def task_no_group_channel_probs(df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    score_bin = assign_bins(preds, artifacts.coarse_score_edges)
    qty_bin = assign_bins(df[QUANTITY_COL].to_numpy(), artifacts.task_qty_edges)
    cost_bin = assign_bins(df[COST_COL].to_numpy(), artifacts.task_cost_edges)

    n_score = len(artifacts.coarse_score_edges) - 1
    n_qty = len(artifacts.task_qty_edges) - 1
    n_cost = len(artifacts.task_cost_edges) - 1

    state_ids = score_bin * (n_qty * n_cost) + qty_bin * n_cost + cost_bin
    n_states = n_score * n_qty * n_cost
    return empirical_categorical_distribution(state_ids, n_states, PSEUDOCOUNT)


CHANNEL_PROB_FUNCTIONS = {
    "null": null_channel_probs,
    "coarse": coarse_channel_probs,
    "task_no_qty": task_no_qty_channel_probs,
    "task_no_cost": task_no_cost_channel_probs,
    "task_no_group": task_no_group_channel_probs,
    "task": task_channel_probs,
}


def channel_probs(channel: str, df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    if channel == "coarse":
        return coarse_channel_probs(preds, artifacts)
    return CHANNEL_PROB_FUNCTIONS[channel](df, preds, artifacts)


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
    df_next = df_t.reset_index(drop=True).copy()
    df_t = df_t.reset_index(drop=True)

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

    return df_next.reset_index(drop=True)


# -----------------------------
# Simulation
# -----------------------------


def split_with_row_ids(df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_df = df.reset_index(drop=True).copy()
    split_df["_row_id"] = np.arange(len(split_df), dtype=int)
    train_df, eval_df = train_test_split(split_df, test_size=TEST_SIZE, random_state=random_state)
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)
    train_model_df = train_df.drop(columns=["_row_id"])
    eval_model_df = eval_df.drop(columns=["_row_id"])
    return train_df, eval_df, train_model_df, eval_model_df


def burn_in_learner(df0: pd.DataFrame, seed: int, ridge_alpha: float) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train_df, eval_df, train_model_df, eval_model_df = split_with_row_ids(
        df0,
        random_state=seed + BURN_IN_RANDOM_OFFSET,
    )
    y_train = train_model_df[TARGET_COL].to_numpy(dtype=float)
    y_eval = eval_model_df[TARGET_COL].to_numpy(dtype=float)
    y_all = df0[TARGET_COL].to_numpy(dtype=float)

    mean_pred = np.full_like(y_eval, fill_value=float(np.mean(y_train)), dtype=float)
    initial_heldout_mse = float(mean_squared_error(y_eval, mean_pred))
    initial_heldout_rmse = rmse(y_eval, mean_pred)

    model = fit_model(train_model_df, alpha=ridge_alpha)
    eval_preds = predict_model(model, eval_model_df)
    all_preds = predict_model(model, df0)
    after_heldout_mse = float(mean_squared_error(y_eval, eval_preds))
    after_population_mse = float(mean_squared_error(y_all, all_preds))
    after_heldout_rmse = rmse(y_eval, eval_preds)
    after_population_rmse = rmse(y_all, all_preds)
    relative_improvement = (initial_heldout_rmse - after_heldout_rmse) / max(initial_heldout_rmse, EPS)

    diagnostics = {
        "seed": seed,
        "burn_in_initial_predictor": "train_mean_baseline",
        "burn_in_train_rows": int(len(train_model_df)),
        "burn_in_eval_rows": int(len(eval_model_df)),
        "burn_in_initial_heldout_mse": initial_heldout_mse,
        "burn_in_initial_heldout_rmse": initial_heldout_rmse,
        "burn_in_after_heldout_mse": after_heldout_mse,
        "burn_in_after_heldout_rmse": after_heldout_rmse,
        "burn_in_after_population_mse": after_population_mse,
        "burn_in_after_population_rmse": after_population_rmse,
        "burn_in_relative_rmse_improvement": float(relative_improvement),
    }
    return model, train_df, eval_df, eval_model_df, diagnostics


def aggregate_round_metrics(rr: pd.DataFrame, *, suffix: str = "") -> dict:
    if rr.empty:
        return {
            f"Rhat_T{suffix}": float("nan"),
            f"Rplus_T{suffix}": float("nan"),
            f"Rtraj_T{suffix}": float("nan"),
            f"Delta_rep_T{suffix}": float("nan"),
            f"Delta_sam_T{suffix}": float("nan"),
            f"V_T{suffix}": float("nan"),
            f"Eval_within_step_drift_T{suffix}": float("nan"),
            f"Mean_abs_delta_y_T{suffix}": float("nan"),
            f"Mean_abs_delta_pred_T{suffix}": float("nan"),
            f"Mean_abs_delta_error_T{suffix}": float("nan"),
            f"Mean_abs_delta_loss_T{suffix}": float("nan"),
        }

    rhat_t = float(rr["empirical_loss"].mean())
    rplus_t = float(rr["preq_target"].mean())
    rtraj_t = float(rr["same_time_target"].mean())
    return {
        f"Rhat_T{suffix}": rhat_t,
        f"Rplus_T{suffix}": rplus_t,
        f"Rtraj_T{suffix}": rtraj_t,
        f"Delta_rep_T{suffix}": abs(rhat_t - rplus_t),
        f"Delta_sam_T{suffix}": abs(rhat_t - rtraj_t),
        f"V_T{suffix}": float(rr["v_t"].mean()),
        f"Eval_within_step_drift_T{suffix}": float(rr["eval_within_step_drift"].mean()),
        f"Mean_abs_delta_y_T{suffix}": float(rr["mean_abs_delta_y"].mean()),
        f"Mean_abs_delta_pred_T{suffix}": float(rr["mean_abs_delta_pred"].mean()),
        f"Mean_abs_delta_error_T{suffix}": float(rr["mean_abs_delta_error"].mean()),
        f"Mean_abs_delta_loss_T{suffix}": float(rr["mean_abs_delta_loss"].mean()),
    }


def run_single_condition(
    df0: pd.DataFrame,
    artifacts: ChannelArtifacts,
    mu: float,
    seed: int,
    scales: FeedbackScales,
    ridge_alpha: float,
    round_records_outpath: Path | None = None,
    round_records_sink: list[dict] | None = None,
    burn_in_records_sink: list[dict] | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    df_current = df0.reset_index(drop=True).copy()
    model, train_df, eval_df, eval_model_df, burn_in_diagnostics = burn_in_learner(df0, seed, ridge_alpha)
    burn_in_diagnostics["mu"] = mu
    if burn_in_records_sink is not None:
        burn_in_records_sink.append(burn_in_diagnostics)

    round_records: list[dict] = []

    for t in range(N_ROUNDS - 1):
        preds_current_all = predict_model(model, df_current)
        preds_train_current = predict_model(model, train_df.drop(columns=["_row_id"]))
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
        preds_next_all_same_model = predict_model(model, df_next)

        y_current_all = df_current[TARGET_COL].to_numpy(dtype=float)
        y_train_current = train_df[TARGET_COL].to_numpy(dtype=float)
        y_eval_current = eval_df[TARGET_COL].to_numpy(dtype=float)
        y_next_all = df_next[TARGET_COL].to_numpy(dtype=float)
        y_eval_next = next_eval_df[TARGET_COL].to_numpy(dtype=float)

        error_current_all = y_current_all - preds_current_all
        error_next_all = y_next_all - preds_next_all_same_model
        loss_current_all = error_current_all**2
        loss_next_all = error_next_all**2
        mean_abs_delta_y = float(np.mean(np.abs(y_next_all - y_current_all)))
        mean_abs_delta_pred = float(np.mean(np.abs(preds_next_all_same_model - preds_current_all)))
        mean_abs_delta_error = float(np.mean(np.abs(error_next_all - error_current_all)))
        mean_abs_delta_loss = float(np.mean(np.abs(loss_next_all - loss_current_all)))

        train_mse = float(mean_squared_error(y_train_current, preds_train_current))
        current_all_mse = float(np.mean(loss_current_all))
        next_all_mse = float(np.mean(loss_next_all))
        eval_mse_current = float(mean_squared_error(y_eval_current, preds_eval_current))
        eval_mse_next = float(mean_squared_error(y_eval_next, preds_eval_next))
        current_rmse = float(np.sqrt(eval_mse_current))
        next_rmse = float(np.sqrt(eval_mse_next))

        # Trajectory components for the paper's gaps. The empirical loss is
        # held out on D_t; the two population targets use the full finite
        # dataset as the experiment's population proxy.
        empirical_loss = eval_mse_current
        same_time_target = current_all_mse
        preq_target = next_all_mse

        # V_T: full-dataset within-step drift under the deployed predictor f_t.
        v_t = abs(current_all_mse - next_all_mse)
        # Held-out diagnostic only. This is a noisy eval-split version of the
        # within-step drift term, not Delta_rep_T.
        eval_within_step_drift = abs(eval_mse_current - eval_mse_next)
        # Internal diagnostic only, not a primary experiment quantity:
        # sampling_candidate = |MSE_D_t_all(f_t) - MSE_D_t_eval(f_t)|.
        # This is same-round, but MSE_D_t_all includes train rows used to fit f_t.
        sampling_candidate = abs(current_all_mse - eval_mse_current)

        # Observable channel probabilities and FR step lengths.
        fr_steps = {}
        for channel in CHANNEL_ORDER:
            p = channel_probs(channel, df_current, preds_current_all, artifacts)
            q = channel_probs(channel, df_next, preds_next_all_same_model, artifacts)
            fr_steps[channel] = fisher_rao_categorical(p, q)

        round_records.append(
            {
                "round": t,
                "mu": mu,
                "seed": seed,
                "eval_mse_current": eval_mse_current,
                "current_all_mse": current_all_mse,
                "next_all_mse": next_all_mse,
                "eval_mse_next": eval_mse_next,
                "train_mse": train_mse,
                "current_rmse": current_rmse,
                "next_rmse": next_rmse,
                "empirical_loss": empirical_loss,
                "same_time_target": same_time_target,
                "preq_target": preq_target,
                "v_t": v_t,
                "eval_within_step_drift": eval_within_step_drift,
                "sampling_candidate": sampling_candidate,
                "mean_abs_delta_y": mean_abs_delta_y,
                "mean_abs_delta_pred": mean_abs_delta_pred,
                "mean_abs_delta_error": mean_abs_delta_error,
                "mean_abs_delta_loss": mean_abs_delta_loss,
                **{f"fr_step_{channel}": fr_steps[channel] for channel in CHANNEL_ORDER},
            }
        )

        df_current = df_next
        if t < N_ROUNDS - 2:
            train_df, eval_df, train_model_df, eval_model_df = split_with_row_ids(
                df_current,
                random_state=seed + 1000 * (t + 1),
            )
            model = fit_model(train_model_df, alpha=ridge_alpha)

    # Aggregate
    rr = pd.DataFrame(round_records)
    if round_records_outpath is not None:
        round_records_outpath.parent.mkdir(parents=True, exist_ok=True)
        round_cols = [
            "round",
            "mu",
            "seed",
            "train_mse",
            "current_all_mse",
            "next_all_mse",
            "eval_mse_current",
            "eval_mse_next",
            "empirical_loss",
            "same_time_target",
            "preq_target",
            "v_t",
            "eval_within_step_drift",
            "sampling_candidate",
            "mean_abs_delta_y",
            "mean_abs_delta_pred",
            "mean_abs_delta_error",
            "mean_abs_delta_loss",
            *[f"fr_step_{channel}" for channel in CHANNEL_ORDER],
        ]
        rr[round_cols].to_csv(round_records_outpath, index=False)
    if round_records_sink is not None:
        round_records_sink.extend(round_records)

    metrics = aggregate_round_metrics(rr)
    sensitivity_metrics = aggregate_round_metrics(
        rr[rr["round"] >= EXCLUDE_INITIAL_FEEDBACK_ROUNDS],
        suffix=f"_excl_first{EXCLUDE_INITIAL_FEEDBACK_ROUNDS}",
    )

    return {
        "mu": mu,
        "seed": seed,
        **burn_in_diagnostics,
        **metrics,
        **sensitivity_metrics,
        "mean_current_rmse": float(rr["current_rmse"].mean()),
        "mean_next_rmse": float(rr["next_rmse"].mean()),
        **{f"A_T_{channel}": float(rr[f"fr_step_{channel}"].sum()) for channel in CHANNEL_ORDER},
        **{f"A_rate_{channel}": float(rr[f"fr_step_{channel}"].mean()) for channel in CHANNEL_ORDER},
    }


# -----------------------------
# Plotting
# -----------------------------


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    sensitivity_suffix = f"_excl_first{EXCLUDE_INITIAL_FEEDBACK_ROUNDS}"
    agg_spec = {
        "burn_in_initial_heldout_mse_mean": ("burn_in_initial_heldout_mse", "mean"),
        "burn_in_initial_heldout_mse_std": ("burn_in_initial_heldout_mse", "std"),
        "burn_in_initial_heldout_rmse_mean": ("burn_in_initial_heldout_rmse", "mean"),
        "burn_in_initial_heldout_rmse_std": ("burn_in_initial_heldout_rmse", "std"),
        "burn_in_after_heldout_mse_mean": ("burn_in_after_heldout_mse", "mean"),
        "burn_in_after_heldout_mse_std": ("burn_in_after_heldout_mse", "std"),
        "burn_in_after_heldout_rmse_mean": ("burn_in_after_heldout_rmse", "mean"),
        "burn_in_after_heldout_rmse_std": ("burn_in_after_heldout_rmse", "std"),
        "burn_in_after_population_mse_mean": ("burn_in_after_population_mse", "mean"),
        "burn_in_after_population_mse_std": ("burn_in_after_population_mse", "std"),
        "burn_in_after_population_rmse_mean": ("burn_in_after_population_rmse", "mean"),
        "burn_in_after_population_rmse_std": ("burn_in_after_population_rmse", "std"),
        "burn_in_relative_rmse_improvement_mean": ("burn_in_relative_rmse_improvement", "mean"),
        "burn_in_relative_rmse_improvement_std": ("burn_in_relative_rmse_improvement", "std"),
        "Rhat_T_mean": ("Rhat_T", "mean"),
        "Rhat_T_std": ("Rhat_T", "std"),
        "Rplus_T_mean": ("Rplus_T", "mean"),
        "Rplus_T_std": ("Rplus_T", "std"),
        "Rtraj_T_mean": ("Rtraj_T", "mean"),
        "Rtraj_T_std": ("Rtraj_T", "std"),
        "Delta_rep_T_mean": ("Delta_rep_T", "mean"),
        "Delta_rep_T_std": ("Delta_rep_T", "std"),
        "Delta_sam_T_mean": ("Delta_sam_T", "mean"),
        "Delta_sam_T_std": ("Delta_sam_T", "std"),
        "V_T_mean": ("V_T", "mean"),
        "V_T_std": ("V_T", "std"),
        "Eval_within_step_drift_T_mean": ("Eval_within_step_drift_T", "mean"),
        "Eval_within_step_drift_T_std": ("Eval_within_step_drift_T", "std"),
        "Mean_abs_delta_y_T_mean": ("Mean_abs_delta_y_T", "mean"),
        "Mean_abs_delta_y_T_std": ("Mean_abs_delta_y_T", "std"),
        "Mean_abs_delta_pred_T_mean": ("Mean_abs_delta_pred_T", "mean"),
        "Mean_abs_delta_pred_T_std": ("Mean_abs_delta_pred_T", "std"),
        "Mean_abs_delta_error_T_mean": ("Mean_abs_delta_error_T", "mean"),
        "Mean_abs_delta_error_T_std": ("Mean_abs_delta_error_T", "std"),
        "Mean_abs_delta_loss_T_mean": ("Mean_abs_delta_loss_T", "mean"),
        "Mean_abs_delta_loss_T_std": ("Mean_abs_delta_loss_T", "std"),
        f"Delta_rep_T{sensitivity_suffix}_mean": (f"Delta_rep_T{sensitivity_suffix}", "mean"),
        f"Delta_rep_T{sensitivity_suffix}_std": (f"Delta_rep_T{sensitivity_suffix}", "std"),
        f"Delta_sam_T{sensitivity_suffix}_mean": (f"Delta_sam_T{sensitivity_suffix}", "mean"),
        f"Delta_sam_T{sensitivity_suffix}_std": (f"Delta_sam_T{sensitivity_suffix}", "std"),
        f"V_T{sensitivity_suffix}_mean": (f"V_T{sensitivity_suffix}", "mean"),
        f"V_T{sensitivity_suffix}_std": (f"V_T{sensitivity_suffix}", "std"),
        f"Eval_within_step_drift_T{sensitivity_suffix}_mean": (f"Eval_within_step_drift_T{sensitivity_suffix}", "mean"),
        f"Eval_within_step_drift_T{sensitivity_suffix}_std": (f"Eval_within_step_drift_T{sensitivity_suffix}", "std"),
        f"Mean_abs_delta_y_T{sensitivity_suffix}_mean": (f"Mean_abs_delta_y_T{sensitivity_suffix}", "mean"),
        f"Mean_abs_delta_y_T{sensitivity_suffix}_std": (f"Mean_abs_delta_y_T{sensitivity_suffix}", "std"),
        f"Mean_abs_delta_pred_T{sensitivity_suffix}_mean": (f"Mean_abs_delta_pred_T{sensitivity_suffix}", "mean"),
        f"Mean_abs_delta_pred_T{sensitivity_suffix}_std": (f"Mean_abs_delta_pred_T{sensitivity_suffix}", "std"),
        f"Mean_abs_delta_error_T{sensitivity_suffix}_mean": (f"Mean_abs_delta_error_T{sensitivity_suffix}", "mean"),
        f"Mean_abs_delta_error_T{sensitivity_suffix}_std": (f"Mean_abs_delta_error_T{sensitivity_suffix}", "std"),
        f"Mean_abs_delta_loss_T{sensitivity_suffix}_mean": (f"Mean_abs_delta_loss_T{sensitivity_suffix}", "mean"),
        f"Mean_abs_delta_loss_T{sensitivity_suffix}_std": (f"Mean_abs_delta_loss_T{sensitivity_suffix}", "std"),
    }
    for channel in CHANNEL_ORDER:
        agg_spec[f"A_rate_{channel}_mean"] = (f"A_rate_{channel}", "mean")
        agg_spec[f"A_rate_{channel}_std"] = (f"A_rate_{channel}", "std")
        agg_spec[f"A_rate_{channel}_excess_mean"] = (f"A_rate_{channel}_excess", "mean")
        agg_spec[f"A_rate_{channel}_excess_std"] = (f"A_rate_{channel}_excess", "std")
    summary = results_df.groupby("mu").agg(**agg_spec).reset_index()
    return summary


def build_early_round_sensitivity_table(results_df: pd.DataFrame) -> pd.DataFrame:
    suffix = f"_excl_first{EXCLUDE_INITIAL_FEEDBACK_ROUNDS}"
    rows = []
    for target in [
        "Delta_rep_T",
        "Delta_sam_T",
        "V_T",
        "Eval_within_step_drift_T",
        "Mean_abs_delta_y_T",
        "Mean_abs_delta_pred_T",
        "Mean_abs_delta_error_T",
        "Mean_abs_delta_loss_T",
    ]:
        sensitivity_col = f"{target}{suffix}"
        if target not in results_df.columns or sensitivity_col not in results_df.columns:
            continue
        for _, row in results_df.iterrows():
            included = float(row[target])
            excluded = float(row[sensitivity_col])
            rows.append(
                {
                    "mu": float(row["mu"]),
                    "seed": int(row["seed"]),
                    "target": target,
                    "excluded_initial_rounds": EXCLUDE_INITIAL_FEEDBACK_ROUNDS,
                    "included_all_rounds": included,
                    "excluding_initial_rounds": excluded,
                    "excluded_minus_included": excluded - included,
                    "excluded_over_included": excluded / included if abs(included) > EPS else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_step_diagnostics_table(round_df: pd.DataFrame) -> pd.DataFrame:
    if round_df.empty:
        return pd.DataFrame()

    targets = [
        ("v_t", "within_step_risk_shift"),
        ("eval_within_step_drift", "diagnostic_heldout_step_shift"),
        ("mean_abs_delta_y", "target_motion"),
        ("mean_abs_delta_pred", "prediction_motion"),
        ("mean_abs_delta_error", "unsigned_pointwise_error_motion"),
        ("mean_abs_delta_loss", "unsigned_pointwise_loss_motion"),
    ]
    channels = sorted(
        col[len("fr_step_") :]
        for col in round_df.columns
        if col.startswith("fr_step_")
    )

    rows = []
    grouped = round_df[round_df["mu"] > 0.0].groupby("mu", sort=True)
    for mu, mu_df in grouped:
        for target, target_role in targets:
            if target not in mu_df.columns:
                continue
            for channel in channels:
                predictor = f"fr_step_{channel}"
                model_df = mu_df[[target, predictor]].dropna()
                if len(model_df) < 2:
                    continue
                rows.append(
                    {
                        "mu": float(mu),
                        "target": target,
                        "target_role": target_role,
                        "channel": channel,
                        "predictor": predictor,
                        "n_steps": int(len(model_df)),
                        "pearson": safe_corr(model_df[predictor], model_df[target], method="pearson"),
                        "spearman": safe_corr(model_df[predictor], model_df[target], method="spearman"),
                        "slope": linear_slope(model_df[predictor], model_df[target]),
                    }
                )

    all_positive = round_df[round_df["mu"] > 0.0]
    for target, target_role in targets:
        if target not in all_positive.columns:
            continue
        for channel in channels:
            predictor = f"fr_step_{channel}"
            model_df = all_positive[[target, predictor]].dropna()
            if len(model_df) < 2:
                continue
            rows.append(
                {
                    "mu": "all_positive",
                    "target": target,
                    "target_role": target_role,
                    "channel": channel,
                    "predictor": predictor,
                    "n_steps": int(len(model_df)),
                    "pearson": safe_corr(model_df[predictor], model_df[target], method="pearson"),
                    "spearman": safe_corr(model_df[predictor], model_df[target], method="spearman"),
                    "slope": linear_slope(model_df[predictor], model_df[target]),
                }
            )

    return pd.DataFrame(rows)


def save_figure_all_formats(fig, fig_path: Path) -> None:
    fig.savefig(fig_path.with_suffix(".pdf"), **SAVEFIG_KW)
    fig.savefig(fig_path.with_suffix(".svg"), **SAVEFIG_KW)
    fig.savefig(fig_path.with_suffix(".png"), dpi=600, **SAVEFIG_KW)
    plt.close(fig)


def standard_error(std_values: pd.Series, n: int) -> pd.Series:
    return std_values / math.sqrt(n)


def style_axis(ax) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.6)


def plot_mean_band(x, y, yerr, xlabel, ylabel, outpath, label=None, marker="o", color=None):
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    fig.set_size_inches(*FIGSIZE)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    yerr_arr = np.asarray(yerr, dtype=float)
    color = PALETTE[0] if color is None else color
    ax.plot(x_arr, y_arr, marker=marker, markersize=3.0, lw=0.9, color=color, label=label)
    ax.fill_between(x_arr, y_arr - yerr_arr, y_arr + yerr_arr, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_axis(ax)
    if label is not None:
        ax.legend(frameon=False, fontsize=7, loc="best")
    save_figure_all_formats(fig, outpath)


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return 0.0
    value = pd.Series(x).corr(pd.Series(y), method=method)
    if pd.isna(value):
        return 0.0
    return float(value)


def discover_rate_channels(results_df: pd.DataFrame, *, suffix: str = "") -> list[str]:
    prefix = "A_rate_"
    channels = []
    for col in results_df.columns:
        if not col.startswith(prefix):
            continue
        if suffix:
            if not col.endswith(suffix):
                continue
            channel = col[len(prefix) : -len(suffix)]
        else:
            if col.endswith("_baseline") or col.endswith("_excess"):
                continue
            channel = col[len(prefix) :]
        channels.append(channel)

    known_order = {channel: idx for idx, channel in enumerate(CHANNEL_ORDER)}
    return sorted(set(channels), key=lambda channel: (known_order.get(channel, len(known_order)), channel))


def add_excess_fr_rates(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    baseline = df[df["mu"] == 0.0].set_index("seed")
    for channel in discover_rate_channels(df):
        rate_col = f"A_rate_{channel}"
        if rate_col not in baseline:
            continue
        baseline_by_seed = baseline[rate_col]
        df[f"{rate_col}_baseline"] = df["seed"].map(baseline_by_seed)
        df[f"{rate_col}_excess"] = df[rate_col] - df[f"{rate_col}_baseline"]
    return df


def coefficient_string(predictors: list[str], coefficients: np.ndarray) -> str:
    return "; ".join(f"{name}={coef:.10g}" for name, coef in zip(predictors, coefficients))


def finite_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total <= EPS:
        return float("nan")
    residual = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - residual / total)


def regression_targets(results_df: pd.DataFrame) -> list[tuple[str, str]]:
    candidates = [
        ("Delta_rep_T", "paper_aligned_prequential_gap"),
        ("V_T", "within_step_population_risk_shift"),
        ("Delta_sam_T", "same_time_sampling_gap"),
        ("Eval_within_step_drift_T", "diagnostic_only_heldout_step_shift"),
    ]
    return [(target, role) for target, role in candidates if target in results_df.columns]


def primary_association_targets(results_df: pd.DataFrame) -> list[tuple[str, str]]:
    candidates = [
        ("V_T", "primary_within_step_population_risk_shift"),
        ("Delta_rep_T", "secondary_prequential_gap_cancellation_sensitive"),
    ]
    return [(target, role) for target, role in candidates if target in results_df.columns]


def operational_raw_rate_targets(results_df: pd.DataFrame) -> list[tuple[str, str]]:
    candidates = [
        ("V_T", "primary_within_step_population_risk_shift"),
        ("Delta_rep_T", "secondary_prequential_gap_cancellation_sensitive"),
        ("Delta_sam_T", "same_time_sampling_gap"),
        ("Eval_within_step_drift_T", "diagnostic_only_heldout_step_shift"),
        ("Mean_abs_delta_loss_T", "unsigned_pointwise_loss_motion"),
        ("Mean_abs_delta_error_T", "unsigned_pointwise_error_motion"),
    ]
    return [(target, role) for target, role in candidates if target in results_df.columns]


def build_regression_specs(results_df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    excess_channels = discover_rate_channels(results_df, suffix="_excess")
    excess_cols = {channel: f"A_rate_{channel}_excess" for channel in excess_channels}
    specs: list[tuple[str, list[str]]] = []

    for channel in excess_channels:
        specs.append((f"single_channel:{channel}", [excess_cols[channel]]))

    for channel in excess_channels:
        specs.append((f"mu_controlled:{channel}", ["mu", excess_cols[channel]]))

    null_col = excess_cols.get("null")
    if null_col is not None:
        for channel in excess_channels:
            if channel == "null":
                continue
            specs.append((f"null_controlled:{channel}", ["mu", excess_cols[channel], null_col]))

    combined_channels = ["task", "coarse", "null"]
    if all(channel in excess_cols for channel in combined_channels):
        specs.append(
            (
                "combined:task_coarse_null",
                ["mu", *(excess_cols[channel] for channel in combined_channels)],
            )
        )

    ablation_channels = ["task", "task_no_qty", "task_no_cost", "task_no_group", "null"]
    if all(channel in excess_cols for channel in ablation_channels):
        specs.append(
            (
                "combined:task_ablation_channels",
                ["mu", *(excess_cols[channel] for channel in ablation_channels)],
            )
        )

    return specs


def fit_linear_regression_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, target_role in regression_targets(results_df):
        for model_name, predictors in build_regression_specs(results_df):
            needed_cols = [target, *predictors]
            model_df = results_df[needed_cols].dropna()
            if len(model_df) == 0:
                continue
            x = model_df[predictors].to_numpy(dtype=float)
            y = model_df[target].to_numpy(dtype=float)
            model = LinearRegression()
            model.fit(x, y)
            preds = model.predict(x)
            rows.append(
                {
                    "target": target,
                    "target_role": target_role,
                    "model_name": model_name,
                    "predictors": " + ".join(predictors),
                    "n": int(len(model_df)),
                    "r2_in_sample": finite_r2(y, preds),
                    "rmse_in_sample": rmse(y, preds),
                    "intercept": float(model.intercept_),
                    "coefficients": coefficient_string(predictors, model.coef_),
                }
            )
    return pd.DataFrame(rows)


def fit_leave_one_mu_out_cv_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mu_values = sorted(results_df["mu"].dropna().unique())
    for target, target_role in regression_targets(results_df):
        for model_name, predictors in build_regression_specs(results_df):
            fold_rmses = []
            fold_r2s = []
            fold_count = 0
            for heldout_mu in mu_values:
                train_df = results_df[results_df["mu"] != heldout_mu][[target, *predictors]].dropna()
                test_df = results_df[results_df["mu"] == heldout_mu][[target, *predictors]].dropna()
                if len(train_df) == 0 or len(test_df) == 0:
                    continue
                model = LinearRegression()
                model.fit(train_df[predictors].to_numpy(dtype=float), train_df[target].to_numpy(dtype=float))
                y_test = test_df[target].to_numpy(dtype=float)
                preds = model.predict(test_df[predictors].to_numpy(dtype=float))
                fold_rmses.append(rmse(y_test, preds))
                fold_r2 = finite_r2(y_test, preds)
                if np.isfinite(fold_r2):
                    fold_r2s.append(fold_r2)
                fold_count += 1

            rows.append(
                {
                    "target": target,
                    "target_role": target_role,
                    "model_name": model_name,
                    "predictors": " + ".join(predictors),
                    "n_folds": int(fold_count),
                    "mean_heldout_rmse": float(np.mean(fold_rmses)) if fold_rmses else float("nan"),
                    "std_heldout_rmse": float(np.std(fold_rmses, ddof=1)) if len(fold_rmses) > 1 else float("nan"),
                    "mean_heldout_r2": float(np.mean(fold_r2s)) if fold_r2s else float("nan"),
                    "std_heldout_r2": float(np.std(fold_r2s, ddof=1)) if len(fold_r2s) > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_regression_diagnostics_table(regression_df: pd.DataFrame, cv_df: pd.DataFrame) -> pd.DataFrame:
    merge_cols = ["target", "target_role", "model_name", "predictors"]
    return cv_df.merge(regression_df, on=merge_cols, how="left")


def bootstrap_within_mu_statistics(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return {
            "slope_ci95_low": float("nan"),
            "slope_ci95_high": float("nan"),
            "pearson_ci95_low": float("nan"),
            "pearson_ci95_high": float("nan"),
            "spearman_ci95_low": float("nan"),
            "spearman_ci95_high": float("nan"),
        }

    slopes = []
    pearsons = []
    spearmans = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        xb = pd.Series(x[idx])
        yb = pd.Series(y[idx])
        if xb.nunique(dropna=True) < 2 or yb.nunique(dropna=True) < 2:
            continue
        slopes.append(linear_slope(xb, yb))
        pearson = xb.corr(yb, method="pearson")
        spearman = xb.corr(yb, method="spearman")
        if np.isfinite(pearson):
            pearsons.append(float(pearson))
        if np.isfinite(spearman):
            spearmans.append(float(spearman))

    def ci(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        low, high = np.percentile(values, [2.5, 97.5])
        return float(low), float(high)

    slope_low, slope_high = ci(slopes)
    pearson_low, pearson_high = ci(pearsons)
    spearman_low, spearman_high = ci(spearmans)
    return {
        "slope_ci95_low": slope_low,
        "slope_ci95_high": slope_high,
        "pearson_ci95_low": pearson_low,
        "pearson_ci95_high": pearson_high,
        "spearman_ci95_low": spearman_low,
        "spearman_ci95_high": spearman_high,
    }


def build_within_mu_association_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(PAIRWISE_TEST_SEED)
    rows = []
    channels = discover_rate_channels(results_df, suffix="_excess")
    for target, target_role in primary_association_targets(results_df):
        for mu, mu_df in results_df[results_df["mu"] > 0.0].groupby("mu", sort=True):
            for channel in channels:
                predictor = f"A_rate_{channel}_excess"
                model_df = mu_df[[target, predictor]].dropna()
                if len(model_df) < 2:
                    continue
                x = model_df[predictor].to_numpy(dtype=float)
                y = model_df[target].to_numpy(dtype=float)
                x_2d = x.reshape(-1, 1)
                model = LinearRegression()
                model.fit(x_2d, y)
                preds = model.predict(x_2d)
                pearson = safe_corr(model_df[predictor], model_df[target], method="pearson")
                spearman = safe_corr(model_df[predictor], model_df[target], method="spearman")
                slope = float(model.coef_[0])
                bootstrap_stats = bootstrap_within_mu_statistics(x, y, rng)
                rows.append(
                    {
                        "target": target,
                        "target_role": target_role,
                        "mu": float(mu),
                        "channel": channel,
                        "predictor": predictor,
                        "n_seeds": int(len(model_df)),
                        "pearson": pearson,
                        "spearman": spearman,
                        "slope": slope,
                        "intercept": float(model.intercept_),
                        "r2_in_sample_within_mu": finite_r2(y, preds),
                        "rmse_in_sample_within_mu": rmse(y, preds),
                        **bootstrap_stats,
                    }
                )
    return pd.DataFrame(rows)


def build_within_mu_channel_summary(within_mu_df: pd.DataFrame) -> pd.DataFrame:
    if within_mu_df.empty:
        return pd.DataFrame()
    ranked = within_mu_df.copy()
    ranked["spearman_rank_within_mu"] = ranked.groupby(["target", "mu"])["spearman"].rank(
        method="average",
        ascending=False,
    )
    summary = (
        ranked.groupby(["target", "target_role", "channel"], sort=False)
        .agg(
            n_mu=("mu", "nunique"),
            mean_spearman=("spearman", "mean"),
            median_spearman=("spearman", "median"),
            fraction_positive_slopes=("slope", lambda values: float(np.mean(np.asarray(values, dtype=float) > 0))),
            fraction_positive_spearman=("spearman", lambda values: float(np.mean(np.asarray(values, dtype=float) > 0))),
            average_rank_across_mu=("spearman_rank_within_mu", "mean"),
            mean_pearson=("pearson", "mean"),
            median_pearson=("pearson", "median"),
            mean_slope=("slope", "mean"),
            median_slope=("slope", "median"),
        )
        .reset_index()
    )
    return summary.sort_values(["target", "average_rank_across_mu", "channel"]).reset_index(drop=True)


def build_condition_mean_trend_table(results_df: pd.DataFrame) -> pd.DataFrame:
    channels = discover_rate_channels(results_df, suffix="_excess")
    rows = []
    mu_means = results_df.groupby("mu", sort=True).mean(numeric_only=True).reset_index()
    for target, target_role in primary_association_targets(results_df):
        for channel in channels:
            predictor = f"A_rate_{channel}_excess"
            if predictor not in mu_means.columns:
                continue
            trend_df = mu_means[["mu", target, predictor]].dropna()
            rows.append(
                {
                    "target": target,
                    "target_role": target_role,
                    "channel": channel,
                    "predictor": predictor,
                    "n_mu": int(len(trend_df)),
                    "mu_values": ";".join(f"{value:.6g}" for value in trend_df["mu"]),
                    "target_means_by_mu": ";".join(f"{value:.10g}" for value in trend_df[target]),
                    "predictor_means_by_mu": ";".join(f"{value:.10g}" for value in trend_df[predictor]),
                    "spearman_mu_with_target_mean": safe_corr(trend_df["mu"], trend_df[target], method="spearman"),
                    "spearman_mu_with_predictor_mean": safe_corr(trend_df["mu"], trend_df[predictor], method="spearman"),
                    "spearman_target_mean_with_predictor_mean": safe_corr(
                        trend_df[predictor],
                        trend_df[target],
                        method="spearman",
                    ),
                    "pearson_target_mean_with_predictor_mean": safe_corr(
                        trend_df[predictor],
                        trend_df[target],
                        method="pearson",
                    ),
                    "slope_target_mean_per_predictor_mean": linear_slope(trend_df[predictor], trend_df[target]),
                }
            )
    return pd.DataFrame(rows)


def fit_raw_rate_regression_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    channels = discover_rate_channels(results_df)
    for target, target_role in operational_raw_rate_targets(results_df):
        for channel in channels:
            predictor = f"A_rate_{channel}"
            model_df = results_df[[target, predictor]].dropna()
            if len(model_df) < 2:
                continue
            x = model_df[[predictor]].to_numpy(dtype=float)
            y = model_df[target].to_numpy(dtype=float)
            model = LinearRegression()
            model.fit(x, y)
            preds = model.predict(x)
            rows.append(
                {
                    "target": target,
                    "target_role": target_role,
                    "model_name": f"raw_rate:{channel}",
                    "channel": channel,
                    "predictor": predictor,
                    "n": int(len(model_df)),
                    "pearson": safe_corr(model_df[predictor], model_df[target], method="pearson"),
                    "spearman": safe_corr(model_df[predictor], model_df[target], method="spearman"),
                    "slope": float(model.coef_[0]),
                    "intercept": float(model.intercept_),
                    "r2_in_sample": finite_r2(y, preds),
                    "rmse_in_sample": rmse(y, preds),
                }
            )
    return pd.DataFrame(rows)


def build_raw_rate_within_mu_association_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    channels = discover_rate_channels(results_df)
    for target, target_role in operational_raw_rate_targets(results_df):
        for mu, mu_df in results_df.groupby("mu", sort=True):
            for channel in channels:
                predictor = f"A_rate_{channel}"
                model_df = mu_df[[target, predictor]].dropna()
                if len(model_df) < 2:
                    continue
                x = model_df[[predictor]].to_numpy(dtype=float)
                y = model_df[target].to_numpy(dtype=float)
                model = LinearRegression()
                model.fit(x, y)
                preds = model.predict(x)
                rows.append(
                    {
                        "target": target,
                        "target_role": target_role,
                        "mu": float(mu),
                        "channel": channel,
                        "predictor": predictor,
                        "n_seeds": int(len(model_df)),
                        "pearson": safe_corr(model_df[predictor], model_df[target], method="pearson"),
                        "spearman": safe_corr(model_df[predictor], model_df[target], method="spearman"),
                        "slope": float(model.coef_[0]),
                        "intercept": float(model.intercept_),
                        "r2_in_sample_within_mu": finite_r2(y, preds),
                        "rmse_in_sample_within_mu": rmse(y, preds),
                    }
                )
    return pd.DataFrame(rows)


def build_raw_rate_condition_mean_trend_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    channels = discover_rate_channels(results_df)
    mu_means = results_df.groupby("mu", sort=True).mean(numeric_only=True).reset_index()
    for target, target_role in operational_raw_rate_targets(results_df):
        for channel in channels:
            predictor = f"A_rate_{channel}"
            if predictor not in mu_means.columns:
                continue
            trend_df = mu_means[["mu", target, predictor]].dropna()
            if len(trend_df) < 2:
                continue
            rows.append(
                {
                    "target": target,
                    "target_role": target_role,
                    "channel": channel,
                    "predictor": predictor,
                    "n_mu": int(len(trend_df)),
                    "mu_values": ";".join(f"{value:.6g}" for value in trend_df["mu"]),
                    "target_means_by_mu": ";".join(f"{value:.10g}" for value in trend_df[target]),
                    "predictor_means_by_mu": ";".join(f"{value:.10g}" for value in trend_df[predictor]),
                    "spearman_mu_with_target_mean": safe_corr(trend_df["mu"], trend_df[target], method="spearman"),
                    "spearman_mu_with_predictor_mean": safe_corr(trend_df["mu"], trend_df[predictor], method="spearman"),
                    "spearman_target_mean_with_predictor_mean": safe_corr(
                        trend_df[predictor],
                        trend_df[target],
                        method="spearman",
                    ),
                    "pearson_target_mean_with_predictor_mean": safe_corr(
                        trend_df[predictor],
                        trend_df[target],
                        method="pearson",
                    ),
                    "slope_target_mean_per_predictor_mean": linear_slope(trend_df[predictor], trend_df[target]),
                }
            )
    return pd.DataFrame(rows)


def build_operational_raw_rate_diagnostics(
    raw_rate_trend_df: pd.DataFrame,
    step_diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    observed_channels = set(raw_rate_trend_df["channel"]).union(set(step_diagnostics_df["channel"]))
    channels = [channel for channel in CHANNEL_ORDER if channel in observed_channels]
    rows = []
    for channel in channels:
        channel_trends = raw_rate_trend_df[raw_rate_trend_df["channel"] == channel]
        channel_steps = step_diagnostics_df[
            (step_diagnostics_df["channel"] == channel)
            & (step_diagnostics_df["mu"].astype(str) == "all_positive")
        ]

        def trend_value(target: str, col: str) -> float:
            row = channel_trends[channel_trends["target"] == target]
            if row.empty or col not in row:
                return float("nan")
            return float(row.iloc[0][col])

        def step_value(target: str, col: str) -> float:
            row = channel_steps[channel_steps["target"] == target]
            if row.empty or col not in row:
                return float("nan")
            return float(row.iloc[0][col])

        rows.append(
            {
                "channel": channel,
                "condition_mean_spearman_V_T": trend_value("V_T", "spearman_target_mean_with_predictor_mean"),
                "condition_mean_spearman_Delta_rep_T": trend_value("Delta_rep_T", "spearman_target_mean_with_predictor_mean"),
                "condition_mean_pearson_V_T": trend_value("V_T", "pearson_target_mean_with_predictor_mean"),
                "condition_mean_pearson_Delta_rep_T": trend_value("Delta_rep_T", "pearson_target_mean_with_predictor_mean"),
                "step_spearman_v_t": step_value("v_t", "spearman"),
                "step_pearson_v_t": step_value("v_t", "pearson"),
                "step_slope_v_t": step_value("v_t", "slope"),
            }
        )

    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return diagnostics

    rank_cols = [
        "condition_mean_spearman_V_T",
        "condition_mean_spearman_Delta_rep_T",
        "step_spearman_v_t",
    ]
    for col in rank_cols:
        diagnostics[f"rank_{col}"] = diagnostics[col].rank(method="average", ascending=False)
    diagnostics["average_operational_rank"] = diagnostics[[f"rank_{col}" for col in rank_cols]].mean(axis=1)
    diagnostics["operational_rank"] = diagnostics["average_operational_rank"].rank(method="dense").astype(int)
    return diagnostics.sort_values(["operational_rank", "channel"]).reset_index(drop=True)


def format_dashboard_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if abs(float(value)) >= 10000 or (abs(float(value)) > 0 and abs(float(value)) < 0.001):
            return f"{float(value):.3e}"
        return f"{float(value):.3f}"
    return str(value)


def channel_label(channel: str) -> str:
    return CHANNEL_LABELS.get(str(channel), str(channel).replace("_", " "))


def target_label(target: str) -> str:
    return TARGET_LABELS.get(str(target), str(target).replace("_", " "))


def display_target_table_name(target: str) -> str:
    return (
        target_label(target)
        .replace("$", "")
        .replace("\\mathrm{rep}", "rep")
        .replace("\\mathrm{sam}", "sam")
        .replace("\\Delta_T^{rep}", "Delta_T^rep")
        .replace("\\Delta_T^{sam}", "Delta_T^sam")
    )


def markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["_No rows._"]
    table = df[columns].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(format_dashboard_value(row[col]) for col in columns) + " |")
    return lines


def best_by_target(df: pd.DataFrame, targets: list[str], *, rank_col: str = "average_rank_across_mu") -> pd.DataFrame:
    rows = []
    for target in targets:
        target_df = df[df["target"] == target]
        if target_df.empty:
            continue
        rows.append(target_df.sort_values([rank_col, "mean_spearman"], ascending=[True, False]).iloc[0])
    return pd.DataFrame(rows)


def write_results_dashboard(
    summary_df: pd.DataFrame,
    operational_raw_rate_df: pd.DataFrame,
    step_diagnostics_df: pd.DataFrame,
    raw_rate_trend_df: pd.DataFrame,
    raw_rate_regression_df: pd.DataFrame,
    outpath: Path,
) -> None:
    step_vt = step_diagnostics_df[
        (step_diagnostics_df["mu"].astype(str) == "all_positive") & (step_diagnostics_df["target"] == "v_t")
    ].copy()
    if not step_vt.empty:
        step_vt = step_vt.sort_values("spearman", ascending=False)
        step_vt_display = step_vt.assign(channel_label=step_vt["channel"].map(channel_label))
    else:
        step_vt_display = pd.DataFrame()

    association_df = build_channel_association_summary_table(raw_rate_regression_df, step_diagnostics_df)
    if not association_df.empty:
        association_display = association_df.copy()
        association_display["channel_label"] = association_display["channel"].map(channel_label)
        association_display["target_label"] = association_display["target"].map(target_label)
        association_display = association_display.sort_values(["scale", "target", "value"], ascending=[True, True, False])
    else:
        association_display = pd.DataFrame()

    lines = [
        "# Regional Feedback Results Dashboard",
        "",
        "Open this file first. The public result is operational observability, not calibrated severity estimation.",
        "",
        "## Purpose",
        "",
        "Raw observed Fisher motion is an uncalibrated but meaningful channel-level footprint of drift-induced risk motion.",
        "The experiment deliberately uses simple fixed monitoring channels rather than optimized residual/error/loss-aware channels.",
        "Raw observed Fisher rates are computed directly from monitoring streams, without using feedback strength as a predictor and without subtracting a matched zero-feedback counterfactual.",
        "",
    ]

    lines.extend(["## Primary Evidence", ""])
    lines.append("1. `figure_main_channel_association_summary.*`: run-level raw-rate $R^2$ and transition-level Spearman $\\rho$ by channel.")
    lines.append("   Channels are compared on their association with the paper-aligned drift quantities: $V_T$, $\\Delta_T^{\\mathrm{rep}}$, and $v_t$.")
    lines.append("2. `figure_main_step_fr_association.*`: per-transition Spearman association between per-step observed FR and $v_t$, including the null blind channel.")
    lines.append("   This shows transition-level observability and channel dependence without using individual-run scatter plots.")
    lines.append("3. `figure_main_feedback_manipulation_check.*`: a secondary manipulation check that feedback strength moves risk quantities and raw observed FR rates.")
    lines.append("")

    lines.extend(["## Main Figure Caption", ""])
    lines.append("Channels are compared on their association with the paper-aligned drift quantities. Panel (a) reports run-level raw-rate $R^2$ for $V_T$ and $\\Delta_T^{\\mathrm{rep}}$. Panel (b) reports transition-level Spearman association with $v_t$. The null blind channel is a fixed row-bucket negative control. Pointwise loss-motion diagnostics are reported separately in the appendix because they are unsigned cancellation-free quantities.")
    lines.append("")

    if not association_display.empty:
        lines.extend(["## Channel Association Highlights", ""])
        lines.append("The null blind channel is a fixed row/hash partition and has no observed Fisher motion. Coarse score is the prediction-score baseline.")
        lines.append("Primary drift quantities are $V_T$ and $v_t$; $\\Delta_T^{\\mathrm{rep}}$ is reported as the paper-aligned prequential gap and is expected to be noisier because it includes sampling deviation and possible cancellation.")
        lines.append("")
        lines.extend(markdown_table(association_display, ["scale", "target_label", "channel_label", "value"]))
        lines.append("")

    if not step_vt_display.empty:
        lines.extend(["## Per-Transition Association", ""])
        lines.append("Bars in `figure_main_step_fr_association.*` report Spearman $\\rho$ over positive-feedback transitions.")
        lines.append("")
        lines.extend(markdown_table(step_vt_display, ["channel_label", "n_steps", "spearman"]))
        lines.append("")

    motion_display = summary_df[
        [
            col
            for col in ["mu", "V_T_mean", "Delta_rep_T_mean", "A_rate_null_mean", "A_rate_coarse_mean", "A_rate_task_mean"]
            if col in summary_df.columns
        ]
    ].copy()
    if not motion_display.empty:
        motion_display = motion_display.rename(
            columns={
                "mu": "$\\mu$",
                "V_T_mean": "$V_T$ mean",
                "Delta_rep_T_mean": "$\\Delta_T^{\\mathrm{rep}}$ mean",
                "A_rate_null_mean": "Null blind raw FR mean",
                "A_rate_coarse_mean": "Coarse score raw FR mean",
                "A_rate_task_mean": "Task-aligned raw FR mean",
            }
        )
        lines.extend(["## Manipulation Check Means", ""])
        lines.append("These means support that the feedback intervention moves the system; they are not the main observability evidence.")
        lines.append("")
        lines.extend(markdown_table(motion_display, list(motion_display.columns)))
        lines.append("")

    lines.extend(["## Interpretation", ""])
    lines.append("The null blind channel is a fixed row-bucket partition and is independent of the feedback-targeted variables. It is a negative control for categorical Fisher motion induced by finite-sample/binning artifacts.")
    lines.append("The coarse score channel is a prediction-score baseline, not a null channel. Task channels add task structure beyond the score baseline.")
    lines.append("Observable FR is a contracted footprint, not an estimate of the full intrinsic $C_T/T$ budget.")
    lines.append("The footprint is channel-dependent and target-dependent; the useful comparison is between the coarse score baseline and task-relevant channels, with Null blind as a negative control.")
    lines.append("Unsigned pointwise loss motion is supporting evidence only and is reported in the appendix because it removes cancellation before averaging.")
    lines.append("The strongest task channel may be an ablation such as Task minus cost or Task minus subgroup. This is not a failure; adding coordinates can introduce sparsity, nuisance variation, or cancellation.")
    lines.append("")

    lines.extend(
        [
            "## Calibration Caveat",
            "",
            "Raw observed FR is not a calibrated estimator of run-level $V_T$.",
            "Calibration or thresholding would require a deployment-specific operating-envelope layer using replay, simulation, or held-out monitoring.",
            "That calibration problem is future work; this experiment establishes the observability principle.",
            "",
        ]
    )

    lines.extend(
        [
            "## Appendix File Map",
            "",
            "| File | Use |",
            "| --- | --- |",
            "| `RESULTS_DASHBOARD.md` | Start here. Human-readable summary. |",
            "| `README.md` | Main vs appendix output map. |",
            "| `main_figure_values_summary.txt` | Exact values used in the main figures. |",
            "| `figure_main_channel_association_summary.*` | Primary channel association summary for $V_T$, $\\Delta_T^{\\mathrm{rep}}$, and $v_t$. |",
            "| `figure_main_step_fr_association.*` | Primary transition-level channel association figure. |",
            "| `figure_main_feedback_manipulation_check.*` | Secondary manipulation check. |",
            "| `table_channel_association_summary.csv` | Values shown in `figure_main_channel_association_summary.*`. |",
            "| `table_operational_raw_rate_diagnostics.csv` | Main deployable raw-rate and per-step diagnostic table. |",
            "| `regional_feedback_step_diagnostics.csv` | Per-transition FR vs risk/motion diagnostics. |",
            "| `regional_feedback_summary.csv` | Main metrics by mu. |",
            "| `regional_feedback_rounds.csv` | Round-level raw data. Large/detail file. |",
            "| `regional_feedback_results_by_seed.csv` | Condition/seed-level raw data. |",
            "| `appendix_diagnostics/appendix_table_motion_diagnostics.csv` | Unsigned pointwise loss/error motion and held-out step-shift support diagnostics. |",
            "| `appendix_diagnostics/` | Excess-rate, leave-one-$\\mu$-out, within-$\\mu$, pooled R2 heatmap, individual-run scatter, motion diagnostics, and condition-mean descriptive diagnostics. |",
            "",
        ]
    )

    outpath.write_text("\n".join(lines), encoding="utf-8")


def linear_slope(x: pd.Series, y: pd.Series) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_var = float(np.var(x_arr))
    if x_var <= EPS:
        return 0.0
    return float(np.cov(x_arr, y_arr, ddof=0)[0, 1] / x_var)


def build_channel_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mu_min = float(results_df["mu"].min())
    mu_max = float(results_df["mu"].max())
    for channel in CHANNEL_ORDER:
        rate_col = f"A_rate_{channel}"
        excess_col = f"A_rate_{channel}_excess"
        rate_by_mu = results_df.groupby("mu")[rate_col].mean()
        excess_by_mu = results_df.groupby("mu")[excess_col].mean()
        spearman_gap = safe_corr(results_df[excess_col], results_df["Delta_rep_T"], method="spearman")
        spearman_vt = safe_corr(results_df[excess_col], results_df["V_T"], method="spearman")
        pearson_gap = safe_corr(results_df[excess_col], results_df["Delta_rep_T"], method="pearson")
        pearson_vt = safe_corr(results_df[excess_col], results_df["V_T"], method="pearson")
        slope_gap = linear_slope(results_df[excess_col], results_df["Delta_rep_T"])
        slope_vt = linear_slope(results_df[excess_col], results_df["V_T"])
        slope_excess_per_gap = linear_slope(results_df["Delta_rep_T"], results_df[excess_col])
        slope_excess_per_vt = linear_slope(results_df["V_T"], results_df[excess_col])
        baseline_rate = float(rate_by_mu.loc[mu_min])
        raw_response = float(rate_by_mu.loc[mu_max] - rate_by_mu.loc[mu_min])
        excess_response = float(excess_by_mu.loc[mu_max])
        baseline_to_excess_ratio = float(baseline_rate / max(abs(excess_response), EPS))
        association_score = float((abs(spearman_gap) + abs(spearman_vt)) / 2.0)
        feedback_sensitivity_score = float(excess_response * association_score / (1.0 + baseline_to_excess_ratio))
        rows.append(
            {
                "channel": channel,
                "label": CHANNEL_LABELS[channel],
                "definition": CHANNEL_DEFINITIONS[channel],
                "raw_FR_rate_mu0": baseline_rate,
                "raw_FR_rate_mu_max": float(rate_by_mu.loc[mu_max]),
                "excess_FR_mu_max": excess_response,
                "raw_FR_response_mu_max_minus_mu0": raw_response,
                "baseline_to_excess_ratio": baseline_to_excess_ratio,
                "fr_rate_std_across_runs": float(results_df[rate_col].std()),
                "pearson_excess_with_Delta_rep_T": pearson_gap,
                "spearman_excess_with_Delta_rep_T": spearman_gap,
                "pearson_excess_with_V_T": pearson_vt,
                "spearman_excess_with_V_T": spearman_vt,
                "slope_Delta_rep_T_per_excess_FR": slope_gap,
                "slope_V_T_per_excess_FR": slope_vt,
                "slope_excess_FR_per_Delta_rep_T": slope_excess_per_gap,
                "slope_excess_FR_per_V_T": slope_excess_per_vt,
                "association_score": association_score,
                "feedback_sensitivity_score": feedback_sensitivity_score,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison["feedback_sensitivity_rank"] = comparison["feedback_sensitivity_score"].rank(method="dense").astype(int)
    return comparison.sort_values(["feedback_sensitivity_score", "excess_FR_mu_max"]).reset_index(drop=True)


def paired_sign_permutation_pvalue(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n == 0:
        return float("nan")

    observed = abs(float(diffs.mean()))
    if observed <= EPS:
        return 1.0

    if n <= 20:
        masks = np.arange(2**n, dtype=np.uint64)[:, None]
        bit_positions = np.arange(n, dtype=np.uint64)[None, :]
        signs = 1.0 - 2.0 * ((masks >> bit_positions) & 1).astype(float)
        permuted_means = np.sum(signs * diffs[None, :], axis=1) / n
    else:
        rng = np.random.default_rng(PAIRWISE_TEST_SEED)
        signs = rng.choice([-1.0, 1.0], size=(N_BOOTSTRAP, n), replace=True)
        permuted_means = np.sum(signs * diffs[None, :], axis=1) / n

    return float(np.mean(np.abs(permuted_means) >= observed - EPS))


def paired_bootstrap_ci(diffs: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        value = float(diffs[0])
        return value, value
    indices = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    boot_means = diffs[indices].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return float(ci_low), float(ci_high)


def build_pairwise_excess_tests_table(
    results_df: pd.DataFrame,
    comparisons: list[tuple[str, str, str]],
    *,
    contrast_label: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(PAIRWISE_TEST_SEED)
    rows = []
    for mu, mu_df in results_df[results_df["mu"] != 0.0].groupby("mu", sort=True):
        mu_df = mu_df.sort_values("seed")
        for comparison, left_channel, right_channel in comparisons:
            left = mu_df[f"A_rate_{left_channel}_excess"].to_numpy(dtype=float)
            right = mu_df[f"A_rate_{right_channel}_excess"].to_numpy(dtype=float)
            diffs = right - left
            ci_low, ci_high = paired_bootstrap_ci(diffs, rng)
            rows.append(
                {
                    "mu": float(mu),
                    "comparison": comparison,
                    "contrast_label": contrast_label,
                    "left_channel": left_channel,
                    "right_channel": right_channel,
                    "left_label": CHANNEL_LABELS[left_channel],
                    "right_label": CHANNEL_LABELS[right_channel],
                    "n_seeds": int(len(diffs)),
                    "left_mean_excess": float(np.mean(left)),
                    "right_mean_excess": float(np.mean(right)),
                    "mean_diff_right_minus_left": float(np.mean(diffs)),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "permutation_p_two_sided": paired_sign_permutation_pvalue(diffs),
                }
            )
    return pd.DataFrame(rows)


def write_channel_definitions(outdir: Path = APPENDIX_OUTPUT_DIR) -> None:
    lines = [
        "This real-data experiment is an applied robustness / partial-observability check.",
        "It is not a direct contraction-theorem test and does not estimate intrinsic C_T/T.",
        "",
        "Channels:",
    ]
    for channel in CHANNEL_ORDER:
        lines.append(f"- {CHANNEL_LABELS[channel]}: {CHANNEL_DEFINITIONS[channel]}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "channel_definitions.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_metric_heatmap(
    matrix_df: pd.DataFrame,
    *,
    row_col: str,
    value_cols: list[str],
    value_labels: list[str],
    outpath: Path,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if matrix_df.empty:
        return
    plot_df = matrix_df[[row_col, *value_cols]].dropna(how="all", subset=value_cols).copy()
    if plot_df.empty:
        return
    values = plot_df[value_cols].to_numpy(dtype=float)

    fig_height = max(1.8, 0.35 * len(plot_df) + 0.7)
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], fig_height), layout="constrained")
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(value_cols)))
    ax.set_xticklabels(value_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels([CHANNEL_LABELS.get(channel, channel) for channel in plot_df[row_col]])
    ax.tick_params(axis="both", which="both", length=0)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                text_color = "white" if value > 0.55 else "black"
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=text_color)
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label(cbar_label)
    save_figure_all_formats(fig, outpath)


def plot_condition_mean_raw_fr_vs_risk(
    summary_df: pd.DataFrame,
    n_seeds: int,
    *,
    channels: list[str],
    outpath: Path,
    figsize: tuple[float, float],
) -> None:
    if summary_df.empty:
        return
    targets = [
        ("V_T", r"$V_T$"),
        ("Delta_rep_T", r"$\Delta_T^{\mathrm{rep}}$"),
    ]
    channels = [
        channel
        for channel in channels
        if f"A_rate_{channel}_mean" in summary_df and f"A_rate_{channel}_std" in summary_df
    ]
    if not channels:
        return

    x_bounds = []
    for channel in channels:
        x_mean = summary_df[f"A_rate_{channel}_mean"].to_numpy(dtype=float)
        x_se = standard_error(summary_df[f"A_rate_{channel}_std"], n_seeds).to_numpy(dtype=float)
        x_bounds.extend([float(np.min(x_mean - x_se)), float(np.max(x_mean + x_se))])
    x_min = min(x_bounds)
    x_max = max(x_bounds)
    x_pad = 0.04 * max(x_max - x_min, EPS)

    fig, axes = plt.subplots(
        len(channels),
        len(targets),
        figsize=figsize,
        layout="constrained",
        squeeze=False,
    )
    mu_values = summary_df["mu"].to_numpy(dtype=float)
    norm = mpl.colors.Normalize(vmin=float(np.min(mu_values)), vmax=float(np.max(mu_values)))
    cmap = plt.get_cmap("viridis")
    mu_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.0,
            markerfacecolor=cmap(norm(mu)),
            markeredgecolor=cmap(norm(mu)),
            label=f"$\\mu$={mu:.2g}",
        )
        for mu in mu_values
    ]

    for row_idx, channel in enumerate(channels):
        x = summary_df[f"A_rate_{channel}_mean"].to_numpy(dtype=float)
        xerr = standard_error(summary_df[f"A_rate_{channel}_std"], n_seeds).to_numpy(dtype=float)
        for col_idx, (target, target_label) in enumerate(targets):
            ax = axes[row_idx, col_idx]
            if f"{target}_mean" not in summary_df or f"{target}_std" not in summary_df:
                ax.set_visible(False)
                continue
            y = summary_df[f"{target}_mean"].to_numpy(dtype=float)
            yerr = standard_error(summary_df[f"{target}_std"], n_seeds).to_numpy(dtype=float)
            slope = linear_slope(pd.Series(x), pd.Series(y))
            intercept = float(np.mean(y) - slope * np.mean(x))
            y_hat = intercept + slope * x
            r2 = finite_r2(y, y_hat)
            x_fit = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(x_fit, intercept + slope * x_fit, color="black", lw=0.75, ls="--", alpha=0.8)
            for point_idx, mu in enumerate(mu_values):
                color = cmap(norm(mu))
                ax.errorbar(
                    x[point_idx],
                    y[point_idx],
                    xerr=xerr[point_idx],
                    yerr=yerr[point_idx],
                    fmt="o",
                    markersize=3.2,
                    color=color,
                    ecolor=color,
                    elinewidth=0.65,
                    capsize=1.8,
                    alpha=0.95,
                )
            ax.plot(x, y, color="0.35", lw=0.65, alpha=0.65)
            ax.text(
                0.50,
                0.95,
                f"condition means over mu\nslope={slope:.2e}; R2={r2:.2f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=5.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
            )
            if row_idx == 0:
                ax.set_title(target_label)
            if col_idx == 0:
                ax.set_ylabel(CHANNEL_LABELS.get(channel, channel))
            if row_idx == len(channels) - 1:
                ax.set_xlabel("Mean raw observable FR rate")
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            style_axis(ax)

    axes[0, -1].legend(handles=mu_handles, frameon=False, fontsize=6.2, loc="lower right")
    save_figure_all_formats(fig, outpath)


def plot_appendix_run_level_raw_rate_r2_heatmap(raw_rate_regression_df: pd.DataFrame) -> None:
    if raw_rate_regression_df.empty:
        return
    targets = [
        "V_T",
        "Delta_rep_T",
        "Delta_sam_T",
        "Eval_within_step_drift_T",
        "Mean_abs_delta_loss_T",
        "Mean_abs_delta_error_T",
    ]
    focus = raw_rate_regression_df[raw_rate_regression_df["target"].isin(targets)].copy()
    if focus.empty:
        return
    matrix = (
        focus.pivot_table(index="channel", columns="target", values="r2_in_sample", aggfunc="mean")
        .reindex(CHANNEL_ORDER)
    )
    available_targets = [target for target in targets if target in matrix.columns]
    values = matrix[available_targets].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 2.9), layout="constrained")
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(available_targets)))
    ax.set_xticklabels([target_label(target) for target in available_targets], rotation=18, ha="right")
    ax.set_yticks(np.arange(len(CHANNEL_ORDER)))
    ax.set_yticklabels([channel_label(channel) for channel in CHANNEL_ORDER])
    ax.tick_params(axis="both", which="both", length=0)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                text_color = "white" if value > 0.55 else "black"
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=7.0, color=text_color)

    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label(r"Raw-rate regression $R^2$")
    save_figure_all_formats(fig, APPENDIX_OUTPUT_DIR / "appendix_figure_run_level_raw_rate_r2_heatmap.png")


def build_channel_association_summary_table(
    raw_rate_regression_df: pd.DataFrame,
    step_diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    channels = CHANNEL_ORDER
    run_targets = [
        ("V_T", r"$V_T$"),
        ("Delta_rep_T", r"$\Delta_T^{\mathrm{rep}}$"),
    ]
    step_targets = [
        ("v_t", r"$v_t$"),
    ]
    positive_step = step_diagnostics_df[step_diagnostics_df["mu"].astype(str) == "all_positive"].copy()

    run_rows = []
    for channel in channels:
        for target, label in run_targets:
            channel_row = raw_rate_regression_df[
                (raw_rate_regression_df["channel"] == channel) & (raw_rate_regression_df["target"] == target)
            ]
            if channel_row.empty:
                continue
            channel_r2 = float(channel_row.iloc[0]["r2_in_sample"])
            run_rows.append(
                {
                    "scale": "run_level",
                    "channel": channel,
                    "target": target,
                    "target_label": label,
                    "metric": "raw_rate_r2",
                    "value": channel_r2,
                }
            )

    step_rows = []
    for channel in channels:
        for target, label in step_targets:
            channel_row = positive_step[(positive_step["channel"] == channel) & (positive_step["target"] == target)]
            if channel_row.empty:
                continue
            channel_rho = float(channel_row.iloc[0]["spearman"])
            step_rows.append(
                {
                    "scale": "transition_level",
                    "channel": channel,
                    "target": target,
                    "target_label": label,
                    "metric": "spearman_rho",
                    "value": channel_rho,
                }
            )

    return pd.DataFrame([*run_rows, *step_rows])


def build_appendix_motion_diagnostics_table(
    raw_rate_regression_df: pd.DataFrame,
    step_diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    channels = CHANNEL_ORDER
    run_targets = [
        ("Mean_abs_delta_loss_T", "pointwise loss motion"),
        ("Mean_abs_delta_error_T", "pointwise error motion"),
        ("Eval_within_step_drift_T", "held-out step shift"),
    ]
    step_targets = [
        ("mean_abs_delta_loss", "pointwise loss motion"),
        ("mean_abs_delta_error", "pointwise error motion"),
        ("eval_within_step_drift", "held-out step shift"),
    ]
    positive_step = step_diagnostics_df[step_diagnostics_df["mu"].astype(str) == "all_positive"].copy()
    rows = []
    for channel in channels:
        for target, label in run_targets:
            channel_row = raw_rate_regression_df[
                (raw_rate_regression_df["channel"] == channel) & (raw_rate_regression_df["target"] == target)
            ]
            if channel_row.empty:
                continue
            rows.append(
                {
                    "scale": "run_level",
                    "channel": channel,
                    "channel_label": channel_label(channel),
                    "target": target,
                    "target_label": label,
                    "metric": "raw_rate_r2",
                    "value": float(channel_row.iloc[0]["r2_in_sample"]),
                }
            )
        for target, label in step_targets:
            channel_row = positive_step[(positive_step["channel"] == channel) & (positive_step["target"] == target)]
            if channel_row.empty:
                continue
            rows.append(
                {
                    "scale": "transition_level",
                    "channel": channel,
                    "channel_label": channel_label(channel),
                    "target": target,
                    "target_label": label,
                    "metric": "spearman_rho",
                    "value": float(channel_row.iloc[0]["spearman"]),
                }
            )
    return pd.DataFrame(rows)


def build_gain_over_coarse_table(association_df: pd.DataFrame) -> pd.DataFrame:
    if association_df.empty:
        return pd.DataFrame()
    coarse = association_df[association_df["channel"] == "coarse"].set_index(["scale", "target"])
    rows = []
    for _, row in association_df[~association_df["channel"].isin(["null", "coarse"])].iterrows():
        key = (row["scale"], row["target"])
        if key not in coarse.index:
            continue
        coarse_value = float(coarse.loc[key]["value"])
        channel_value = float(row["value"])
        rows.append(
            {
                "scale": row["scale"],
                "target": row["target"],
                "target_label": row["target_label"],
                "channel": row["channel"],
                "channel_label": channel_label(row["channel"]),
                "channel_value": channel_value,
                "coarse_score_value": coarse_value,
                "gain_over_coarse_score": channel_value - coarse_value,
            }
        )
    return pd.DataFrame(rows)


def plot_main_channel_association_summary(
    raw_rate_regression_df: pd.DataFrame,
    step_diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    channels = CHANNEL_ORDER
    run_targets = [
        ("V_T", r"$V_T$"),
        ("Delta_rep_T", r"$\Delta_T^{\mathrm{rep}}$"),
    ]
    step_targets = [
        ("v_t", r"$v_t$"),
    ]
    association_df = build_channel_association_summary_table(raw_rate_regression_df, step_diagnostics_df)
    if association_df.empty:
        return association_df

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), layout="constrained")
    bar_width = 0.36
    x = np.arange(len(channels))

    for idx, (target, label) in enumerate(run_targets):
        values = [
            float(association_df[(association_df["scale"] == "run_level") & (association_df["channel"] == channel) & (association_df["target"] == target)]["value"].iloc[0])
            if not association_df[(association_df["scale"] == "run_level") & (association_df["channel"] == channel) & (association_df["target"] == target)].empty
            else float("nan")
            for channel in channels
        ]
        axes[0].bar(x + (idx - 0.5) * bar_width, values, width=bar_width, label=label, color=PALETTE[idx])
    axes[0].axhline(0.0, color="black", lw=0.7)
    axes[0].set_ylabel(r"Raw-rate regression $R^2$")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([channel_label(channel) for channel in channels], rotation=25, ha="right", fontsize=7.0)
    axes[0].legend(frameon=False, fontsize=7.0, loc="upper left")

    target, label = step_targets[0]
    values = [
        float(association_df[(association_df["scale"] == "transition_level") & (association_df["channel"] == channel) & (association_df["target"] == target)]["value"].iloc[0])
        if not association_df[(association_df["scale"] == "transition_level") & (association_df["channel"] == channel) & (association_df["target"] == target)].empty
        else float("nan")
        for channel in channels
    ]
    axes[1].bar(x, values, width=0.58, label=label, color=PALETTE[0])
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].set_ylabel(r"Spearman $\rho$")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([channel_label(channel) for channel in channels], rotation=25, ha="right", fontsize=7.0)
    axes[1].legend(frameon=False, fontsize=7.0, loc="upper left")

    for label, ax in zip(["(a)", "(b)"], axes):
        ax.text(0.98, 0.96, label, transform=ax.transAxes, ha="right", va="top", fontsize=8.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.6)

    save_figure_all_formats(fig, OUTPUT_DIR / "figure_main_channel_association_summary.png")
    return association_df


def plot_main_step_fr_association(step_diagnostics_df: pd.DataFrame) -> pd.DataFrame:
    if step_diagnostics_df.empty:
        return pd.DataFrame()
    step_vt = step_diagnostics_df[
        (step_diagnostics_df["mu"].astype(str) == "all_positive") & (step_diagnostics_df["target"] == "v_t")
    ].copy()
    if step_vt.empty:
        return pd.DataFrame()
    step_vt = step_vt.sort_values("spearman", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.2, 2.7), layout="constrained")
    x = np.arange(len(step_vt))
    bars = ax.bar(x, step_vt["spearman"].to_numpy(dtype=float), color=PALETTE[0], width=0.68)
    ax.set_ylabel(r"Spearman $\rho(\mathrm{FR\ step}, v_t)$")
    ax.set_xticks(x)
    ax.set_xticklabels([channel_label(channel) for channel in step_vt["channel"]], rotation=25, ha="right", fontsize=7.0)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", which="major", alpha=0.25, linewidth=0.6)
    for bar, value in zip(bars, step_vt["spearman"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.015,
            f"{float(value):.2f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
    n_steps = int(step_vt["n_steps"].iloc[0])
    ax.text(0.98, 0.94, f"n={n_steps} positive-$\\mu$ transitions", transform=ax.transAxes, ha="right", va="top", fontsize=7.0)
    save_figure_all_formats(fig, OUTPUT_DIR / "figure_main_step_fr_association.png")
    return step_vt


def plot_main_feedback_manipulation_check(summary_df: pd.DataFrame, n_seeds: int) -> None:
    if summary_df.empty:
        return
    mu = summary_df["mu"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.55), layout="constrained")

    risk_specs = [
        ("V_T", r"$V_T$", PALETTE[0], "o"),
        ("Delta_rep_T", r"$\Delta_T^{\mathrm{rep}}$", PALETTE[1], "s"),
    ]
    for target, label, color, marker in risk_specs:
        mean_col = f"{target}_mean"
        std_col = f"{target}_std"
        if mean_col not in summary_df or std_col not in summary_df:
            continue
        y = summary_df[mean_col].to_numpy(dtype=float)
        yerr = standard_error(summary_df[std_col], n_seeds).to_numpy(dtype=float)
        axes[0].errorbar(mu, y, yerr=yerr, marker=marker, markersize=3.2, lw=0.9, color=color, capsize=2.0, label=label)
    axes[0].set_xlabel(r"Feedback strength $\mu$")
    axes[0].set_ylabel("Risk quantity")
    axes[0].legend(frameon=False, fontsize=7.0, loc="upper left")
    style_axis(axes[0])

    channel_specs = [
        ("null", "o", PALETTE[0]),
        ("coarse", "s", PALETTE[1]),
        ("task", "^", PALETTE[2]),
    ]
    for channel, marker, color in channel_specs:
        mean_col = f"A_rate_{channel}_mean"
        std_col = f"A_rate_{channel}_std"
        if mean_col not in summary_df or std_col not in summary_df:
            continue
        y = summary_df[mean_col].to_numpy(dtype=float)
        yerr = standard_error(summary_df[std_col], n_seeds).to_numpy(dtype=float)
        axes[1].errorbar(mu, y, yerr=yerr, marker=marker, markersize=3.2, lw=0.9, color=color, capsize=2.0, label=channel_label(channel))
    axes[1].set_xlabel(r"Feedback strength $\mu$")
    axes[1].set_ylabel("Raw observed FR rate")
    axes[1].legend(frameon=False, fontsize=7.0, loc="upper left")
    style_axis(axes[1])

    for label, ax in zip(["(a)", "(b)"], axes):
        ax.text(0.02, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=8.0)

    save_figure_all_formats(fig, OUTPUT_DIR / "figure_main_feedback_manipulation_check.png")


def plot_appendix_operational_spearman_heatmap(operational_df: pd.DataFrame) -> None:
    if operational_df.empty:
        return
    row_order = ["task_no_group", "task", "task_no_cost", "null", "task_no_qty", "coarse"]
    plot_df = operational_df.set_index("channel").reindex(row_order).dropna(how="all").reset_index()
    value_cols = [
        "condition_mean_spearman_V_T",
        "condition_mean_spearman_Mean_abs_delta_loss_T",
        "step_spearman_v_t",
        "step_spearman_mean_abs_delta_loss",
    ]
    value_labels = [
        "Condition mean FR\nvs $V_T$",
        "Condition mean FR\nvs pointwise loss motion",
        "Per-step FR\nvs $v_t$",
        "Per-step FR\nvs pointwise loss motion",
    ]
    available_cols = [col for col in value_cols if col in plot_df.columns]
    available_labels = [label for col, label in zip(value_cols, value_labels) if col in plot_df.columns]
    plot_metric_heatmap(
        plot_df,
        row_col="channel",
        value_cols=available_cols,
        value_labels=available_labels,
        outpath=APPENDIX_OUTPUT_DIR / "appendix_figure_operational_spearman_heatmap.png",
        cbar_label=r"Spearman $\rho$",
        vmin=0.0,
        vmax=1.0,
    )


def plot_appendix_individual_run_raw_fr_scatter(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        return
    channels = ["task_no_cost", "null"]
    targets = [("V_T", r"$V_T$"), ("Delta_rep_T", r"$\Delta_T^{\mathrm{rep}}$")]
    fig, axes = plt.subplots(
        len(channels),
        len(targets),
        figsize=(7.2, 3.8),
        layout="constrained",
        squeeze=False,
    )
    mu_values = results_df["mu"].to_numpy(dtype=float)
    norm = mpl.colors.Normalize(vmin=float(np.min(mu_values)), vmax=float(np.max(mu_values)))
    cmap = plt.get_cmap("viridis")

    for row_idx, channel in enumerate(channels):
        predictor = f"A_rate_{channel}"
        if predictor not in results_df:
            continue
        for col_idx, (target, target_name) in enumerate(targets):
            ax = axes[row_idx, col_idx]
            if target not in results_df:
                ax.set_visible(False)
                continue
            model_df = results_df[[predictor, target, "mu"]].dropna()
            if model_df.empty:
                ax.set_visible(False)
                continue
            ax.scatter(
                model_df[predictor],
                model_df[target],
                c=model_df["mu"],
                cmap=cmap,
                norm=norm,
                s=12,
                alpha=0.78,
                linewidths=0,
            )
            ax.set_xlabel("Raw observed FR rate")
            ax.set_ylabel(target_name if col_idx == 0 else "")
            if row_idx == 0:
                ax.set_title(target_name)
            if col_idx == 0:
                ax.text(
                    -0.17,
                    0.5,
                    channel_label(channel),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8.0,
                )
            ax.text(
                0.5,
                0.95,
                "diagnostic only; not a calibrated\nrun-level estimator",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=5.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )
            style_axis(ax)

    mu_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4.0,
            markerfacecolor=cmap(norm(mu)),
            markeredgecolor=cmap(norm(mu)),
            label=f"$\\mu$={mu:.2g}",
        )
        for mu in sorted(results_df["mu"].unique())
    ]
    axes[0, -1].legend(handles=mu_handles, frameon=False, fontsize=6.0, loc="lower right")
    save_figure_all_formats(fig, APPENDIX_OUTPUT_DIR / "appendix_figure_individual_run_raw_fr_vs_risk_diagnostic.png")


def make_plots(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    round_df: pd.DataFrame,
    raw_rate_regression_df: pd.DataFrame,
    raw_rate_operational_summary_df: pd.DataFrame,
    step_diagnostics_df: pd.DataFrame,
) -> None:
    n_seeds = results_df["seed"].nunique()
    plot_main_channel_association_summary(raw_rate_regression_df, step_diagnostics_df)
    plot_main_step_fr_association(step_diagnostics_df)
    plot_main_feedback_manipulation_check(summary_df, n_seeds)
    plot_appendix_run_level_raw_rate_r2_heatmap(raw_rate_regression_df)
    plot_appendix_operational_spearman_heatmap(raw_rate_operational_summary_df)
    plot_appendix_individual_run_raw_fr_scatter(results_df)
    plot_condition_mean_raw_fr_vs_risk(
        summary_df,
        n_seeds,
        channels=["task_no_cost", "null"],
        outpath=APPENDIX_OUTPUT_DIR / "appendix_figure_condition_mean_raw_fr_vs_risk_focus.png",
        figsize=(7.2, 3.8),
    )
    plot_condition_mean_raw_fr_vs_risk(
        summary_df,
        n_seeds,
        channels=CHANNEL_ORDER,
        outpath=APPENDIX_OUTPUT_DIR / "appendix_figure_condition_mean_raw_fr_vs_risk_all_channels.png",
        figsize=(7.2, max(7.2, 1.15 * len(CHANNEL_ORDER))),
    )


def write_dataset_profile(df: pd.DataFrame, outdir: Path = APPENDIX_OUTPUT_DIR) -> None:
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
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "dataset_profile.txt").write_text("\n".join(profile_lines), encoding="utf-8")


def write_outputs_readme(outpath: Path) -> None:
    lines = [
        "# Regional Feedback Outputs",
        "",
        "Main outputs use raw observed Fisher rates and operational diagnostics.",
        "Appendix diagnostics are robustness and stress-test outputs.",
        "",
        "## Main Claim",
        "",
        "Raw observed Fisher motion is an uncalibrated but meaningful channel-level footprint of drift-induced risk motion.",
        "It is not a calibrated estimator of run-level $V_T$ and it is not an estimate of the intrinsic $C_T/T$ budget.",
        "The null blind channel is a fixed row/hash bucket partition and is used as a negative control.",
        "",
        "## Main Outputs",
        "",
        "- `RESULTS_DASHBOARD.md`: paper-facing result summary.",
        "- `main_figure_values_summary.txt`: exact values used in the main figures.",
        "- `figure_main_channel_association_summary.*`: run-level raw-rate $R^2$ for $V_T$ and $\\Delta_T^{\\mathrm{rep}}$, plus transition-level Spearman $\\rho$ for $v_t$.",
        "- `figure_main_step_fr_association.*`: positive-feedback transition Spearman association between per-step observed FR and $v_t$, including null.",
        "- `figure_main_feedback_manipulation_check.*`: secondary manipulation check against feedback strength $\\mu$.",
        "- `table_channel_association_summary.csv`: values shown in the main channel association figure.",
        "- `regional_feedback_raw_rate_regressions.csv`: raw-rate single-channel regressions with no $\\mu$ covariate.",
        "- `regional_feedback_step_diagnostics.csv`: transition-level association diagnostics.",
        "- `table_operational_raw_rate_diagnostics.csv`: compact operational diagnostics table.",
        "",
        "## Appendix Diagnostics",
        "",
        "Appendix files live in `appendix_diagnostics/`.",
        "CSV files use the literal channel id `null`; with pandas, read these files with `keep_default_na=False` if you need to preserve that string instead of parsing it as missing.",
        "The pooled raw-rate $R^2$ heatmap is appendix-only because it primarily summarizes regime-level co-movement and should not be interpreted as calibrated run-level prediction.",
        "`appendix_table_gain_over_coarse_score.csv` reports channel minus Coarse score diagnostics for task-relevant channels.",
        "`appendix_table_motion_diagnostics.csv` reports unsigned pointwise loss/error motion and held-out step-shift diagnostics.",
        "Unsigned pointwise loss motion removes cancellation before averaging, so it is expected to align more strongly with unsigned Fisher motion than $\\Delta_T^{\\mathrm{rep}}$.",
        "Excess rates are controlled attribution diagnostics only, because deployment does not have matched $\\mu=0$ counterfactuals.",
        "Leave-one-$\\mu$-out asks a stronger cross-regime extrapolation question than the observability claim.",
        "Seed-level within-$\\mu$ associations ask whether naive fixed channels rank run-level severity at fixed feedback strength.",
        "Condition-mean and individual-run scatter plots are descriptive diagnostics, not calibrated monitoring results.",
        "",
        "## Caveat",
        "",
        "The experiment uses simple fixed binning rather than optimized channels.",
        "Calibration or thresholding would require a deployment-specific operating-envelope calibration layer using replay, simulation, or held-out monitoring.",
    ]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def write_main_figure_values_summary(
    raw_rate_regression_df: pd.DataFrame,
    step_diagnostics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    outpath: Path,
) -> None:
    lines = [
        "Main figure values for regional feedback observability experiment",
        "",
        "Figure A: channel association summary",
        "Panel (a) reports run-level raw-rate R^2 for $V_T$ and $\\Delta_T^{\\mathrm{rep}}$.",
        "Panel (b) reports transition-level Spearman rho for $v_t$.",
        "Null blind is a fixed row-bucket negative control with no observed Fisher motion.",
        "Pointwise loss-motion diagnostics are appendix-only because they are unsigned cancellation-free quantities.",
        "",
    ]
    association_df = build_channel_association_summary_table(raw_rate_regression_df, step_diagnostics_df)
    for scale in ["run_level", "transition_level"]:
        scale_df = association_df[association_df["scale"] == scale]
        lines.append(f"{scale}:")
        for _, row in scale_df.sort_values(["target", "channel"]).iterrows():
            lines.append(
                f"  {channel_label(row['channel'])}, {target_label(row['target'])}: "
                f"{row['metric']}={float(row['value']):.6g}"
            )
        lines.append("")

    gain_df = build_gain_over_coarse_table(association_df)
    if not gain_df.empty:
        lines.extend(
            [
                "Appendix: gain over Coarse score",
                "Gain is channel value minus the Coarse score baseline for the same target and scale.",
                "",
            ]
        )
        for _, row in gain_df.sort_values(["scale", "target", "channel"]).iterrows():
            lines.append(
                f"{row['scale']}, {channel_label(row['channel'])}, {target_label(row['target'])}: "
                f"channel={float(row['channel_value']):.6g}, coarse={float(row['coarse_score_value']):.6g}, "
                f"gain={float(row['gain_over_coarse_score']):.6g}"
            )
        lines.append("")

    motion_df = build_appendix_motion_diagnostics_table(raw_rate_regression_df, step_diagnostics_df)
    if not motion_df.empty:
        lines.extend(
            [
                "Appendix: unsigned motion diagnostics",
                "These diagnostics remove cancellation before averaging and are supporting evidence only.",
                "",
            ]
        )
        for _, row in motion_df.sort_values(["scale", "target", "channel"]).iterrows():
            lines.append(
                f"{row['scale']}, {channel_label(row['channel'])}, {row['target_label']}: "
                f"{row['metric']}={float(row['value']):.6g}"
            )
        lines.append("")

    lines.extend(
        [
        "Figure B: per-transition association by channel",
        "Values are Spearman correlations between per-step observed FR and $v_t$ over positive-feedback transitions.",
            "",
        ]
    )
    step_vt = step_diagnostics_df[
        (step_diagnostics_df["mu"].astype(str) == "all_positive") & (step_diagnostics_df["target"] == "v_t")
    ].sort_values("spearman", ascending=False)
    for _, row in step_vt.iterrows():
        lines.append(
            f"{channel_label(row['channel'])}: Spearman rho={float(row['spearman']):.6g}, n={int(row['n_steps'])}"
        )
    lines.append("")

    lines.extend(
        [
            "Figure C: manipulation check condition means",
            "Values are condition means by feedback strength. Standard errors in the figure are computed across seeds.",
            "",
        ]
    )
    manipulation_cols = [
        ("V_T_mean", "$V_T$ mean"),
        ("Delta_rep_T_mean", "$\\Delta_T^{\\mathrm{rep}}$ mean"),
        ("A_rate_null_mean", "Null blind raw observed FR mean"),
        ("A_rate_coarse_mean", "Coarse score raw observed FR mean"),
        ("A_rate_task_mean", "Task-aligned raw observed FR mean"),
    ]
    for _, row in summary_df.sort_values("mu").iterrows():
        lines.append(f"$\\mu$={float(row['mu']):.6g}:")
        for col, label in manipulation_cols:
            if col in summary_df.columns:
                lines.append(f"  {label}: {float(row[col]):.10g}")
        lines.append("")

    outpath.write_text("\n".join(lines), encoding="utf-8")


def cleanup_obsolete_public_outputs() -> None:
    obsolete_patterns = [
        "figure_1_gap_vs_mu.*",
        "figure_2_drift_vs_mu.*",
        "figure_3_raw_fr_rate_vs_mu.*",
        "figure_3_excess_fr_rate_vs_mu.*",
        "figure_4_step_fr_vs_vt.*",
        "figure_5_raw_rate_regression_r2.*",
        "figure_6_operational_association_heatmap.*",
        "figure_7_best_raw_rate_regression_fits.*",
        "figure_8_condition_mean_raw_fr_vs_risk.*",
        "figure_8_condition_mean_raw_fr_vs_risk_focus.*",
        "figure_main_operational_raw_rate_r2_heatmap.*",
        "figure_main_channel_lift.*",
        "regression_interpretation.txt",
        "regional_feedback_regressions.csv",
        "regional_feedback_regressions_cv.csv",
        "regional_feedback_raw_rate_within_mu_associations.csv",
        "regional_feedback_raw_rate_condition_mean_trends.csv",
        "table_5_regression_diagnostics.csv",
    ]
    for pattern in obsolete_patterns:
        for path in OUTPUT_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    df0 = load_and_prepare_base_dataframe(DATA_PATH)
    write_dataset_profile(df0, APPENDIX_OUTPUT_DIR)
    print(f"Loaded dataframe with shape: {df0.shape}")
    print("Columns:")
    for c in df0.columns:
        print(f"  - {c}")

    ridge_alpha = choose_ridge_alpha(df0)
    print(f"Chosen Ridge alpha: {ridge_alpha}")

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
    round_records: list[dict] = []
    burn_in_records: list[dict] = []
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
                round_records_sink=round_records,
                burn_in_records_sink=burn_in_records,
            )
            results.append(result)

    results_df = add_excess_fr_rates(pd.DataFrame(results))
    round_df = pd.DataFrame(round_records)
    burn_in_df = pd.DataFrame(burn_in_records)
    summary_df = summarize_results(results_df)
    sensitivity_df = build_early_round_sensitivity_table(results_df)
    step_diagnostics_df = build_step_diagnostics_table(round_df)

    results_df.to_csv(OUTPUT_DIR / "regional_feedback_results_by_seed.csv", index=False)
    round_df.to_csv(OUTPUT_DIR / "regional_feedback_rounds.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "regional_feedback_summary.csv", index=False)
    step_diagnostics_df.to_csv(OUTPUT_DIR / "regional_feedback_step_diagnostics.csv", index=False)
    burn_in_df.to_csv(APPENDIX_OUTPUT_DIR / "appendix_burnin_diagnostics.csv", index=False)
    summary_df.to_csv(APPENDIX_OUTPUT_DIR / "table_appendix_summary_with_uncertainty.csv", index=False)
    sensitivity_df.to_csv(APPENDIX_OUTPUT_DIR / "appendix_early_round_sensitivity.csv", index=False)

    # Channel comparison table for the partial-observability robustness check.
    corr_rows = []
    for ch_name in CHANNEL_ORDER:
        x = results_df[f"A_rate_{ch_name}_excess"]
        y_gap = results_df["Delta_rep_T"]
        y_vt = results_df["V_T"]
        corr_rows.append(
            {
                "channel": ch_name,
                "label": CHANNEL_LABELS[ch_name],
                "pearson_excess_with_gap": safe_corr(x, y_gap, method="pearson"),
                "spearman_excess_with_gap": safe_corr(x, y_gap, method="spearman"),
                "pearson_excess_with_vt": safe_corr(x, y_vt, method="pearson"),
                "spearman_excess_with_vt": safe_corr(x, y_vt, method="spearman"),
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    channel_comparison_df = build_channel_comparison_table(results_df)
    regression_df = fit_linear_regression_table(results_df)
    regression_cv_df = fit_leave_one_mu_out_cv_table(results_df)
    regression_stress_test_df = build_regression_diagnostics_table(regression_df, regression_cv_df)
    within_mu_df = build_within_mu_association_table(results_df)
    within_mu_summary_df = build_within_mu_channel_summary(within_mu_df)
    condition_mean_trend_df = build_condition_mean_trend_table(results_df)
    raw_rate_regression_df = fit_raw_rate_regression_table(results_df)
    raw_rate_within_mu_df = build_raw_rate_within_mu_association_table(results_df)
    raw_rate_condition_mean_trend_df = build_raw_rate_condition_mean_trend_table(results_df)
    raw_rate_operational_summary_df = build_operational_raw_rate_diagnostics(
        raw_rate_condition_mean_trend_df,
        step_diagnostics_df,
    )
    channel_association_summary_df = build_channel_association_summary_table(
        raw_rate_regression_df,
        step_diagnostics_df,
    )
    gain_over_coarse_df = build_gain_over_coarse_table(channel_association_summary_df)
    appendix_motion_diagnostics_df = build_appendix_motion_diagnostics_table(
        raw_rate_regression_df,
        step_diagnostics_df,
    )
    primary_pairwise_tests_df = build_pairwise_excess_tests_table(
        results_df,
        comparisons=[
            ("null_vs_coarse", "null", "coarse"),
            ("coarse_vs_task", "coarse", "task"),
            ("null_vs_task", "null", "task"),
        ],
        contrast_label="right_minus_left",
    )
    ablation_pairwise_tests_df = build_pairwise_excess_tests_table(
        results_df,
        comparisons=[
            ("full_task_vs_minus_quantity", "task_no_qty", "task"),
            ("full_task_vs_minus_cost", "task_no_cost", "task"),
            ("full_task_vs_minus_subgroup", "task_no_group", "task"),
        ],
        contrast_label="task_minus_ablation",
    )
    cleanup_obsolete_public_outputs()
    raw_rate_regression_df.to_csv(OUTPUT_DIR / "regional_feedback_raw_rate_regressions.csv", index=False)
    raw_rate_condition_mean_trend_df.to_csv(OUTPUT_DIR / "regional_feedback_condition_mean_trends.csv", index=False)
    raw_rate_operational_summary_df.to_csv(OUTPUT_DIR / "table_operational_raw_rate_diagnostics.csv", index=False)
    channel_association_summary_df.to_csv(OUTPUT_DIR / "table_channel_association_summary.csv", index=False)

    corr_df.to_csv(APPENDIX_OUTPUT_DIR / "appendix_excess_rate_correlations.csv", index=False)
    channel_comparison_df.to_csv(APPENDIX_OUTPUT_DIR / "appendix_excess_rate_channel_comparison.csv", index=False)
    regression_df.to_csv(APPENDIX_OUTPUT_DIR / "appendix_pooled_mu_controlled_regressions.csv", index=False)
    regression_cv_df.to_csv(APPENDIX_OUTPUT_DIR / "appendix_leave_one_mu_out_regressions_cv.csv", index=False)
    regression_stress_test_df.to_csv(APPENDIX_OUTPUT_DIR / "appendix_leave_one_mu_out_regression_stress_test.csv", index=False)
    within_mu_df.to_csv(APPENDIX_OUTPUT_DIR / "appendix_excess_rate_within_mu_seed_associations.csv", index=False)
    within_mu_summary_df.to_csv(APPENDIX_OUTPUT_DIR / "table_appendix_excess_rate_diagnostics.csv", index=False)
    condition_mean_trend_df.to_csv(APPENDIX_OUTPUT_DIR / "appendix_excess_rate_condition_mean_trends.csv", index=False)
    raw_rate_within_mu_df.to_csv(APPENDIX_OUTPUT_DIR / "appendix_raw_rate_within_mu_seed_associations.csv", index=False)
    gain_over_coarse_df.to_csv(APPENDIX_OUTPUT_DIR / "appendix_table_gain_over_coarse_score.csv", index=False)
    appendix_motion_diagnostics_df.to_csv(APPENDIX_OUTPUT_DIR / "appendix_table_motion_diagnostics.csv", index=False)
    primary_pairwise_tests_df.to_csv(APPENDIX_OUTPUT_DIR / "appendix_pairwise_excess_tests.csv", index=False)
    ablation_pairwise_tests_df.to_csv(APPENDIX_OUTPUT_DIR / "appendix_ablation_pairwise_excess_tests.csv", index=False)

    write_channel_definitions(APPENDIX_OUTPUT_DIR)
    write_results_dashboard(
        summary_df,
        raw_rate_operational_summary_df,
        step_diagnostics_df,
        raw_rate_condition_mean_trend_df,
        raw_rate_regression_df,
        OUTPUT_DIR / "RESULTS_DASHBOARD.md",
    )
    write_outputs_readme(OUTPUT_DIR / "README.md")
    write_main_figure_values_summary(
        raw_rate_regression_df,
        step_diagnostics_df,
        summary_df,
        OUTPUT_DIR / "main_figure_values_summary.txt",
    )
    make_plots(
        results_df,
        summary_df,
        round_df,
        raw_rate_regression_df,
        raw_rate_operational_summary_df,
        step_diagnostics_df,
    )

    print("\nSaved outputs to:", OUTPUT_DIR)
    print("Open regional_feedback_outputs/RESULTS_DASHBOARD.md for the main readout.")
    print("Appendix/stress-test diagnostics are under regional_feedback_outputs/appendix_diagnostics/.")


if __name__ == "__main__":
    main()
