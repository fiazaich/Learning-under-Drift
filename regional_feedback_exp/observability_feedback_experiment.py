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
BLIND_COL = "WarehouseCode"

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
BLIND_SCORE_BINS = 2
BLIND_RANDOM_BUCKETS = 4
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

CHANNEL_ORDER = ["blind", "coarse", "task_no_qty", "task_no_cost", "task_no_group", "task"]
CHANNEL_LABELS = {
    "blind": "Weak blind",
    "coarse": "Coarse score",
    "task_no_qty": "Task minus quantity",
    "task_no_cost": "Task minus cost",
    "task_no_group": "Task minus subgroup",
    "task": "Task-aligned",
}
CHANNEL_DEFINITIONS = {
    "blind": "2 predicted-price bins x 4 deterministic random row buckets; buckets are fixed from row identity and unrelated to feedback-targeted variables.",
    "coarse": "5 quantile bins of deployed predicted Unit Price, with bin edges fixed from D_0 baseline predictions.",
    "task_no_qty": "5 predicted-price bins x 4 Unit Cost bins x binary Sales Channel subgroup; omits Order Quantity.",
    "task_no_cost": "5 predicted-price bins x 4 Order Quantity bins x binary Sales Channel subgroup; omits Unit Cost.",
    "task_no_group": "5 predicted-price bins x 4 Order Quantity bins x 4 Unit Cost bins; omits Sales Channel subgroup.",
    "task": "5 predicted-price bins x 4 Order Quantity bins x 4 Unit Cost bins x binary Sales Channel subgroup; score bins match the coarse channel and all bins are fixed from D_0.",
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
    blind_score_edges: np.ndarray
    coarse_score_edges: np.ndarray
    task_score_edges: np.ndarray
    task_qty_edges: np.ndarray
    task_cost_edges: np.ndarray
    subgroup_ref: str
    sales_channel_categories: list[str]


def fit_channel_artifacts(df0: pd.DataFrame, preds0: np.ndarray) -> ChannelArtifacts:
    blind_score_edges = compute_quantile_edges(preds0, BLIND_SCORE_BINS)
    coarse_score_edges = compute_quantile_edges(preds0, COARSE_SCORE_BINS)
    task_score_edges = compute_quantile_edges(preds0, TASK_SCORE_BINS)
    task_qty_edges = compute_quantile_edges(df0[QUANTITY_COL].to_numpy(), TASK_QTY_BINS)
    task_cost_edges = compute_quantile_edges(df0[COST_COL].to_numpy(), TASK_COST_BINS)

    mode_value = df0[GROUP_COL].mode(dropna=True)
    subgroup_ref = str(mode_value.iloc[0] if len(mode_value) else df0[GROUP_COL].iloc[0])
    sales_channel_categories = sorted(df0[GROUP_COL].astype(str).unique())

    return ChannelArtifacts(
        blind_score_edges=blind_score_edges,
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


def blind_channel_probs(df: pd.DataFrame, preds: np.ndarray, artifacts: ChannelArtifacts) -> np.ndarray:
    score_bin = assign_bins(preds, artifacts.blind_score_edges)
    random_bin = fixed_random_buckets(len(df), BLIND_RANDOM_BUCKETS)
    state_ids = score_bin * BLIND_RANDOM_BUCKETS + random_bin
    n_states = (len(artifacts.blind_score_edges) - 1) * BLIND_RANDOM_BUCKETS
    return empirical_categorical_distribution(state_ids, n_states, PSEUDOCOUNT)


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
    "blind": blind_channel_probs,
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
    round_records_sink: list[dict] | None = None,
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
        preds_next_all_same_model = predict_model(model, df_next)

        y_current_all = df_current[TARGET_COL].to_numpy(dtype=float)
        y_train_current = train_df[TARGET_COL].to_numpy(dtype=float)
        y_eval_current = eval_df[TARGET_COL].to_numpy(dtype=float)
        y_next_all = df_next[TARGET_COL].to_numpy(dtype=float)
        y_eval_next = next_eval_df[TARGET_COL].to_numpy(dtype=float)

        train_mse = float(mean_squared_error(y_train_current, preds_train_current))
        current_all_mse = float(mean_squared_error(y_current_all, preds_current_all))
        next_all_mse = float(mean_squared_error(y_next_all, preds_next_all_same_model))
        eval_mse_current = float(mean_squared_error(y_eval_current, preds_eval_current))
        eval_mse_next = float(mean_squared_error(y_eval_next, preds_eval_next))
        current_rmse = float(np.sqrt(eval_mse_current))
        next_rmse = float(np.sqrt(eval_mse_next))

        # Delta_rep_T: eval-split one-step reproducibility gap.
        delta_rep = abs(eval_mse_current - eval_mse_next)
        # V_T: full-dataset within-step drift under the deployed predictor.
        v_t = abs(current_all_mse - next_all_mse)
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
                "delta_rep": delta_rep,
                "v_t": v_t,
                "sampling_candidate": sampling_candidate,
                **{f"fr_step_{channel}": fr_steps[channel] for channel in CHANNEL_ORDER},
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
            "current_all_mse",
            "next_all_mse",
            "eval_mse_current",
            "eval_mse_next",
            "delta_rep",
            "v_t",
            "sampling_candidate",
            *[f"fr_step_{channel}" for channel in CHANNEL_ORDER],
        ]
        rr[round_cols].to_csv(round_records_outpath, index=False)
    if round_records_sink is not None:
        round_records_sink.extend(round_records)

    return {
        "mu": mu,
        "seed": seed,
        "Delta_rep_T": float(rr["delta_rep"].mean()),
        "V_T": float(rr["v_t"].mean()),
        "mean_current_rmse": float(rr["current_rmse"].mean()),
        "mean_next_rmse": float(rr["next_rmse"].mean()),
        **{f"A_T_{channel}": float(rr[f"fr_step_{channel}"].sum()) for channel in CHANNEL_ORDER},
        **{f"A_rate_{channel}": float(rr[f"fr_step_{channel}"].mean()) for channel in CHANNEL_ORDER},
    }


# -----------------------------
# Plotting
# -----------------------------


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
        agg_spec[f"A_rate_{channel}_excess_mean"] = (f"A_rate_{channel}_excess", "mean")
        agg_spec[f"A_rate_{channel}_excess_std"] = (f"A_rate_{channel}_excess", "std")
    summary = results_df.groupby("mu").agg(**agg_spec).reset_index()
    return summary


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


def add_excess_fr_rates(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    baseline = df[df["mu"] == 0.0].set_index("seed")
    for channel in CHANNEL_ORDER:
        rate_col = f"A_rate_{channel}"
        baseline_by_seed = baseline[rate_col]
        df[f"{rate_col}_baseline"] = df["seed"].map(baseline_by_seed)
        df[f"{rate_col}_excess"] = df[rate_col] - df[f"{rate_col}_baseline"]
    return df


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


def compact_pairwise_excess_tests_table(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    compact = pairwise_df[
        [
            "mu",
            "comparison",
            "left_label",
            "right_label",
            "n_seeds",
            "mean_diff_right_minus_left",
            "bootstrap_ci95_low",
            "bootstrap_ci95_high",
            "permutation_p_two_sided",
        ]
    ].copy()
    compact = compact.rename(
        columns={
            "left_label": "baseline_side",
            "right_label": "comparison_side",
            "mean_diff_right_minus_left": "mean_diff_excess_fr",
            "bootstrap_ci95_low": "bootstrap_ci95_low_excess_fr",
            "bootstrap_ci95_high": "bootstrap_ci95_high_excess_fr",
            "permutation_p_two_sided": "permutation_p_value_two_sided",
        }
    )
    return compact


def write_channel_definitions() -> None:
    lines = [
        "This real-data experiment is an applied robustness / partial-observability check.",
        "It is not a direct contraction-theorem test and does not estimate intrinsic C_T/T.",
        "",
        "Channels:",
    ]
    for channel in CHANNEL_ORDER:
        lines.append(f"- {CHANNEL_LABELS[channel]}: {CHANNEL_DEFINITIONS[channel]}")
    (OUTPUT_DIR / "channel_definitions.txt").write_text("\n".join(lines), encoding="utf-8")


def make_plots(results_df: pd.DataFrame, summary_df: pd.DataFrame, round_df: pd.DataFrame) -> None:
    del round_df
    n_seeds = results_df["seed"].nunique()

    plot_mean_band(
        summary_df["mu"],
        summary_df["Delta_rep_T_mean"],
        standard_error(summary_df["Delta_rep_T_std"], n_seeds),
        xlabel="Feedback strength $\\mu$",
        ylabel=r"$\Delta_T^{\mathrm{rep}}$",
        outpath=OUTPUT_DIR / "figure_1_gap_vs_mu.png",
        color=PALETTE[0],
    )

    plot_mean_band(
        summary_df["mu"],
        summary_df["V_T_mean"],
        standard_error(summary_df["V_T_std"], n_seeds),
        xlabel="Feedback strength $\\mu$",
        ylabel=r"$V_T$",
        outpath=OUTPUT_DIR / "figure_2_drift_vs_mu.png",
        color=PALETTE[0],
    )

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    fig.set_size_inches(*FIGSIZE)
    x = summary_df["mu"].to_numpy(dtype=float)
    for channel, marker, label, color in [
        ("blind", "x", "Blind", PALETTE[0]),
        ("coarse", "o", "Coarse", PALETTE[1]),
        ("task", "s", "Task-aligned", PALETTE[2]),
    ]:
        y = summary_df[f"A_rate_{channel}_excess_mean"].to_numpy(dtype=float)
        yse = standard_error(summary_df[f"A_rate_{channel}_excess_std"], n_seeds).to_numpy(dtype=float)
        ax.plot(x, y, marker=marker, markersize=3.0, lw=0.9, color=color, label=label)
        ax.fill_between(x, y - yse, y + yse, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("Feedback strength $\\mu$")
    ax.set_ylabel(r"Excess observable FR rate")
    style_axis(ax)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save_figure_all_formats(fig, OUTPUT_DIR / "figure_3_excess_fr_rate_vs_mu.png")


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
            )
            results.append(result)

    results_df = add_excess_fr_rates(pd.DataFrame(results))
    round_df = pd.DataFrame(round_records)
    summary_df = summarize_results(results_df)

    results_df.to_csv(OUTPUT_DIR / "regional_feedback_results_by_seed.csv", index=False)
    round_df.to_csv(OUTPUT_DIR / "regional_feedback_rounds.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "regional_feedback_summary.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "table_1_summary.csv", index=False)

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
    primary_pairwise_tests_df = build_pairwise_excess_tests_table(
        results_df,
        comparisons=[
            ("blind_vs_coarse", "blind", "coarse"),
            ("coarse_vs_task", "coarse", "task"),
            ("blind_vs_task", "blind", "task"),
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
    primary_pairwise_compact_df = compact_pairwise_excess_tests_table(primary_pairwise_tests_df)
    ablation_pairwise_compact_df = compact_pairwise_excess_tests_table(ablation_pairwise_tests_df)
    corr_df.to_csv(OUTPUT_DIR / "regional_feedback_correlations.csv", index=False)
    corr_df.to_csv(OUTPUT_DIR / "table_2_correlations.csv", index=False)
    channel_comparison_df.to_csv(OUTPUT_DIR / "table_2_channel_comparison.csv", index=False)
    primary_pairwise_tests_df.to_csv(OUTPUT_DIR / "table_3_pairwise_excess_tests.csv", index=False)
    ablation_pairwise_tests_df.to_csv(OUTPUT_DIR / "table_4_ablation_pairwise_excess_tests.csv", index=False)
    primary_pairwise_compact_df.to_csv(OUTPUT_DIR / "appendix_pairwise_channel_contrasts.csv", index=False)
    ablation_pairwise_compact_df.to_csv(OUTPUT_DIR / "appendix_pairwise_ablation_contrasts.csv", index=False)

    write_channel_definitions()
    make_plots(results_df, summary_df, round_df)

    print("\nSaved outputs to:", OUTPUT_DIR)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print("\nChannel comparison:")
    print(channel_comparison_df.to_string(index=False))
    print("\nPairwise excess Fisher tests:")
    print(primary_pairwise_compact_df.to_string(index=False))
    print("\nAblation pairwise excess Fisher tests:")
    print(ablation_pairwise_compact_df.to_string(index=False))


if __name__ == "__main__":
    main()
