
import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
DEVICE = torch.device("cpu")
DEFAULT_HIDDEN = 128


@dataclass
class TimingConfig:
    d_in: int = 5
    simulation_seeds: Sequence[int] = tuple(range(4))
    allocation_seeds: Sequence[int] = tuple(range(4))
    k_policy: float = 0.25
    sigma: float = 0.1
    C_pol: float = 8.0
    T_grid: Sequence[int] = (800, 1600, 3200, 6400)
    drift_ratios: Sequence[float] = (0.0, 0.0025, 0.005, 0.01, 0.015)
    gamma_grid: Sequence[float] = (0.0, 0.001, 0.0025, 0.005)
    allocation_families: Sequence[str] = ("uniform", "early", "late", "bursty")
    n_bins: int = 8
    hidden_dim: int = DEFAULT_HIDDEN
    N_pop: int = 2048
    eval_every: int = 10


def create_run_dir(tag: str) -> Path:
    root = Path(__file__).resolve().parent / "runs"
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = root / f"{tag}_{stamp}"
    path.mkdir(parents=False, exist_ok=False)
    return path


def make_feature_map(d_in=5, m_feat=64, seed=123):
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=1.0 / np.sqrt(d_in), size=(m_feat, d_in)).astype(np.float32)
    b = rng.normal(scale=0.1, size=(m_feat,)).astype(np.float32)
    W = torch.from_numpy(W)
    b = torch.from_numpy(b)

    def phi(x):
        return torch.tanh(x @ W.T + b)

    return phi, m_feat


@torch.no_grad()
def fisher_gram(phi_fn, Xp, sigma=0.1, ridge=1e-6):
    Phi = phi_fn(Xp)
    G = (Phi.T @ Phi) / (Xp.shape[0] * (sigma**2))
    return G + ridge * torch.eye(G.shape[0], dtype=G.dtype)


@torch.no_grad()
def fisher_norm(G, delta_theta):
    q_sq = float((delta_theta @ (G @ delta_theta)).item())
    if not np.isfinite(q_sq) or q_sq <= 0.0:
        return 0.0
    return math.sqrt(q_sq)


class FisherBudget:
    def __init__(self, total):
        self.remaining = float(max(0.0, total))

    def take(self, G, direction, proposed_len):
        if self.remaining <= 0.0 or proposed_len <= 0.0:
            return torch.zeros_like(direction), 0.0
        q_sq = float((direction @ (G @ direction)).item())
        if not np.isfinite(q_sq) or q_sq <= 1e-16:
            return torch.zeros_like(direction), 0.0
        q = math.sqrt(q_sq)
        target = min(float(proposed_len), self.remaining)
        step = (target / q) * direction
        actual_len = fisher_norm(G, step)
        if actual_len <= 0.0:
            return torch.zeros_like(direction), 0.0
        self.remaining = max(0.0, self.remaining - actual_len)
        return step, actual_len


class MLP(nn.Module):
    def __init__(self, d_in=5, h=128):
        super().__init__()
        self.l1 = nn.Linear(d_in, h)
        self.l2 = nn.Linear(h, 1)
        self.reset()

    def reset(self):
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.zeros_(self.l1.bias)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.zeros_(self.l2.bias)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


def make_sampler(d_in=5, seed=0):
    rng = np.random.default_rng(seed)

    def sample_x(N=1):
        return torch.from_numpy(rng.normal(size=(N, d_in)).astype(np.float32))

    return sample_x


@torch.no_grad()
def pop_risk_pair_mse(f, theta, theta_next, phi_fn, sampler_pop, N=256, sigma=0.1):
    X = sampler_pop(N=N).to(DEVICE)
    Phi = phi_fn(X)
    y_theta = (Phi @ theta).view(-1, 1)
    y_next = (Phi @ theta_next).view(-1, 1)
    y_pred = f(X)
    R_t = float(F.mse_loss(y_pred, y_theta).item() + sigma**2)
    R_plus = float(F.mse_loss(y_pred, y_next).item() + sigma**2)
    return R_t, R_plus


def allocation_rng(seed: int, family_index: int, channel: int) -> np.random.Generator:
    return np.random.default_rng(1_000_003 + 10_007 * seed + 101 * family_index + channel)


def sample_bin_weights(n_bins: int, family: str, seed: int, family_index: int, channel: int):
    rng = allocation_rng(seed=seed, family_index=family_index, channel=channel)
    tau = (np.arange(int(n_bins), dtype=float) + 0.5) / float(n_bins)
    family = str(family)
    if family == "uniform":
        profile = np.ones(int(n_bins), dtype=float)
        concentration = 200.0
    elif family == "early":
        profile = np.exp(-3.0 * tau)
        concentration = 80.0
    elif family == "late":
        profile = np.exp(3.0 * tau)
        concentration = 80.0
    elif family == "bursty":
        profile = np.ones(int(n_bins), dtype=float)
        concentration = 0.3
    else:
        raise ValueError(f"Unknown allocation family: {family}")
    profile = profile / float(np.sum(profile))
    alpha = np.maximum(concentration * profile, 1e-6)
    return rng.dirichlet(alpha)


def schedule_from_bin_weights(T: int, bin_weights: np.ndarray) -> np.ndarray:
    bin_weights = np.asarray(bin_weights, dtype=float)
    edges = np.linspace(0, int(T), len(bin_weights) + 1, dtype=int)
    schedule = np.zeros(int(T), dtype=float)
    for j, weight in enumerate(bin_weights):
        start, stop = int(edges[j]), int(edges[j + 1])
        n = max(1, stop - start)
        schedule[start:stop] = float(T) * float(weight) / float(n)
    return schedule * (float(T) / float(np.sum(schedule)))


def timing_features(bin_weights: np.ndarray) -> dict:
    w = np.asarray(bin_weights, dtype=float)
    w = w / float(np.sum(w))
    tau = (np.arange(len(w), dtype=float) + 0.5) / float(len(w))
    entropy = -float(np.sum(w * np.log(np.maximum(w, 1e-300)))) / math.log(float(len(w)))
    center = float(np.sum(tau * w))
    return {"center": center, "entropy": entropy}


def combined_timing_features(exo_weights, endo_weights, C_exo: float, C_endo: float) -> dict:
    total = float(C_exo + C_endo)
    if total <= 0.0:
        w = np.ones_like(exo_weights, dtype=float) / float(len(exo_weights))
    else:
        w = (float(C_exo) * exo_weights + float(C_endo) * endo_weights) / total
    out = timing_features(w)
    out["combined_weights"] = w
    return out


def one_run(
    T,
    cfg: TimingConfig,
    phi_fn,
    m_feat,
    G_fisher,
    Xrisk,
    Phi_risk,
    C_exo,
    gamma,
    simulation_seed,
    exo_schedule,
    endo_schedule,
):
    torch.manual_seed(simulation_seed)
    np.random.seed(simulation_seed)

    theta = torch.zeros(m_feat, dtype=torch.float32)
    f = MLP(d_in=cfg.d_in, h=cfg.hidden_dim).to(DEVICE)
    opt = torch.optim.SGD(f.parameters(), lr=0.05)

    sample_x = make_sampler(d_in=cfg.d_in, seed=10_000 + simulation_seed)
    rng_exo = np.random.default_rng(simulation_seed + 7)

    exo_budget = FisherBudget(C_exo)
    C_endo = float(max(0.0, gamma * cfg.k_policy * cfg.C_pol * T))
    endo_budget = FisherBudget(C_endo)
    mean_len_exo = (C_exo / T) if C_exo > 0 else 0.0
    mean_len_endo = (C_endo / T) if C_endo > 0 else 0.0

    pop_mse, pop_plus, VT_terms = [], [], []
    train_all = []
    sum_dt = 0.0
    sum_kappa = 0.0
    sum_exo_sq = 0.0
    sum_endo_sq = 0.0
    sum_exo_prop_sq = 0.0
    sum_endo_prop_sq = 0.0
    sum_total_motion = 0.0
    sum_total_sq = 0.0
    max_exo_step = 0.0
    max_endo_step = 0.0
    max_total_step = 0.0

    for t in range(1, int(T) + 1):
        x = sample_x(N=1).to(DEVICE)
        with torch.no_grad():
            y = (phi_fn(x) @ theta).view(-1, 1) + cfg.sigma * torch.randn(1, 1)

        opt.zero_grad()
        yhat = f(x)
        train_loss = F.mse_loss(yhat, y)
        train_all.append(float(train_loss.item()))

        Xprobe = sample_x(N=256).to(DEVICE)
        Phi_probe = phi_fn(Xprobe)

        dir_exo = torch.from_numpy(rng_exo.normal(size=(m_feat,)).astype(np.float32))
        prop_len_exo = float(mean_len_exo * exo_schedule[t - 1]) if mean_len_exo > 0 else 0.0
        step_exo, len_exo = exo_budget.take(G_fisher, dir_exo, prop_len_exo)

        step_endo = torch.zeros_like(theta)
        len_endo = 0.0
        prop_len_endo = 0.0
        if endo_budget.remaining > 0:
            with torch.no_grad():
                f_probe = f(Xprobe).view(-1)
                g_probe = (Phi_probe @ theta).view(-1)
                resid = f_probe - g_probe
                grad_theta = -2.0 * (Phi_probe.T @ resid) / Xprobe.shape[0]

            grad_norm = float(grad_theta.norm().item())
            if grad_norm > 0:
                dir_endo = grad_theta / grad_norm
            else:
                dir_endo = torch.zeros_like(theta)

            prop_len_endo = float(mean_len_endo * endo_schedule[t - 1])
            step_endo, len_endo = endo_budget.take(G_fisher, dir_endo, prop_len_endo)

        theta_next = theta + step_exo + step_endo
        len_total = fisher_norm(G_fisher, step_exo + step_endo)

        with torch.no_grad():
            y_pred = f(Xrisk).view(-1)
            g_t = (Phi_risk @ theta).view(-1)
            g_next = (Phi_risk @ theta_next).view(-1)
            resid_t = y_pred - g_t
            resid_next = y_pred - g_next
            R_t = float(torch.mean(resid_t * resid_t).item() + cfg.sigma**2)
            R_plus = float(torch.mean(resid_next * resid_next).item() + cfg.sigma**2)
            pop_mse.append(R_t)
            pop_plus.append(R_plus)
            VT_terms.append(abs(R_plus - R_t))

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), max_norm=1.0)
        opt.step()
        theta = theta_next

        sum_dt += float(len_exo)
        sum_kappa += float(len_endo)
        sum_exo_sq += float(len_exo * len_exo)
        sum_endo_sq += float(len_endo * len_endo)
        sum_exo_prop_sq += float(prop_len_exo * prop_len_exo)
        sum_endo_prop_sq += float(prop_len_endo * prop_len_endo)
        sum_total_motion += float(len_total)
        sum_total_sq += float(len_total * len_total)
        max_exo_step = max(max_exo_step, float(len_exo))
        max_endo_step = max(max_endo_step, float(len_endo))
        max_total_step = max(max_total_step, float(len_total))

    Xeval = make_sampler(d_in=cfg.d_in, seed=20_000 + simulation_seed)(N=1024).to(DEVICE)
    with torch.no_grad():
        err_T = float(torch.mean(torch.abs(f(Xeval).view(-1) - (phi_fn(Xeval) @ theta).view(-1))).item())

    Rhat_T = float(np.mean(train_all))
    R_T = float(np.mean(pop_mse))
    Rplus_T = float(np.mean(pop_plus))
    V_T = float(np.mean(VT_terms))
    traj_emp_legacy = float(np.mean(train_all))

    return {
        "err_T": err_T,
        "sum_dt": float(sum_dt),
        "sum_kappa": float(sum_kappa),
        "sum_exo_sq": float(sum_exo_sq),
        "sum_endo_sq": float(sum_endo_sq),
        "sum_exo_prop_sq": float(sum_exo_prop_sq),
        "sum_endo_prop_sq": float(sum_endo_prop_sq),
        "sum_total_motion": float(sum_total_motion),
        "sum_total_sq": float(sum_total_sq),
        "A_T_over_T": float(sum_total_motion / T),
        "component_budget_over_T": float((sum_dt + sum_kappa) / T),
        "triangle_slack_over_T": float(max(0.0, sum_dt + sum_kappa - sum_total_motion) / T),
        "max_exo_step": float(max_exo_step),
        "max_endo_step": float(max_endo_step),
        "max_total_step": float(max_total_step),
        "delta_preq": float(abs(Rhat_T - Rplus_T)),
        "delta_sam": float(abs(Rhat_T - R_T)),
        "V_T": V_T,
        "Rhat_T": Rhat_T,
        "R_T": R_T,
        "Rplus_T": Rplus_T,
        "traj_emp_risk_legacy": traj_emp_legacy,
        "gen_gap_traj_legacy": float(abs(traj_emp_legacy - R_T)),
        "exo_used_ratio": float(sum_dt / C_exo) if C_exo > 0 else 1.0,
        "endo_used_ratio": float(sum_kappa / C_endo) if C_endo > 0 else 1.0,
    }


def mean_se(vals):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return mean, se, n


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(cfg: TimingConfig, output_tag: str):
    run_dir = create_run_dir(output_tag)
    print(f"Writing balanced timing run to: {run_dir}", flush=True)

    phi_fn, m_feat = make_feature_map(d_in=cfg.d_in, m_feat=64, seed=123)
    Xrisk = make_sampler(d_in=cfg.d_in, seed=42)(N=cfg.N_pop).to(DEVICE)
    Phi_risk = phi_fn(Xrisk)
    G_fisher = (Phi_risk.T @ Phi_risk) / (Xrisk.shape[0] * (cfg.sigma**2))
    G_fisher = G_fisher + 1e-6 * torch.eye(G_fisher.shape[0], dtype=G_fisher.dtype)

    raw_rows = []
    for T in cfg.T_grid:
        for ratio_i, ratio in enumerate(cfg.drift_ratios):
            C_exo_total = float(ratio * T)
            for gamma_i, gamma in enumerate(cfg.gamma_grid):
                C_endo_total = float(max(0.0, gamma * cfg.k_policy * cfg.C_pol * T))
                for family_i, family in enumerate(cfg.allocation_families):
                    label = str(family)
                    print(
                        f"Running T={T}, C_exo_ratio={ratio:g}, gamma={gamma:g}, allocation={label}",
                        flush=True,
                    )
                    for allocation_seed in cfg.allocation_seeds:
                        exo_bin_w = sample_bin_weights(
                            cfg.n_bins,
                            family,
                            seed=allocation_seed,
                            family_index=family_i,
                            channel=17,
                        )
                        endo_bin_w = sample_bin_weights(
                            cfg.n_bins,
                            family,
                            seed=allocation_seed,
                            family_index=family_i,
                            channel=53,
                        )
                        exo_schedule = schedule_from_bin_weights(T, exo_bin_w)
                        endo_schedule = schedule_from_bin_weights(T, endo_bin_w)

                        exo_feat = timing_features(exo_bin_w)
                        endo_feat = timing_features(endo_bin_w)
                        combo_feat = combined_timing_features(
                            exo_bin_w, endo_bin_w, C_exo=C_exo_total, C_endo=C_endo_total
                        )

                        for simulation_seed in cfg.simulation_seeds:
                            metrics = one_run(
                                T=T,
                                cfg=cfg,
                                phi_fn=phi_fn,
                                m_feat=m_feat,
                                G_fisher=G_fisher,
                                Xrisk=Xrisk,
                                Phi_risk=Phi_risk,
                                C_exo=C_exo_total,
                                gamma=gamma,
                                simulation_seed=simulation_seed,
                                exo_schedule=exo_schedule,
                                endo_schedule=endo_schedule,
                            )

                            row = {
                                "T": int(T),
                                "C_exo_total": C_exo_total,
                                "C_exo_ratio": float(ratio),
                                "gamma": float(gamma),
                                "C_endo_total": C_endo_total,
                                "simulation_seed": int(simulation_seed),
                                "allocation_seed": int(allocation_seed),
                                "n_bins": int(cfg.n_bins),
                                "allocation_family": label,
                                "exo_center": exo_feat["center"],
                                "exo_entropy": exo_feat["entropy"],
                                "endo_center": endo_feat["center"],
                                "endo_entropy": endo_feat["entropy"],
                                "timing_center": combo_feat["center"],
                                "timing_entropy": combo_feat["entropy"],
                                "exo_schedule_cv": float(np.std(exo_schedule) / (np.mean(exo_schedule) + 1e-12)),
                                "endo_schedule_cv": float(np.std(endo_schedule) / (np.mean(endo_schedule) + 1e-12)),
                            }
                            row.update(metrics)
                            raw_rows.append(row)

    raw_fields = [
        "T",
        "C_exo_total",
        "C_exo_ratio",
        "gamma",
        "C_endo_total",
        "simulation_seed",
        "allocation_seed",
        "n_bins",
        "allocation_family",
        "exo_center",
        "exo_entropy",
        "endo_center",
        "endo_entropy",
        "timing_center",
        "timing_entropy",
        "exo_schedule_cv",
        "endo_schedule_cv",
        "err_T",
        "sum_dt",
        "sum_kappa",
        "sum_exo_sq",
        "sum_endo_sq",
        "sum_exo_prop_sq",
        "sum_endo_prop_sq",
        "sum_total_motion",
        "sum_total_sq",
        "A_T_over_T",
        "component_budget_over_T",
        "triangle_slack_over_T",
        "max_exo_step",
        "max_endo_step",
        "max_total_step",
        "delta_preq",
        "delta_sam",
        "V_T",
        "Rhat_T",
        "R_T",
        "Rplus_T",
        "traj_emp_risk_legacy",
        "gen_gap_traj_legacy",
        "exo_used_ratio",
        "endo_used_ratio",
    ]
    write_csv(run_dir / "balanced_timing_raw.csv", raw_rows, raw_fields)
    write_summary(run_dir, raw_rows)

    meta = {
        "experiment": "balanced_timing_allocation_nn",
        "description": "Cross horizon, total budgets, and Dirichlet temporal allocation families.",
        "T_grid": list(map(int, cfg.T_grid)),
        "drift_ratios": list(map(float, cfg.drift_ratios)),
        "gamma_grid": list(map(float, cfg.gamma_grid)),
        "allocation_families": list(map(str, cfg.allocation_families)),
        "n_bins": int(cfg.n_bins),
        "simulation_seeds": list(map(int, cfg.simulation_seeds)),
        "allocation_seeds": list(map(int, cfg.allocation_seeds)),
        "N_pop": int(cfg.N_pop),
        "eval_every": int(cfg.eval_every),
        "hidden_dim": int(cfg.hidden_dim),
        "output_location": str(run_dir),
        "allocation_seed_note": "Allocation seeds do not include T, C_exo_ratio, gamma, or simulation_seed, so timing patterns are crossed orthogonally with horizon, budgets, and learner/data randomness.",
        "population_note": "The same fixed empirical population is used to define the Fisher Gram matrix and to evaluate population risks in every run.",
        "local_regime_note": "Default budget grids are reduced to keep max_total_step in a local regime; verify max_total_step in diagnostics after each run.",
    }
    with open(run_dir / "balanced_timing_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    analyze_run(run_dir)
    print(f"Saved balanced timing outputs to: {run_dir}", flush=True)
    return run_dir


def write_summary(run_dir: Path, raw_rows: list[dict]):
    group_keys = ["T", "C_exo_ratio", "gamma", "allocation_family"]
    value_cols = [
        "err_T",
        "sum_dt",
        "sum_kappa",
        "sum_exo_sq",
        "sum_endo_sq",
        "sum_exo_prop_sq",
        "sum_endo_prop_sq",
        "sum_total_motion",
        "sum_total_sq",
        "A_T_over_T",
        "component_budget_over_T",
        "triangle_slack_over_T",
        "max_exo_step",
        "max_endo_step",
        "max_total_step",
        "delta_preq",
        "delta_sam",
        "V_T",
        "Rhat_T",
        "R_T",
        "Rplus_T",
        "timing_center",
        "timing_entropy",
        "exo_center",
        "exo_entropy",
        "endo_center",
        "endo_entropy",
        "exo_schedule_cv",
        "endo_schedule_cv",
        "exo_used_ratio",
        "endo_used_ratio",
    ]
    grouped = defaultdict(lambda: {k: [] for k in value_cols})
    key_values = {}
    for row in raw_rows:
        key = tuple(row[k] for k in group_keys)
        key_values[key] = {k: row[k] for k in group_keys}
        for col in value_cols:
            grouped[key][col].append(row[col])

    summary_rows = []
    for key in sorted(grouped):
        out = dict(key_values[key])
        for col in value_cols:
            m, se, n = mean_se(grouped[key][col])
            out[f"mean_{col}"] = m
            out[f"se_{col}"] = se
            out["n"] = n
        summary_rows.append(out)

    fields = group_keys + ["n"]
    for col in value_cols:
        fields.extend([f"mean_{col}", f"se_{col}"])
    write_csv(run_dir / "balanced_timing_summary.csv", summary_rows, fields)


def load_csv_dicts(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        converted = {}
        for k, v in row.items():
            if k in ("allocation_label", "allocation_family"):
                converted[k] = v
            else:
                try:
                    converted[k] = float(v)
                except ValueError:
                    converted[k] = v
        out.append(converted)
    return out


def split_masks(rows: list[dict]) -> dict[str, np.ndarray]:
    T = np.array([r["T"] for r in rows], dtype=float)
    exo = np.array([r["C_exo_ratio"] for r in rows], dtype=float)
    gamma = np.array([r["gamma"] for r in rows], dtype=float)
    simulation_seed = np.array([r["simulation_seed"] for r in rows], dtype=float)
    alloc_seed = np.array([r["allocation_seed"] for r in rows], dtype=float)
    family = np.array([r["allocation_family"] for r in rows], dtype=object)

    max_T = float(np.max(T))
    max_exo = float(np.max(exo))
    max_simulation_seed = float(np.max(simulation_seed))
    max_alloc_seed = float(np.max(alloc_seed))
    unique_gamma = sorted(set(float(g) for g in gamma))
    hold_policy = gamma == float(np.max(gamma)) if len(unique_gamma) > 2 else np.zeros(len(rows), dtype=bool)

    masks = {
        "heldout_horizon": T == max_T,
        "heldout_exo_budget": exo == max_exo,
        "heldout_policy_budget": hold_policy,
        "heldout_allocation_family_late": family == "late",
        "heldout_simulation_seed": simulation_seed == max_simulation_seed,
        "heldout_allocation_seed": alloc_seed == max_alloc_seed,
    }
    holdout_any = np.zeros(len(rows), dtype=bool)
    for mask in masks.values():
        holdout_any |= mask
    masks["calibration"] = ~holdout_any
    masks["heldout_any"] = holdout_any
    return masks


def ratio_stats(y: np.ndarray, bound: np.ndarray, mask: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    bound = np.asarray(bound, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if int(np.sum(mask)) == 0:
        return {
            "n": 0,
            "coverage": float("nan"),
            "max_ratio": float("nan"),
            "q95_ratio": float("nan"),
            "median_ratio": float("nan"),
            "violations": 0,
        }
    yy = y[mask]
    bb = bound[mask]
    ratios = np.divide(yy, bb, out=np.full_like(yy, np.inf, dtype=float), where=bb > 1e-15)
    zero_ok = (bb <= 1e-15) & (yy <= 1e-15)
    ratios[zero_ok] = 0.0
    covered = yy <= bb * (1.0 + 1e-10)

    def quantile_allow_inf(values: np.ndarray, q: float) -> float:
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return float("inf")
        finite_fraction = len(finite) / len(values)
        if q > finite_fraction:
            return float("inf")
        return float(np.quantile(finite, min(1.0, q / finite_fraction)))

    return {
        "n": int(len(yy)),
        "coverage": float(np.mean(covered)),
        "max_ratio": float(np.max(ratios)),
        "q95_ratio": quantile_allow_inf(ratios, 0.95),
        "median_ratio": quantile_allow_inf(ratios, 0.5),
        "violations": int(np.sum(~covered)),
    }


def calibrate_scale(y: np.ndarray, proxy: np.ndarray, mask: np.ndarray) -> float:
    yy = np.asarray(y, dtype=float)[mask]
    pp = np.asarray(proxy, dtype=float)[mask]
    bad = (pp <= 1e-15) & (yy > 1e-15)
    if bool(np.any(bad)):
        return float("inf")
    valid = pp > 1e-15
    if not bool(np.any(valid)):
        return 0.0
    return float(np.max(yy[valid] / pp[valid]))


def calibrate_remainder_scale(y: np.ndarray, linear_proxy: np.ndarray, quad_proxy: np.ndarray, mask: np.ndarray, c_rem: float) -> float:
    residual = np.maximum(0.0, np.asarray(y, dtype=float) - float(c_rem) * np.asarray(quad_proxy, dtype=float))
    return calibrate_scale(residual, linear_proxy, mask)


def calibrate_decomposed_alpha(y: np.ndarray, dbar: np.ndarray, kbar: np.ndarray, mask: np.ndarray) -> dict:
    alpha_grid = np.array(
        [0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        dtype=float,
    )
    best = None
    for alpha in alpha_grid:
        proxy = dbar + alpha * kbar
        scale = calibrate_scale(y, proxy, mask)
        if not np.isfinite(scale):
            continue
        bound = scale * proxy
        score = float(np.mean(bound[mask]))
        if best is None or score < best["score"]:
            best = {"alpha": float(alpha), "scale": float(scale), "score": score, "proxy": proxy, "bound": bound}
    if best is None:
        proxy = dbar + kbar
        return {"alpha": 1.0, "scale": float("inf"), "score": float("inf"), "proxy": proxy, "bound": np.inf * proxy}
    return best


def envelope_rows_for_model(name: str, y: np.ndarray, bound: np.ndarray, masks: dict[str, np.ndarray], params: dict) -> list[dict]:
    out = []
    for split_name in [
        "calibration",
        "heldout_any",
        "heldout_horizon",
        "heldout_exo_budget",
        "heldout_policy_budget",
        "heldout_allocation_family_late",
        "heldout_simulation_seed",
        "heldout_allocation_seed",
    ]:
        stats = ratio_stats(y, bound, masks[split_name])
        row = {"model": name, "split": split_name}
        row.update(params)
        row.update(stats)
        out.append(row)
    return out


def analyze_envelopes(rows: list[dict]) -> tuple[list[dict], dict]:
    y = np.array([r["V_T"] for r in rows], dtype=float)
    dbar = np.array([r["sum_dt"] / r["T"] for r in rows], dtype=float)
    kbar = np.array([r["sum_kappa"] / r["T"] for r in rows], dtype=float)
    Abar = np.array([r["A_T_over_T"] for r in rows], dtype=float)
    component_bar = np.array([r["component_budget_over_T"] for r in rows], dtype=float)
    Qbar = np.array([r["sum_total_sq"] / r["T"] for r in rows], dtype=float)
    masks = split_masks(rows)
    cal = masks["calibration"]
    if int(np.sum(cal)) == 0:
        raise RuntimeError("Calibration split is empty; relax holdout rules.")

    envelope_rows = []

    risk_scale = calibrate_scale(y, Abar, cal)
    if np.isfinite(risk_scale):
        path_bound = risk_scale * Abar
        component_bound = risk_scale * component_bar
    else:
        path_bound = np.where(Abar > 1e-15, np.inf, 0.0)
        component_bound = np.where(component_bar > 1e-15, np.inf, 0.0)

    envelope_rows.extend(
        envelope_rows_for_model(
            "actual_path_A_T_over_T",
            y,
            path_bound,
            masks,
            {"scale": risk_scale, "alpha": float("nan")},
        )
    )
    envelope_rows.extend(
        envelope_rows_for_model(
            "component_sum_dbar_plus_kbar",
            y,
            component_bound,
            masks,
            {"scale": risk_scale, "alpha": 1.0},
        )
    )

    c_rem = 1.0
    rem_scale = calibrate_remainder_scale(y, component_bar, Qbar, cal, c_rem)
    if np.isfinite(rem_scale):
        rem_bound = rem_scale * component_bar + c_rem * Qbar
    else:
        rem_bound = np.where(component_bar > 1e-15, np.inf, c_rem * Qbar)
    envelope_rows.extend(
        envelope_rows_for_model(
            "remainder_aware_component_plus_Q",
            y,
            rem_bound,
            masks,
            {"scale": rem_scale, "alpha": 1.0, "c_rem": c_rem},
        )
    )

    decomp = calibrate_decomposed_alpha(y, dbar, kbar, cal)
    envelope_rows.extend(
        envelope_rows_for_model(
            "decomposed_dbar_plus_alpha_kbar",
            y,
            decomp["bound"],
            masks,
            {"scale": decomp["scale"], "alpha": decomp["alpha"]},
        )
    )

    locality = {
        "max_total_step_all": float(np.max([r["max_total_step"] for r in rows])),
        "q95_max_total_step_all": float(np.quantile([r["max_total_step"] for r in rows], 0.95)),
        "median_triangle_slack_over_T": float(np.median([r["triangle_slack_over_T"] for r in rows])),
        "max_triangle_slack_over_T": float(np.max([r["triangle_slack_over_T"] for r in rows])),
        "n_calibration": int(np.sum(masks["calibration"])),
        "n_heldout_any": int(np.sum(masks["heldout_any"])),
        "exo_only_uncontrolled_calibration": int(np.sum((dbar[cal] <= 1e-15) & (y[cal] > 1e-15))),
        "policy_only_uncontrolled_calibration": int(np.sum((kbar[cal] <= 1e-15) & (y[cal] > 1e-15))),
        "risk_scale_calibrated_on_A_T_over_T": float(risk_scale),
        "remainder_scale_calibrated_on_component_budget": float(rem_scale),
        "remainder_c_fixed": float(c_rem),
    }
    return envelope_rows, locality


def write_envelope_diagnostics(run_dir: Path, envelope_rows: list[dict], locality: dict):
    fields = ["model", "split", "scale", "alpha", "c_rem", "n", "coverage", "violations", "median_ratio", "q95_ratio", "max_ratio"]
    write_csv(run_dir / "balanced_timing_envelope_summary.csv", envelope_rows, fields)

    path = run_dir / "balanced_timing_diagnostics.txt"
    with open(path, "w") as f:
        f.write("Balanced timing-allocation envelope diagnostics\n")
        f.write("Primary target: V_T upper control, not regression equality.\n")
        f.write("Constants are calibrated only on the calibration split and transported without refit.\n")
        f.write("Held-out conditions are the union of largest horizon, largest exogenous budget,")
        f.write(" largest policy budget when available, late allocation family, held-out simulation seed,")
        f.write(" and held-out allocation seed.\n")
        f.write("The same calibrated risk scale is used for A_T/T and sum_dt/T + sum_kappa/T.\n")
        f.write("The remainder-aware envelope adds fixed c_rem=1 times sum_total_sq/T and calibrates only the remaining linear scale.\n")
        f.write("The fitted-alpha envelope is exploratory empirical tightening only; it is not part of the principal figure.\n\n")
        f.write("Locality / path diagnostics:\n")
        for key, value in locality.items():
            f.write(f"  {key} = {value:.6g}\n")
        f.write("\nEnvelope results:\n")
        for row in envelope_rows:
            if row["split"] not in ("calibration", "heldout_any"):
                continue
            prefix = "exploratory " if row["model"] == "decomposed_dbar_plus_alpha_kbar" else ""
            alpha_txt = "" if not np.isfinite(row["alpha"]) else f", alpha={row['alpha']:.6g}"
            c_rem_txt = ""
            if "c_rem" in row and np.isfinite(row["c_rem"]):
                c_rem_txt = f", c_rem={row['c_rem']:.6g}"
            f.write(
                f"{prefix}{row['model']} [{row['split']}]: n={row['n']}, "
                f"scale={row['scale']:.6g}{alpha_txt}{c_rem_txt}, coverage={row['coverage']:.4f}, "
                f"violations={row['violations']}, q95_ratio={row['q95_ratio']:.4f}, "
                f"max_ratio={row['max_ratio']:.4f}\n"
            )


def plot_envelope_ratios(run_dir: Path, envelope_rows: list[dict]):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    try:
        import scienceplots

        plt.style.use(["science", "ieee"])
    except Exception:
        pass

    mpl.rcParams.update(
        {
            "axes.titlesize": 12.0,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.8,
        }
    )

    split_order = [
        ("heldout_any", "Union"),
        ("heldout_horizon", "Largest T"),
        ("heldout_exo_budget", "Largest exo\nbudget"),
        ("heldout_policy_budget", "Largest\nfeedback"),
        ("heldout_allocation_family_late", "Late\ntiming"),
        ("heldout_simulation_seed", "Held-out sim\nseed"),
        ("heldout_allocation_seed", "Held-out alloc\nseed"),
    ]
    split_labels = dict(split_order)
    model_specs = [
        ("component_sum_dbar_plus_kbar", r"(a) Leading-order envelope", r"$V_T/\widehat{U}_T^{\mathrm{lead}}$"),
        ("remainder_aware_component_plus_Q", r"(b) Remainder-aware envelope", r"$V_T/\widehat{U}_T^{\mathrm{rem}}$"),
    ]
    panel_rows = []
    all_max_ratio = []
    for model_name, _, _ in model_specs:
        heldout = [
            r
            for r in envelope_rows
            if r["model"] == model_name and r["split"] in split_labels
        ]
        by_split = {r["split"]: r for r in heldout}
        rows = [by_split[s] for s, _ in split_order if s in by_split]
        panel_rows.append(rows)
        all_max_ratio.extend(float(r["max_ratio"]) for r in rows)

    labels = [label for split, label in split_order if split in {r["split"] for r in panel_rows[0]}]
    x = np.arange(len(labels))

    palette = {"q95": "#1f77b4", "max": "#ff7f0e"}
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.15))
    width = 0.34
    ylim_top = max(1.08, 1.08 * max(all_max_ratio))
    legend_handles = None
    legend_labels = None

    for ax, rows, (_, title, ylabel) in zip(axes, panel_rows, model_specs):
        q95 = [float(r["q95_ratio"]) for r in rows]
        max_ratio = [float(r["max_ratio"]) for r in rows]
        coverage = [float(r["coverage"]) for r in rows]
        ax.bar(x - width / 2, q95, width=width, color=palette["q95"], label="95th percentile")
        ax.bar(x + width / 2, max_ratio, width=width, color=palette["max"], label="maximum")
        ax.axhline(1.0, color="0.25", linewidth=1.0)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, ylim_top)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        y_text = -0.29
        ax.text(
            -0.48,
            y_text,
            "coverage",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=10.5,
            clip_on=False,
        )
        for xi, cov in zip(x, coverage):
            ax.text(
                xi,
                y_text,
                f"{100.0 * cov:.1f}\\%",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10.5,
                clip_on=False,
            )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        borderaxespad=0.0,
        frameon=False,
        ncol=2,
    )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.15, hspace=0.64)
    for stem in ["fig_envelope_heldout_ratios", "fig_nn_envelope_by_holdout_split"]:
        fig.savefig(run_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(run_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_union_and_violation_localization(run_dir: Path, envelope_rows: list[dict], rows: list[dict]):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    try:
        import scienceplots

        plt.style.use(["science", "ieee"])
    except Exception:
        pass

    mpl.rcParams.update(
        {
            "axes.titlesize": 12.0,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10.5,
        }
    )

    def get_env(model: str, split: str) -> dict:
        matches = [r for r in envelope_rows if r["model"] == model and r["split"] == split]
        if not matches:
            raise KeyError(f"Missing envelope row for {model} / {split}")
        return matches[0]

    lead = get_env("component_sum_dbar_plus_kbar", "heldout_any")
    rem = get_env("remainder_aware_component_plus_Q", "heldout_any")

    y = np.array([r["V_T"] for r in rows], dtype=float)
    component_bar = np.array([r["component_budget_over_T"] for r in rows], dtype=float)
    family = np.array([r["allocation_family"] for r in rows], dtype=object)
    gamma = np.array([r["gamma"] for r in rows], dtype=float)
    exo = np.array([r["C_exo_ratio"] for r in rows], dtype=float)
    masks = split_masks(rows)
    lead_bound = float(get_env("component_sum_dbar_plus_kbar", "calibration")["scale"]) * component_bar
    lead_violation = masks["heldout_any"] & (y > lead_bound * (1.0 + 1e-10))

    gamma_levels = sorted(float(v) for v in set(gamma))
    exo_levels = sorted(float(v) for v in set(exo))
    heat = np.zeros((len(gamma_levels), len(exo_levels)), dtype=int)
    for i, g in enumerate(gamma_levels):
        for j, c in enumerate(exo_levels):
            heat[i, j] = int(
                np.sum(
                    lead_violation
                    & (family == "bursty")
                    & np.isclose(gamma, g)
                    & np.isclose(exo, c)
                )
            )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.0, 1.25]})

    method_labels = ["Leading-order", "Remainder-aware"]
    q95_values = [float(lead["q95_ratio"]), float(rem["q95_ratio"])]
    max_values = [float(lead["max_ratio"]), float(rem["max_ratio"])]
    x = np.arange(len(method_labels))
    width = 0.32
    axes[0].bar(x - width / 2, q95_values, color="#1f77b4", width=width, label="95th percentile")
    axes[0].bar(x + width / 2, max_values, color="#ff7f0e", width=width, label="maximum")
    axes[0].axhline(1.0, color="0.25", linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_labels)
    axes[0].set_ylabel(r"$V_T/\widehat{U}_T$")
    axes[0].set_title("(a) Held-out union envelope ratios", loc="left", pad=6)
    axes[0].set_ylim(0.0, max(1.08, 1.08 * max(max_values)))
    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    axes[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        borderaxespad=0.0,
        frameon=False,
        ncol=2,
    )
    axes[0].text(
        0.0,
        -0.13,
        f"coverage {100.0 * float(lead['coverage']):.1f}\\%",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=11.0,
        clip_on=False,
    )
    axes[0].text(
        1.0,
        -0.13,
        f"coverage {100.0 * float(rem['coverage']):.1f}\\%",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=11.0,
        clip_on=False,
    )

    im = axes[1].imshow(
        heat,
        origin="lower",
        aspect="auto",
        cmap="RdPu",
        vmin=0,
        vmax=max(1, int(np.max(heat))),
    )
    axes[1].set_xticks(np.arange(len(exo_levels)))
    axes[1].set_xticklabels([f"{v:g}" for v in exo_levels])
    axes[1].set_yticks(np.arange(len(gamma_levels)))
    axes[1].set_yticklabels([f"{v:g}" for v in gamma_levels])
    axes[1].set_xlabel("exogenous budget ratio")
    axes[1].set_ylabel("feedback strength")
    axes[1].set_title("(b) Leading-order violations under bursty timing", loc="left", pad=6)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = int(heat[i, j])
            color = "white" if val >= 0.6 * max(1, int(np.max(heat))) else "black"
            axes[1].text(j, i, str(val), ha="center", va="center", fontsize=10.5, color=color)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.047, pad=0.025)
    cbar.set_label("violations")
    cbar.ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))

    fig.subplots_adjust(left=0.09, right=0.995, top=0.91, bottom=0.18, wspace=0.42)
    for stem in ["fig_nn_envelope_union_violation_localization"]:
        fig.savefig(run_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(run_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def analyze_run(run_dir: Path):
    raw_path = run_dir / "balanced_timing_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    rows = load_csv_dicts(raw_path)
    envelope_rows, locality = analyze_envelopes(rows)
    write_envelope_diagnostics(run_dir, envelope_rows, locality)
    plot_envelope_ratios(run_dir, envelope_rows)
    plot_union_and_violation_localization(run_dir, envelope_rows, rows)
    with open(run_dir / "balanced_timing_envelope_results.json", "w") as f:
        json.dump({"envelope_rows": envelope_rows, "locality": locality}, f, indent=2)
    print(f"Wrote analysis to: {run_dir}", flush=True)


def quick_config(cfg: TimingConfig) -> TimingConfig:
    return replace(
        cfg,
        simulation_seeds=tuple(range(3)),
        allocation_seeds=tuple(range(2)),
        T_grid=(200, 400),
        drift_ratios=(0.0, 0.005, 0.015),
        gamma_grid=(0.0, 0.0025, 0.005),
        allocation_families=("uniform", "late", "bursty"),
        hidden_dim=32,
        N_pop=256,
        eval_every=10,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Balanced timing-allocation NN experiment")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run the balanced experiment and analyze it")
    run.add_argument("--output-tag", default="nn_timing_balanced")
    run.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN)
    run.add_argument("--quick", action="store_true", help="Small smoke-test grid")

    analyze = sub.add_parser("analyze", help="Analyze an existing run directory")
    analyze.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cmd == "run":
        cfg = TimingConfig(hidden_dim=int(args.hidden_dim))
        if args.quick:
            cfg = quick_config(cfg)
        run_experiment(cfg, output_tag=args.output_tag)
    elif args.cmd == "analyze":
        analyze_run(Path(args.run_dir))


if __name__ == "__main__":
    main()
