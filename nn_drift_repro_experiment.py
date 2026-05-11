"""Teacher–learner drift environment with paper-aligned reproducibility metrics."""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.linalg as npl
import torch
import torch.nn as nn
import torch.nn.functional as F

from out_utils import create_results_dir

ALPHA_THEORY = 1.0  # speed-limit mixing constant (>0)
DEFAULT_HIDDEN = 128


@dataclass
class MultiTConfig:
    d_in: int = 5
    seeds: Sequence[int] = tuple(range(12))
    k_policy: float = 0.25
    sigma: float = 0.1
    C_pol: float = 8.0
    T_grid: Sequence[int] = (800, 1600, 3200, 6400)
    drift_ratios: Sequence[float] = (2.5e-3, 5e-3, 1e-2, 2e-2, 4e-2, 8e-2)
    gamma_grid = (0.0, 0.0025, 0.005, 0.01, 0.02)
    refresh_G_every: int = 25
    hidden_dim: int = DEFAULT_HIDDEN
    N_pop: int = 2048  
    eval_every = 10


def fit_linear_plane_safe(y, X):
    # Guard: finite values only
    if not (np.isfinite(y).all() and np.isfinite(X).all()):
        iy = np.where(~np.isfinite(y))[0]
        iX = np.argwhere(~np.isfinite(X))
        raise ValueError(f"Non-finite values: y idx={iy}, X idxs={iX}")
    try:
        coef, *_ = npl.lstsq(X, y, rcond=None)
    except Exception:
        coef = npl.pinv(X) @ y
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-20
    r2 = 1.0 - ss_res / ss_tot
    return coef, r2


torch.set_default_dtype(torch.float32)
DEVICE = torch.device("cpu")


try:
    from common_sim import fit_linear_plane as fit_linear_plane_np  # noqa: F401
except Exception:

    def fit_linear_plane_np(y, X):
        Xt = torch.from_numpy(X.astype(np.float64))
        yt = torch.from_numpy(y.astype(np.float64))
        beta, *_ = torch.linalg.lstsq(Xt, yt)
        yhat = Xt @ beta
        ss_res = torch.sum((yt - yhat) ** 2)
        ss_tot = torch.sum((yt - yt.mean()) ** 2)
        r2 = float((1.0 - ss_res / (ss_tot + 1e-20)).item())
        return beta.numpy().astype(np.float64), r2


# ----------------- Nonlinear features for target -----------------
def make_feature_map(d_in=5, m_feat=64, seed=123):
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=1.0 / np.sqrt(d_in), size=(m_feat, d_in)).astype(np.float32)
    b = rng.normal(scale=0.1, size=(m_feat,)).astype(np.float32)
    W = torch.from_numpy(W)
    b = torch.from_numpy(b)

    def phi(x):  # x: (N,d_in) -> (N,m_feat)
        return torch.tanh(x @ W.T + b)

    return phi, m_feat


# ----------------- Fisher Gram for theta -----------------
@torch.no_grad()
def fisher_gram(phi_fn, Xp, sigma=0.1, ridge=1e-6):
    Phi = phi_fn(Xp)  # (N, m)
    G = (Phi.T @ Phi) / (Xp.shape[0] * (sigma**2))
    m = G.shape[0]
    G = G + ridge * torch.eye(m, dtype=G.dtype)
    return G


@torch.no_grad()
def fisher_norm(G, delta_theta):
    # sqrt(delta^T G delta)
    v = delta_theta.view(-1, 1)
    val = (v.T @ (G @ v)).clamp_min(1e-18).sqrt().item()
    return float(val)


class FisherBudget:
    def __init__(self, total):
        self.remaining = float(max(0.0, total))

    def take(self, G, direction, proposed_len):
        """
        direction: torch vector in theta space (raw, any scale)
        proposed_len: desired Fisher-norm length for this step (float, >=0)
        Returns the actual step (torch vector) clipped to remaining Fisher budget.
        """
        if self.remaining <= 0.0 or proposed_len <= 0.0:
            return torch.zeros_like(direction), 0.0

        # Fisher length of 'direction'
        q = float((direction @ (G @ direction)).clamp_min(1e-18).sqrt().item())
        if q == 0.0:
            return torch.zeros_like(direction), 0.0

        # we want length = min(proposed_len, remaining)
        target = min(proposed_len, self.remaining)
        scale = target / q
        step = scale * direction
        self.remaining -= target
        return step, target


# ----------------- Learner MLP -----------------
class MLP(nn.Module):
    def __init__(self, d_in=5, h=128, d_out=1, nonlin="tanh"):
        super().__init__()
        self.l1 = nn.Linear(d_in, h)
        self.l2 = nn.Linear(h, d_out)
        self.nonlin = nonlin
        self.reset()

    def reset(self):
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.zeros_(self.l1.bias)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.zeros_(self.l2.bias)

    def forward(self, x):
        z = self.l1(x)
        z = torch.tanh(z) if self.nonlin == "tanh" else F.relu(z)
        return self.l2(z)


# ----------------- Data sampler -----------------
def make_sampler(d_in=5, seed=0):
    rng = np.random.default_rng(seed)

    def sample_x(N=1):
        return torch.from_numpy(rng.normal(size=(N, d_in)).astype(np.float32))

    return sample_x


# ----------------- Population risk helper -----------------
@torch.no_grad()
def pop_risk_mse(f, theta, phi_fn, sampler_pop, N=256, sigma=0.1):
    """
    Population risk R(theta,f) consistent with the empirical (noisy) squared loss:
      y = <phi(x), theta> + eps, eps~N(0,sigma^2)
      R(theta,f) = E[(f(x) - y)^2] = E[(f(x) - <phi,theta>)^2] + sigma^2
    We estimate it by sampling fresh X and using clean target + sigma^2.
    """
    X = sampler_pop(N=N).to(DEVICE)
    y_true = (phi_fn(X) @ theta).view(-1, 1)
    y_pred = f(X)
    return float(F.mse_loss(y_pred, y_true).item() + sigma**2)

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


# ----------------- One run -----------------
def one_run(
    T,
    d_in,
    phi_fn,
    m_feat,
    G_fisher,
    C_exo,
    gamma,
    k_policy,
    seed,
    lr_learner=0.1,
    sigma=0.1,
    C_pol=10.0,
    endo_mode="repel",
    refresh_G_every=0,
    s_end=1.0,
    hidden_dim: int = DEFAULT_HIDDEN,
    N_pop: int = 256,
    eval_every: int = 1,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if endo_mode not in ("repel", "attract"):
        raise ValueError("endo_mode should be 'repel' or 'attract'")

    if eval_every < 1:
        eval_every = 1

    theta = torch.zeros(m_feat, dtype=torch.float32)

    # -------------------------------- Learner -----------------------------------
    f = MLP(d_in=d_in, h=hidden_dim, d_out=1).to(DEVICE)
    opt = torch.optim.SGD(f.parameters(), lr=lr_learner)

    sample_x = make_sampler(d_in=d_in, seed=10_000 + seed)
    sample_x_pop = make_sampler(d_in=d_in, seed=40_000 + seed)

    # ------------------------------- Budgets ------------------------------------
    exo_budget = FisherBudget(C_exo)
    endo_budget_total = max(0.0, gamma * k_policy * C_pol * T)  # scaled with T
    endo_budget = FisherBudget(endo_budget_total)

    rng_exo = np.random.default_rng(seed + 7)

    mean_len_exo = (C_exo / T) if C_exo > 0 else 0.0
    mean_len_endo = (endo_budget_total / T) if endo_budget_total > 0 else 0.0

    sum_dt = 0.0
    sum_kappa_path = 0.0

    # Trajectories for metrics
    train_mse_traj = []         # all-step training losses (debug)
    train_mse_eval_traj = []
    pop_mse_traj = []           # R(theta_t, f_t) on eval steps
    pop_mse_plus_traj = []      # R(theta_{t+1}, f_t) on eval steps
    VT_terms = []               # |R_plus - R_t| on eval steps

    # ============================= MAIN SIMULATION ==============================
    for t in range(1, T + 1):
        # ---------- (1) Deployed predictor and empirical loss ----------
        x = sample_x(N=1).to(DEVICE)
        with torch.no_grad():
            y = (phi_fn(x) @ theta).view(-1, 1) + sigma * torch.randn(1, 1)

        opt.zero_grad()
        yhat = f(x)
        train_loss = F.mse_loss(yhat, y)
        train_mse_traj.append(float(train_loss.item()))

        # ---------- (2) Probe distribution ----------
        Xprobe = sample_x(N=256).to(DEVICE)
        Phi_probe = phi_fn(Xprobe)

        # Optional dynamic Fisher metric:
        # if refresh_G_every and t % refresh_G_every == 0:
        #     G_fisher = fisher_gram(phi_fn, Xprobe, sigma=sigma)

        # ---------- (3) Exogenous drift ----------
        dir_exo = torch.from_numpy(rng_exo.normal(size=(m_feat,)).astype(np.float32))
        prop_len_exo = float(rng_exo.exponential(mean_len_exo)) if mean_len_exo > 0 else 0.0
        step_exo, len_exo = exo_budget.take(G_fisher, dir_exo, prop_len_exo)

        # ---------- (4) Endogenous drift ----------
        step_endo = torch.zeros_like(theta)
        len_endo = 0.0

        if endo_budget.remaining > 0:
            with torch.no_grad():
                f_probe = f(Xprobe).view(-1)
                g_probe = (Phi_probe @ theta).view(-1)
                resid = (f_probe - g_probe)
                grad_theta = -2.0 * (Phi_probe.T @ resid) / Xprobe.shape[0]

            dir_endo = grad_theta / (grad_theta.norm() + 1e-12)

            kappa_raw = fisher_norm(G_fisher, dir_endo)
            if kappa_raw > 0.05:
                dir_endo *= 0.05 / (kappa_raw + 1e-12)

            if endo_mode == "attract":
                dir_endo = -dir_endo

            grad_norm = dir_endo.norm()
            if grad_norm > 0:
                dir_endo = dir_endo / grad_norm

            dir_endo = s_end * dir_endo

            kappa_dir = fisher_norm(G_fisher, dir_endo)
            kappa_cap = 0.1
            if kappa_dir > kappa_cap:
                dir_endo = dir_endo * (kappa_cap / (kappa_dir + 1e-12))

            prop_len_endo = mean_len_endo
            step_endo, len_endo = endo_budget.take(G_fisher, dir_endo, prop_len_endo)

        # ---------- (5) Candidate update ----------
        theta_next = theta + step_exo + step_endo

        # ---------- (6) EVALUATION (consistent schedule) ----------
        do_eval = (t % eval_every == 0) or (t == T)
        if do_eval:
            # training loss sampled on same schedule as pop eval
            train_mse_eval_traj.append(float(train_loss.item()))

            # population risks sampled on same schedule, using the deployed
            # pre-update f_t for both theta_t and theta_{t+1}
            R_t, R_plus = pop_risk_pair_mse(
                f, theta, theta_next, phi_fn, sample_x_pop, N=N_pop, sigma=sigma
            )
            pop_mse_traj.append(R_t)
            pop_mse_plus_traj.append(R_plus)
            VT_terms.append(abs(R_plus - R_t))

        # ---------- (7) Learner update f_t -> f_{t+1} ----------
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), max_norm=1.0)
        opt.step()

        # commit state update
        theta = theta_next
        sum_dt += float(len_exo)
        sum_kappa_path += float(len_endo)

    # ============================= FINAL EVAL ==============================
    Xeval = make_sampler(d_in=d_in, seed=20_000 + seed)(N=1024).to(DEVICE)
    with torch.no_grad():
        f_eval = f(Xeval).view(-1)
        g_eval = (phi_fn(Xeval) @ theta).view(-1)
        err_T = float(torch.mean(torch.abs(f_eval - g_eval)).item())

    # Paper-aligned aggregates (NOW consistent under eval_every>1)
    Rhat_T = float(np.mean(train_mse_eval_traj)) if train_mse_eval_traj else float("nan")
    R_T = float(np.mean(pop_mse_traj)) if pop_mse_traj else float("nan")
    Rplus_T = float(np.mean(pop_mse_plus_traj)) if pop_mse_plus_traj else float("nan")
    V_T = float(np.mean(VT_terms)) if VT_terms else float("nan")
    delta_sam = float(abs(Rhat_T - R_T))
    delta_rep = float(abs(Rhat_T - Rplus_T))

    # Legacy aggregates (kept for compatibility)
    traj_emp_risk = float(np.mean(train_mse_traj)) if train_mse_traj else float("nan")
    traj_pop_risk = R_T
    gen_gap_traj = float(abs(traj_emp_risk - traj_pop_risk))

    initial_pop_mse = float(pop_mse_traj[0]) if pop_mse_traj else float("nan")
    final_pop_mse = float(pop_mse_traj[-1]) if pop_mse_traj else float("nan")
    initial_pop_plus = float(pop_mse_plus_traj[0]) if pop_mse_plus_traj else float("nan")
    final_pop_plus = float(pop_mse_plus_traj[-1]) if pop_mse_plus_traj else float("nan")

    # Sanity checks
    exo_used = float(C_exo - exo_budget.remaining)
    endo_total = float(max(0.0, gamma * k_policy * C_pol * T))
    endo_used = float(endo_total - endo_budget.remaining)

    if not np.isfinite(sum_dt) or not np.isfinite(sum_kappa_path):
        raise RuntimeError(
            f"Non-finite paths: sum_dt={sum_dt}, sum_kappa_path={sum_kappa_path}, "
            f"exo_used={exo_used}, endo_used={endo_used}"
        )

    diff_exo = abs(exo_used - sum_dt)
    diff_endo = abs(endo_used - sum_kappa_path)
    if diff_exo > 1e-5 or diff_endo > 1e-5:
        print(
            f"[budget mismatch] exo_used={exo_used:.6f}, sum_dt={sum_dt:.6f}, "
            f"endo_used={endo_used:.6f}, sum_kappa_path={sum_kappa_path:.6f}, "
            f"diff_exo={diff_exo:.3e}, diff_endo={diff_endo:.3e}"
        )

    assert abs(exo_used - sum_dt) < 1e-5
    assert abs(endo_used - sum_kappa_path) < 1e-5

    return (
        err_T,
        float(sum_dt),
        float(sum_kappa_path),
        float(delta_rep),
        float(delta_sam),
        float(V_T),
        float(traj_emp_risk),
        float(traj_pop_risk),
        float(Rplus_T),
        initial_pop_mse,
        final_pop_mse,
        initial_pop_plus,
        final_pop_plus,
        float(gen_gap_traj),
    )

# ----------------- Main experiment -----------------
def main(hidden_dim: int = DEFAULT_HIDDEN):
    cfg = MultiTConfig(hidden_dim=int(hidden_dim))
    results_dir = create_results_dir("nn_multiT")

    phi_fn, m_feat = make_feature_map(d_in=cfg.d_in, m_feat=64, seed=123)
    rng_probe = np.random.default_rng(42)
    Xp = torch.from_numpy(rng_probe.normal(size=(512, cfg.d_in)).astype(np.float32))
    G_fisher = fisher_gram(phi_fn, Xp, sigma=cfg.sigma)

    rows, raw_rows = [], []

    for T in cfg.T_grid:
        for ratio in cfg.drift_ratios:
            C_exo_target = ratio * T
            for gamma in cfg.gamma_grid:
                err_all, sdt_all, skap_all = [], [], []
                delta_rep_all, delta_sam_all, VT_all = [], [], []
                traj_emp_all, traj_pop_all, traj_pop_plus_all = [], [], []
                pop_init_all, pop_final_all = [], []
                pop_plus_init_all, pop_plus_final_all = [], []
                legacy_gap_all = []

                for s in cfg.seeds:
                    C_exo_total = C_exo_target

                    (
                        err,
                        sdt,
                        skap,
                        delta_rep,
                        delta_sam,
                        V_T,
                        traj_emp,
                        traj_pop,
                        traj_pop_plus,
                        pop_init,
                        pop_final,
                        pop_plus_init,
                        pop_plus_final,
                        legacy_gap,
                    ) = one_run(
                        T=T,
                        d_in=cfg.d_in,
                        phi_fn=phi_fn,
                        m_feat=m_feat,
                        G_fisher=G_fisher,
                        C_exo=C_exo_total,
                        gamma=gamma,
                        k_policy=cfg.k_policy,
                        seed=s,
                        lr_learner=0.05,
                        sigma=cfg.sigma,
                        C_pol=cfg.C_pol,
                        refresh_G_every=cfg.refresh_G_every,
                        hidden_dim=cfg.hidden_dim,
                        N_pop=cfg.N_pop,
                        eval_every=cfg.eval_every,
                    )

                    if not all(np.isfinite(v) for v in [err, sdt, skap, delta_rep, delta_sam, V_T]):
                        raise RuntimeError(
                            f"Non-finite metrics: seed={s}, T={T}, C_exo={C_exo_total}, gamma={gamma} "
                            f"-> err={err}, sum_dt={sdt}, sum_kappa={skap}, delta_rep={delta_rep}, "
                            f"delta_sam={delta_sam}, V_T={V_T}"
                        )

                    err_all.append(err)
                    sdt_all.append(sdt)
                    skap_all.append(skap)
                    delta_rep_all.append(delta_rep)
                    delta_sam_all.append(delta_sam)
                    VT_all.append(V_T)
                    traj_emp_all.append(traj_emp)
                    traj_pop_all.append(traj_pop)
                    traj_pop_plus_all.append(traj_pop_plus)
                    pop_init_all.append(pop_init)
                    pop_final_all.append(pop_final)
                    pop_plus_init_all.append(pop_plus_init)
                    pop_plus_final_all.append(pop_plus_final)
                    legacy_gap_all.append(legacy_gap)

                    raw_rows.append(
                        {
                            "T": int(T),
                            "C_exo_total": float(C_exo_total),
                            "C_exo_ratio": float(ratio),
                            "gamma": float(gamma),
                            "seed": int(s),
                            "regime": "mixed",
                            "err_T": float(err),
                            "sum_dt": float(sdt),
                            "sum_kappa": float(skap),
                            # Paper-aligned:
                            "delta_rep": float(delta_rep),
                            "delta_sam": float(delta_sam),
                            "V_T": float(V_T),
                            "traj_Rhat": float(traj_emp),
                            "traj_R": float(traj_pop),
                            "traj_R_plus": float(traj_pop_plus),
                            # For sanity/time-series endpoints:
                            "pop_R_initial": float(pop_init),
                            "pop_R_final": float(pop_final),
                            "pop_Rplus_initial": float(pop_plus_init),
                            "pop_Rplus_final": float(pop_plus_final),
                            # Legacy proxy kept:
                            "gen_gap_traj_legacy": float(legacy_gap),
                        }
                    )

                rows.append(
                    {
                        "T": int(T),
                        "C_exo_ratio": float(ratio),
                        "gamma": float(gamma),
                        "err": float(np.mean(err_all)),
                        "sum_dt": float(np.mean(sdt_all)),
                        "sum_kappa": float(np.mean(skap_all)),
                        # Paper-aligned summaries:
                        "delta_rep": float(np.mean(delta_rep_all)),
                        "delta_sam": float(np.mean(delta_sam_all)),
                        "V_T": float(np.mean(VT_all)),
                        "traj_Rhat": float(np.mean(traj_emp_all)),
                        "traj_R": float(np.mean(traj_pop_all)),
                        "traj_R_plus": float(np.mean(traj_pop_plus_all)),
                        # Endpoints:
                        "traj_pop_initial": float(np.mean(pop_init_all)),
                        "traj_pop_final": float(np.mean(pop_final_all)),
                        "traj_pop_plus_initial": float(np.mean(pop_plus_init_all)),
                        "traj_pop_plus_final": float(np.mean(pop_plus_final_all)),
                        # Legacy:
                        "gen_gap_legacy": float(np.mean(legacy_gap_all)),
                    }
                )

    # ---------------- Plane fits / collapses ----------------
    Tv = np.array([r["T"] for r in rows], dtype=float)
    inv_sqrt_T = Tv ** (-0.5)
    dt_over_T = np.array([r["sum_dt"] / r["T"] for r in rows], dtype=float)
    sum_kappa_over_T = np.array([r["sum_kappa"] / r["T"] for r in rows], dtype=float)

    y_err = np.array([r["err"] for r in rows], dtype=float)

    #  changed "risk gap" target is now paper-aligned Delta_rep (not the legacy proxy)
    y_gap = np.array([r["delta_rep"] for r in rows], dtype=float)

    # ------- HOLD-OUT SPLIT for alpha* and 1-D collapse -------
    unique_T = np.unique(Tv)
    T_hold = unique_T[-1]  # largest horizon
    fit_mask = Tv != T_hold
    eval_mask = ~fit_mask
    holdout_info = {
        "T": int(T_hold),
        "strategy": "hold_largest_T",
        "n_fit": int(np.sum(fit_mask)),
        "n_eval": int(np.sum(eval_mask)),
    }

    # --- Risk-gap plane fit on FIT ONLY (for alpha estimation) ---
    X_gap_fit = np.c_[np.ones(np.sum(fit_mask)), inv_sqrt_T[fit_mask], dt_over_T[fit_mask], sum_kappa_over_T[fit_mask]]
    coef_gap_fit, r2_gap_fit = fit_linear_plane_safe(y_gap[fit_mask], X_gap_fit)
    b0g_fit, bsg_fit, b1g_fit, b2g_fit = map(float, coef_gap_fit)
    alpha_gap_fit = float(ALPHA_THEORY)
    if abs(b1g_fit) > 1e-12:
        alpha_gap_fit = float(b2g_fit / b1g_fit)

    # Full plane for ERR and GAP
    X = np.c_[np.ones_like(Tv), inv_sqrt_T, dt_over_T, sum_kappa_over_T]

    coef_err, r2_err = fit_linear_plane_safe(y_err, X)
    b0, b_s, b1, b2 = map(float, coef_err)

    coef_gap, r2_gap_plane = fit_linear_plane_safe(y_gap, X)
    b0g, bsg, b1g, b2g = map(float, coef_gap)

    # Collapse for ERR: alpha_opt from err-plane
    alpha_opt = (b2 / b1) if abs(b1) > 1e-12 else 1.0
    C_over_T_err = dt_over_T + alpha_opt * sum_kappa_over_T
    X1_err = np.c_[np.ones_like(C_over_T_err), C_over_T_err]
    coef1_err, r2_line_err = fit_linear_plane_safe(y_err, X1_err)
    a0_err, a1_err = map(float, coef1_err)

    # Collapse for GAP: alpha_theory from FIT-only (paper narrative knob)
    alpha_theory = alpha_gap_fit
    C_over_T_theory = dt_over_T + alpha_theory * sum_kappa_over_T
    X1_gap = np.c_[np.ones_like(C_over_T_theory), C_over_T_theory]
    coef1_gap, r2_line_gap = fit_linear_plane_safe(y_gap, X1_gap)
    a0_gap, a1_gap = map(float, coef1_gap)

    # Speed-limit fit: y_gap ≈ c0 + c1*T^{-1/2} + c2*(C/T)
    X_full = np.c_[np.ones_like(y_gap), inv_sqrt_T, C_over_T_theory]
    coef_full, r2_full = fit_linear_plane_safe(y_gap, X_full)
    c0, c1, c2 = map(float, coef_full)

    # Diagnostics ratio
    denom = inv_sqrt_T + C_over_T_theory + 1e-12
    rho = y_gap / denom
    rho_stats = {
        "mean": float(np.mean(rho)),
        "std": float(np.std(rho, ddof=1)),
        "min": float(np.min(rho)),
        "max": float(np.max(rho)),
    }

    # ---------------- Exports ----------------
    plane_path = results_dir / "figNN_plane_fit.txt"
    raw_csv_path = results_dir / "figNN_additivity_raw.csv"
    summary_csv_path = results_dir / "figNN_additivity_summary.csv"
    meta_path = results_dir / "figNN_additivity_meta.json"

    with open(plane_path, "w") as f:
        f.write("NN Additivity plane: err ~ b0 + b_s*T^{-1/2} + b1*(sum_dt/T) + b2*(sum_kappa/T)\n")
        f.write(f"b0   = {b0:.6f}\n")
        f.write(f"b_s  = {b_s:.6f}\n")
        f.write(f"b1   = {b1:.6f}\n")
        f.write(f"b2   = {b2:.6f}\n")
        f.write(f"R^2  = {r2_err:.4f}\n")

        f.write("\n--- Paper-aligned risk gap (Delta_rep) plane ---\n")
        f.write("Delta_rep ~ b0g + bsg*T^{-1/2} + b1g*(sum_dt/T) + b2g*(sum_kappa/T)\n")
        f.write(f"b0g = {b0g:.6f}, bsg = {bsg:.6f}, b1g = {b1g:.6f}, b2g = {b2g:.6f}, R^2 = {r2_gap_plane:.4f}\n")
        f.write(f"alpha_theory (fit-only) = {alpha_theory:.6f}\n")
        f.write(f"Collapse (Delta_rep): y ≈ {a0_gap:.6f} + {a1_gap:.6f} * (C_T/T), R^2 = {r2_line_gap:.4f}\n")

        f.write("\n=== Speed-limit fit: Delta_rep ≈ c0 + c1*T^{-1/2} + c2*(C_T/T) ===\n")
        f.write(f"c0 = {c0:.6f}, c1 = {c1:.6f}, c2 = {c2:.6f}, R^2 = {r2_full:.4f}\n")
        f.write("--- Ratio ρ_T = y / (T^{-1/2}+C_T/T) ---\n")
        f.write(
            f"mean={rho_stats['mean']:.4f}, std={rho_stats['std']:.4f}, "
            f"min={rho_stats['min']:.4f}, max={rho_stats['max']:.4f}\n"
        )

        f.write("\n--- Legacy note ---\n")
        f.write("The CSV also includes gen_gap_traj_legacy = |mean(train_loss) - mean(R(theta_t,f_t))| for backward compatibility.\n")

    # raw csv
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "T",
                "C_exo_total",
                "C_exo_ratio",
                "gamma",
                "seed",
                "regime",
                "err_T",
                "sum_dt",
                "sum_kappa",
                # Paper-aligned:
                "delta_rep",
                "delta_sam",
                "V_T",
                "traj_Rhat",
                "traj_R",
                "traj_R_plus",
                "pop_R_initial",
                "pop_R_final",
                "pop_Rplus_initial",
                "pop_Rplus_final",
                # Legacy:
                "gen_gap_traj_legacy",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    # summaries
    def mean_se(vals):
        n = len(vals)
        m = float(np.mean(vals)) if n else float("nan")
        se = float(np.std(vals, ddof=1) / np.sqrt(max(n, 1))) if n > 1 else float("nan")
        return m, se, n

    group = defaultdict(
        lambda: {
            "err_T": [],
            "sum_dt": [],
            "sum_kappa": [],
            "delta_rep": [],
            "delta_sam": [],
            "V_T": [],
            "traj_Rhat": [],
            "traj_R": [],
            "traj_R_plus": [],
            "pop_R_initial": [],
            "pop_R_final": [],
            "pop_Rplus_initial": [],
            "pop_Rplus_final": [],
            "gen_gap_traj_legacy": [],
        }
    )

    for r in raw_rows:
        key = (r["T"], r["C_exo_ratio"], r["gamma"])
        g = group[key]
        for k in g:
            g[k].append(r[k] if k in r else float("nan"))

    summary_rows = []
    for (Tval, ratio, gamma), g in sorted(group.items()):
        m_err, se_err, n = mean_se(g["err_T"])
        m_dt, se_dt, _ = mean_se(g["sum_dt"])
        m_kap, se_kap, _ = mean_se(g["sum_kappa"])
        m_delta, se_delta, _ = mean_se(g["delta_rep"])
        m_sam, se_sam, _ = mean_se(g["delta_sam"])
        m_VT, se_VT, _ = mean_se(g["V_T"])
        m_Rhat, se_Rhat, _ = mean_se(g["traj_Rhat"])
        m_R, se_R, _ = mean_se(g["traj_R"])
        m_Rp, se_Rp, _ = mean_se(g["traj_R_plus"])
        m_R0, se_R0, _ = mean_se(g["pop_R_initial"])
        m_RF, se_RF, _ = mean_se(g["pop_R_final"])
        m_Rp0, se_Rp0, _ = mean_se(g["pop_Rplus_initial"])
        m_RpF, se_RpF, _ = mean_se(g["pop_Rplus_final"])
        m_leg, se_leg, _ = mean_se(g["gen_gap_traj_legacy"])

        summary_rows.append(
            {
                "T": int(Tval),
                "C_exo_ratio": float(ratio),
                "gamma": float(gamma),
                "regime": "mixed",
                "n": int(n),
                "mean_err_T": m_err,
                "se_err_T": se_err,
                "mean_sum_dt": m_dt,
                "se_sum_dt": se_dt,
                "mean_sum_kappa": m_kap,
                "se_sum_kappa": se_kap,
                # Paper-aligned:
                "mean_delta_rep": m_delta,
                "se_delta_rep": se_delta,
                "mean_delta_sam": m_sam,
                "se_delta_sam": se_sam,
                "mean_V_T": m_VT,
                "se_V_T": se_VT,
                "mean_traj_Rhat": m_Rhat,
                "se_traj_Rhat": se_Rhat,
                "mean_traj_R": m_R,
                "se_traj_R": se_R,
                "mean_traj_R_plus": m_Rp,
                "se_traj_R_plus": se_Rp,
                "mean_pop_R_initial": m_R0,
                "se_pop_R_initial": se_R0,
                "mean_pop_R_final": m_RF,
                "se_pop_R_final": se_RF,
                "mean_pop_Rplus_initial": m_Rp0,
                "se_pop_Rplus_initial": se_Rp0,
                "mean_pop_Rplus_final": m_RpF,
                "se_pop_Rplus_final": se_RpF,
                # Legacy:
                "mean_gen_gap_traj_legacy": m_leg,
                "se_gen_gap_traj_legacy": se_leg,
                # Convenience:
                "mean_C_over_T(alpha_theory)": float((m_dt + alpha_theory * m_kap) / Tval),
                "mean_C_over_T(alpha_opt_err)": float((m_dt + alpha_opt * m_kap) / Tval),
            }
        )

    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "T",
                "C_exo_ratio",
                "gamma",
                "regime",
                "n",
                "mean_err_T",
                "se_err_T",
                "mean_sum_dt",
                "se_sum_dt",
                "mean_sum_kappa",
                "se_sum_kappa",
                "mean_delta_rep",
                "se_delta_rep",
                "mean_delta_sam",
                "se_delta_sam",
                "mean_V_T",
                "se_V_T",
                "mean_traj_Rhat",
                "se_traj_Rhat",
                "mean_traj_R",
                "se_traj_R",
                "mean_traj_R_plus",
                "se_traj_R_plus",
                "mean_pop_R_initial",
                "se_pop_R_initial",
                "mean_pop_R_final",
                "se_pop_R_final",
                "mean_pop_Rplus_initial",
                "se_pop_Rplus_initial",
                "mean_pop_Rplus_final",
                "se_pop_Rplus_final",
                "mean_gen_gap_traj_legacy",
                "se_gen_gap_traj_legacy",
                "mean_C_over_T(alpha_theory)",
                "mean_C_over_T(alpha_opt_err)",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    meta = {
        "figure": "NN Additivity & budget scaling (paper-aligned Delta_rep)",
        "d_in": cfg.d_in,
        "m_feat": m_feat,
        "nn_architecture": {
            "input_dim": int(cfg.d_in),
            "hidden_dim": int(cfg.hidden_dim),
            "output_dim": 1,
        },
        "T_grid": list(map(int, cfg.T_grid)),
        "seeds": list(map(int, cfg.seeds)),
        "k_policy": float(cfg.k_policy),
        "sigma": float(cfg.sigma),
        "C_exo_ratios": list(map(float, cfg.drift_ratios)),
        "gamma_grid": list(map(float, cfg.gamma_grid)),
        "plane_fit_err": {"b0": b0, "b_s": b_s, "b1": b1, "b2": b2, "R2": r2_err},
        "plane_fit_delta_rep": {"b0": b0g, "b_s": bsg, "b1": b1g, "b2": b2g, "R2": r2_gap_plane},
        "alpha_opt_err": float(alpha_opt),
        "alpha_theory_fit": float(alpha_theory),
        "collapse_err": {"a0": float(a0_err), "a1": float(a1_err), "R2": float(r2_line_err)},
        "collapse_delta_rep": {"a0": float(a0_gap), "a1": float(a1_gap), "R2": float(r2_line_gap)},
        "speed_limit_full": {"c0": c0, "c1": c1, "c2": c2, "R2": r2_full},
        "rho_stats": rho_stats,
        "holdout": holdout_info,
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balanced NN drift experiment (paper-aligned metrics)")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN,
        help="Hidden layer width for the learner MLP (default: 128).",
    )
    args = parser.parse_args()
    main(hidden_dim=args.hidden_dim)
