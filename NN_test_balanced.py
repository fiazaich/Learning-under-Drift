import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy.linalg as npl
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
    gamma_grid: Sequence[float] = (0.0, 0.01)
    refresh_G_every: int = 25
    hidden_dim: int = DEFAULT_HIDDEN

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
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - y.mean())**2)) + 1e-20
    r2 = 1.0 - ss_res/ss_tot
    return coef, r2

@torch.no_grad()
def estimate_generalization_gap(f, theta, phi_fn, sampler, n=512, sigma=0.1):
    """
    Returns |R_hat - R| at the terminal time T:
      R     := E[(f(X) - g_theta(X))^2] under clean labels (no noise)
      R_hat := E[(f(X) - Y)^2]   with Y = g_theta(X) + noise, noise~N(0, sigma^2)
    """
    X = sampler(N=n)
    y_true = (phi_fn(X) @ theta).view(-1,1)
    y_noisy = y_true + sigma * torch.randn_like(y_true)

    loss_true = F.mse_loss(f(X), y_true).item()
    loss_emp  = F.mse_loss(f(X), y_noisy).item()
    return abs(loss_emp - loss_true)


torch.set_default_dtype(torch.float32)
DEVICE = torch.device('cpu')

# ---- Optional import of your helper; else fallback ----
try:
    from common_sim import fit_linear_plane as fit_linear_plane_np
except Exception:
    def fit_linear_plane_np(y, X):
        Xt = torch.from_numpy(X.astype(np.float64))
        yt = torch.from_numpy(y.astype(np.float64))
        beta, *_ = torch.linalg.lstsq(Xt, yt)
        yhat = (Xt @ beta)
        ss_res = torch.sum((yt - yhat)**2)
        ss_tot = torch.sum((yt - yt.mean())**2)
        r2 = float((1.0 - ss_res/(ss_tot + 1e-20)).item())
        return beta.numpy().astype(np.float64), r2

# ----------------- Nonlinear features for target -----------------
def make_feature_map(d_in=5, m_feat=64, seed=123):
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=1.0/np.sqrt(d_in), size=(m_feat, d_in)).astype(np.float32)
    b = rng.normal(scale=0.1, size=(m_feat,)).astype(np.float32)
    W = torch.from_numpy(W)
    b = torch.from_numpy(b)
    def phi(x):  # x: (N,d_in) -> (N,m_feat)
        return torch.tanh(x @ W.T + b)
    return phi, m_feat

# ----------------- Fisher Gram for theta -----------------
@torch.no_grad()
def fisher_gram(phi_fn, Xp, sigma=0.1, ridge=1e-6):
    Phi = phi_fn(Xp)          # (N, m)
    G = (Phi.T @ Phi) / (Xp.shape[0] * (sigma**2))
    m = G.shape[0]
    G = G + ridge * torch.eye(m, dtype=G.dtype)
    return G


@torch.no_grad()
def fisher_norm(G, delta_theta):
    # sqrt(delta^T G delta)
    v = delta_theta.view(-1,1)
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
    def __init__(self, d_in=5, h=128, d_out=1, nonlin='tanh'):
        super().__init__()
        self.l1 = nn.Linear(d_in, h)
        self.l2 = nn.Linear(h, d_out)
        self.nonlin = nonlin
        self.reset()

    def reset(self):
        nn.init.xavier_uniform_(self.l1.weight); nn.init.zeros_(self.l1.bias)
        nn.init.xavier_uniform_(self.l2.weight); nn.init.zeros_(self.l2.bias)

    def forward(self, x):
        z = self.l1(x)
        z = torch.tanh(z) if self.nonlin == 'tanh' else F.relu(z)
        return self.l2(z)

# ----------------- Data sampler -----------------
def make_sampler(d_in=5, seed=0):
    rng = np.random.default_rng(seed)
    def sample_x(N=1):
        return torch.from_numpy(rng.normal(size=(N,d_in)).astype(np.float32))
    return sample_x

# ----------------- One run -----------------
def one_run(
    T, d_in, phi_fn, m_feat, G_fisher, C_exo, gamma, k_policy,
    seed, lr_learner=0.1, sigma=0.1, C_pol=10.0, endo_mode="repel",
    refresh_G_every=0, s_end=1.0, hidden_dim: int = DEFAULT_HIDDEN,
):

    torch.manual_seed(seed); np.random.seed(seed)
    if endo_mode not in ("repel", "attract"):
        raise ValueError("endo_mode should be 'repel' or 'attract'")

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
    rng_endo = np.random.default_rng(seed + 17)

    mean_len_exo  = (C_exo / T) if C_exo > 0 else 0.0
    mean_len_endo = (endo_budget_total / T) if endo_budget_total > 0 else 0.0

    sum_dt = 0.0
    sum_kappa_path = 0.0
    train_mse_traj = []
    pop_mse_traj = []

    # ============================= MAIN SIMULATION ==============================
    for t in range(1, T+1):
        # ---------- (1) SGD STEP ----------
        x = sample_x(N=1).to(DEVICE)
        with torch.no_grad():
            y = (phi_fn(x) @ theta).view(-1,1) + sigma * torch.randn(1,1)

        opt.zero_grad()
        yhat = f(x)
        train_loss = F.mse_loss(yhat, y)
        train_mse_traj.append(float(train_loss.item()))
        with torch.no_grad():
            X_pop = sample_x_pop(N=256).to(DEVICE)
            y_true_pop = (phi_fn(X_pop) @ theta).view(-1,1)
            y_pred_pop = f(X_pop)
            pop_loss = F.mse_loss(y_pred_pop, y_true_pop).item() + sigma**2
            pop_mse_traj.append(float(pop_loss))
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(f.parameters(), max_norm=1.0)
        opt.step()

        # ---------- (2) Probe distribution ----------
        Xprobe = sample_x(N=256).to(DEVICE)
        Phi_probe = phi_fn(Xprobe)

        # Uncomment if you want dynamic Fisher metric:
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
                # ∇_θ E[(f-g)^2]
                grad_theta = -2.0 * (Phi_probe.T @ resid) / Xprobe.shape[0]

            # base direction = disagreement gradient
            dir_endo = grad_theta
            dir_endo = dir_endo / (dir_endo.norm() + 1e-12)

            kappa_raw = fisher_norm(G_fisher, dir_endo)
            if kappa_raw > 0.05:
                dir_endo *= 0.05 / kappa_raw


            # attract vs repel
            if endo_mode == "attract":
                dir_endo = -dir_endo

            # Normalize the direction in Euclidean norm so the budget controls the step length.
            grad_norm = dir_endo.norm()
            if grad_norm > 0:
                dir_endo = dir_endo / grad_norm

            # optional: extra scaling knob if you want weaker/stronger endo push
            dir_endo = s_end * dir_endo

            # Soft cap on the Fisher norm of the direction vector.
            kappa_dir = fisher_norm(G_fisher, dir_endo)
            kappa_cap = 0.1   # try 0.05–0.2; this is “how curved” a direction we tolerate
            if kappa_dir > kappa_cap:
                dir_endo = dir_endo * (kappa_cap / (kappa_dir + 1e-12))

            # proposed Fisher length for this step (random quota)
            # prop_len_endo = (
            #     float(rng_endo.exponential(mean_len_endo))
            #     if mean_len_endo > 0 else 0.0
            # )
            prop_len_endo = mean_len_endo    # deterministic per-step κ


            # FisherBudget.take will now:
            #   - set the Fisher norm of the step to min(prop_len_endo, remaining),
            #   - decrement the endo budget accordingly.
            step_endo, len_endo = endo_budget.take(G_fisher, dir_endo, prop_len_endo)

        # ---------- (5) Update ----------
        theta = theta + step_exo + step_endo
        sum_dt += float(len_exo)
        sum_kappa_path += float(len_endo)


    # ============================= FINAL EVAL ==============================
    Xeval = make_sampler(d_in=d_in, seed=20_000+seed)(N=1024).to(DEVICE)
    with torch.no_grad():
        f_eval = f(Xeval).view(-1)
        g_eval = (phi_fn(Xeval) @ theta).view(-1)
        err_T = float(torch.mean(torch.abs(f_eval - g_eval)).item())

    traj_emp_risk = float(np.mean(train_mse_traj))
    traj_pop_risk = float(np.mean(pop_mse_traj))
    initial_pop_mse = float(pop_mse_traj[0]) if pop_mse_traj else float("nan")
    final_pop_mse = float(pop_mse_traj[-1]) if pop_mse_traj else float("nan")
    gen_gap_traj = float(abs(traj_emp_risk - traj_pop_risk))



    # Sanity checks
    exo_used = float(C_exo - exo_budget.remaining)
    endo_total = float(max(0.0, gamma * k_policy * C_pol * T))
    endo_used = float(endo_total - endo_budget.remaining)

    if not np.isfinite(sum_dt) or not np.isfinite(sum_kappa_path):
        raise RuntimeError(
            f"Non-finite paths: sum_dt={sum_dt}, sum_kappa_path={sum_kappa_path}, "
            f"exo_used={exo_used}, endo_used={endo_used}"
        )

    # Soft check instead of hard assert, so we can see discrepancies
    diff_exo  = abs(exo_used - sum_dt)
    diff_endo = abs(endo_used - sum_kappa_path)

    if diff_exo > 1e-5 or diff_endo > 1e-5:
        print(
            f"[budget mismatch] exo_used={exo_used:.6f}, sum_dt={sum_dt:.6f}, "
            f"endo_used={endo_used:.6f}, sum_kappa_path={sum_kappa_path:.6f}, "
            f"diff_exo={diff_exo:.3e}, diff_endo={diff_endo:.3e}"
        )
        # If you *really* want to keep failing here, uncomment:
        # raise AssertionError("Budget accounting mismatch")

    # (otherwise just continue)

    assert abs(exo_used - sum_dt) < 1e-5
    assert abs(endo_used - sum_kappa_path) < 1e-5

    return (
        err_T,
        float(sum_dt),
        float(sum_kappa_path),
        float(gen_gap_traj),
        float(traj_emp_risk),
        float(traj_pop_risk),
        initial_pop_mse,
        final_pop_mse,
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
                err_all, sdt_all, skap_all, gap_all = [], [], [], []
                traj_emp_all, traj_pop_all = [], []
                pop_init_all, pop_final_all = [], []
                for s in cfg.seeds:
                    rng_jitter = np.random.default_rng(777 + s)
                    j_exo = float(rng_jitter.uniform(0.9, 1.1))
                    j_endo = float(rng_jitter.uniform(0.9, 1.1))
                    C_exo_total = C_exo_target * j_exo
                    err, sdt, skap, gap, traj_emp, traj_pop, pop_init, pop_final = one_run(
                        T=T, d_in=cfg.d_in, phi_fn=phi_fn, m_feat=m_feat, G_fisher=G_fisher,
                        C_exo=C_exo_total, gamma=gamma, k_policy=cfg.k_policy,
                        seed=s, lr_learner=0.05, sigma=cfg.sigma, C_pol=cfg.C_pol * j_endo,
                        refresh_G_every=cfg.refresh_G_every, hidden_dim=cfg.hidden_dim
                    )
                    if not (np.isfinite(err) and np.isfinite(sdt) and np.isfinite(skap) and np.isfinite(gap)):
                        raise RuntimeError(
                            f"Non-finite metrics: seed={s}, T={T}, C_exo={C_exo_total}, gamma={gamma} "
                            f"-> err={err}, sum_dt={sdt}, sum_kappa={skap}, gen_gap={gap}"
                        )

                    err_all.append(err); sdt_all.append(sdt); skap_all.append(skap); gap_all.append(gap)
                    traj_emp_all.append(traj_emp); traj_pop_all.append(traj_pop)
                    pop_init_all.append(pop_init); pop_final_all.append(pop_final)
                    raw_rows.append({
                        "T": int(T),
                        "C_exo_total": float(C_exo_total),
                        "C_exo_ratio": float(ratio),
                        "gamma": float(gamma),
                        "seed": int(s),
                        "regime": "mixed",
                        "err_T": float(err),
                        "sum_dt": float(sdt),
                        "sum_kappa": float(skap),
                        "gen_gap_traj": float(gap),
                        "traj_emp_risk": float(traj_emp),
                        "traj_pop_risk": float(traj_pop),
                        "pop_mse_initial": float(pop_init),
                        "pop_mse_final": float(pop_final),
                    })

                rows.append({
                    "T": int(T),
                    "C_exo_ratio": float(ratio),
                    "gamma": float(gamma),
                    "err": float(np.mean(err_all)),
                    "sum_dt": float(np.mean(sdt_all)),
                    "sum_kappa": float(np.mean(skap_all)),
                    "gen_gap": float(np.mean(gap_all)),
                    "traj_emp_risk": float(np.mean(traj_emp_all)),
                    "traj_pop_risk": float(np.mean(traj_pop_all)),
                    "traj_pop_initial": float(np.mean(pop_init_all)),
                    "traj_pop_final": float(np.mean(pop_final_all)),
                })


    # -------- plane fit with T^{-1/2} restored --------
    y  = np.array([r["err"] for r in rows], dtype=float)
    Tv = np.array([r["T"]   for r in rows], dtype=float)

    # ------- HOLD-OUT SPLIT for alpha* and 1-D collapse -------
    # Build arrays we need
    y_err  = np.array([r["err"]     for r in rows], dtype=float)
    y_gap  = np.array([r["gen_gap"] for r in rows], dtype=float)
    dt_over_T   = np.array([r["sum_dt"]/r["T"]    for r in rows], dtype=float)
    sum_kappa_over_T  = np.array([r["sum_kappa"]/r["T"] for r in rows], dtype=float)
    inv_sqrt_T  = Tv ** (-0.5)

    # Hold out the largest T to check generalization at the longest horizon.
    unique_T = np.unique(Tv)
    T_hold = unique_T[-1]         # largest horizon
    fit_mask = (Tv != T_hold)
    eval_mask = ~fit_mask
    holdout_info = {
        "T": int(T_hold),
        "gamma": None,
        "strategy": "hold_largest_T",
        "n_fit": int(np.sum(fit_mask)),
        "n_eval": int(np.sum(eval_mask)),
    }


    # --- Risk-gap plane fit on FIT ONLY (for alpha estimation) ---
    X_gap_fit = np.c_[np.ones(np.sum(fit_mask)),
                      inv_sqrt_T[fit_mask], dt_over_T[fit_mask], sum_kappa_over_T[fit_mask]]
    coef_gap_fit, r2_gap_fit = fit_linear_plane_safe(y_gap[fit_mask], X_gap_fit)
    b0g_fit, bsg_fit, b1g_fit, b2g_fit = map(float, coef_gap_fit)
    alpha_gap_fit = float(ALPHA_THEORY)
    if abs(b1g_fit) > 1e-12:
        alpha_gap_fit = max(0.0, float(b2g_fit / b1g_fit))

   
    # Design matrix: [1, T^{-1/2}, sum_dt/T, sum_kappa/T]
    X = np.c_[np.ones_like(y), inv_sqrt_T, dt_over_T, sum_kappa_over_T]
    # ---------- Absolute-budget collapse for the *risk gap* (theory upper envelope) ----------
    # plane fit for the risk gap
    coef_gap, r2_gap_plane = fit_linear_plane_safe(y_gap, X)
    b0g, bsg, b1g, b2g = map(float, coef_gap)

    # absolute mixing: treat both components as nonnegative costs (Cor. 14)
    alpha_abs = abs(alpha_gap_fit) if alpha_gap_fit > 0 else 1.0

    alpha_theory = alpha_gap_fit
    C_over_T_theory = dt_over_T + alpha_theory * sum_kappa_over_T

      # Constant-factor tightness ratio
    denom = inv_sqrt_T + C_over_T_theory + 1e-12
    rho = y_gap / denom
    rho_stats = {
        "mean": float(np.mean(rho)),
        "std":  float(np.std(rho, ddof=1)),
        "min":  float(np.min(rho)),
        "max":  float(np.max(rho))
    }

    # --- Plane fit on FIT ONLY (for ERR) ---
    X_fit = np.c_[np.ones(np.sum(fit_mask)),
                  inv_sqrt_T[fit_mask], dt_over_T[fit_mask], sum_kappa_over_T[fit_mask]]
    coef_fit, r2_fit = fit_linear_plane_safe(y_err[fit_mask], X_fit)
    b0_f, b_s_f, b1_f, b2_f = map(float, coef_fit)
    alpha_opt_fit = (b2_f / b1_f) if abs(b1_f) > 1e-12 else 1.0

    # --- Evaluate 1-D collapse on EVAL ONLY ---
    C_over_T_eval_err = dt_over_T[eval_mask] + alpha_opt_fit * sum_kappa_over_T[eval_mask]
    X1_eval = np.c_[np.ones_like(C_over_T_eval_err), C_over_T_eval_err]
    coef1_eval, r2_line_eval = fit_linear_plane_safe(y_err[eval_mask], X1_eval)
    a0_eval, a1_eval = map(float, coef1_eval)

    C_over_T_eval_gap = dt_over_T[eval_mask] + alpha_theory * sum_kappa_over_T[eval_mask]
    X1_eval_gap = np.c_[np.ones_like(C_over_T_eval_gap), C_over_T_eval_gap]
    coef1_eval_gap, r2_line_eval_gap = fit_linear_plane_safe(y_gap[eval_mask], X1_eval_gap)
    a0_eval_gap, a1_eval_gap = map(float, coef1_eval_gap)

    # === Speed-limit fit on the TRUE risk gap (calibrated alpha) ===
    X_full = np.c_[np.ones_like(y_gap), inv_sqrt_T, C_over_T_theory]
    coef_full, r2_full = fit_linear_plane_safe(y_gap, X_full)
    c0, c1, c2 = map(float, coef_full)

    # Reduced A: drop T^{-1/2}
    X_noT = np.c_[np.ones_like(y_gap), C_over_T_theory]
    coef_noT, r2_noT = fit_linear_plane_safe(y_gap, X_noT)
    a0_noT, a1_noT = map(float, coef_noT)

    # Reduced B: drop C_T/T
    X_noC = np.c_[np.ones_like(y_gap), inv_sqrt_T]
    coef_noC, r2_noC = fit_linear_plane_safe(y_gap, X_noC)
    a0_noC, a1_noC = map(float, coef_noC)

    # Nested-model RSS + F tests
    def rss(yv, Xmat, beta):
        yhat = Xmat @ beta
        return float(np.sum((yv - yhat)**2))

    n = y_gap.shape[0]
    p_full, p_noT, p_noC = X_full.shape[1], X_noT.shape[1], X_noC.shape[1]
    RSS_full = rss(y_gap, X_full, coef_full)
    RSS_noT  = rss(y_gap, X_noT,  coef_noT)
    RSS_noC  = rss(y_gap, X_noC,  coef_noC)

    F_noT = ((RSS_noT - RSS_full)/(p_full - p_noT)) / (RSS_full/(n - p_full))
    F_noC = ((RSS_noC - RSS_full)/(p_full - p_noC)) / (RSS_full/(n - p_full))

    delta_R2_noT = r2_full - r2_noT
    delta_R2_noC = r2_full - r2_noC

    # Partial-residual (remove T^{-1/2}), then collapse on C_T/T
    y_gap_res = y_gap - (c0 + c1 * inv_sqrt_T)
    X1_res = np.c_[np.ones_like(C_over_T_theory), C_over_T_theory]
    coef_res, r2_res = fit_linear_plane_safe(y_gap_res, X1_res)
    a0_res, a1_res = map(float, coef_res)

    # Per-horizon slopes after removing T^{-1/2}
    byT = defaultdict(lambda: {"x": [], "y": []})
    for xval, yval, Tval in zip(C_over_T_theory, y_gap_res, Tv):
        byT[int(Tval)]["x"].append(float(xval))
        byT[int(Tval)]["y"].append(float(yval))
    perT = {}
    for Tval, data in byT.items():
        xv = np.array(data["x"], dtype=float)
        yv = np.array(data["y"], dtype=float)
        X_t = np.c_[np.ones_like(xv), xv]
        beta_t, r2_t = fit_linear_plane_safe(yv, X_t)
        perT[Tval] = {"a0": float(beta_t[0]), "a1": float(beta_t[1]), "R2": float(r2_t)}

    Cplus_over_T = dt_over_T + alpha_abs * sum_kappa_over_T

    X1_plus = np.c_[np.ones_like(Cplus_over_T), Cplus_over_T]
    coef1_plus, r2_line_plus = fit_linear_plane_safe(y_gap, X1_plus)
    a0_plus, a1_plus = map(float, coef1_plus)

    coef, r2 = fit_linear_plane_safe(y, X)

    # alpha* and 1-D collapse
    b0, b_s, b1, b2 = map(float, coef)
    alpha_opt = (b2 / b1) if abs(b1) > 1e-12 else 1.0
    C_over_T = dt_over_T + alpha_opt * sum_kappa_over_T

    # Collapse line fit: y ~ a0 + a1 * (C_over_T)
    X1 = np.c_[np.ones_like(C_over_T), C_over_T]
    coef1, r2_line = fit_linear_plane_safe(y, X1)
    a0, a1 = map(float, coef1)

    # Build regime labels again (for plotting)
    regime = np.array(["mixed"] * len(rows))

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
        f.write(f"R^2  = {r2:.4f}\n")
        f.write("\n--- Risk-gap plane & absolute-budget collapse (theory upper envelope) ---\n")
        f.write("Risk-gap plane: gap ~ b0^g + b_s^g*T^{-1/2} + b1^g*(sum_dt/T) + b2^g*(sum_kappa/T)\n")
        f.write(f"b0^g = {b0g:.6f}, b_s^g = {bsg:.6f}, b1^g = {b1g:.6f}, b2^g = {b2g:.6f}, R^2 = {r2_gap_plane:.4f}\n")
        f.write(f"alpha_abs (|alpha_gap_fit|) = {alpha_abs:.6f}\n")
        f.write(f"Collapse (risk gap): y ≈ {a0_plus:.6f} + {a1_plus:.6f} * (C_T^+/T), R^2 = {r2_line_plus:.4f}\n")
    
    holdout_desc = f"holdout: {holdout_info['strategy']} (T={holdout_info['T']})"
    with open(plane_path, "a") as f:
        f.write("\n--- Held-out collapse (alpha* fit on FIT, evaluated on EVAL) ---\n")
        f.write(f"{holdout_desc}\n")
        f.write(f"n_fit={holdout_info['n_fit']}, n_eval={holdout_info['n_eval']}\n")
        f.write(f"alpha*_fit = {alpha_opt_fit:.6f}\n")
        f.write(f"Eval line (err):   y ≈ {a0_eval:.6f} + {a1_eval:.6f} * (C_T/T), R^2={r2_line_eval:.4f}\n")
        f.write(f"Eval line (gap):   y ≈ {a0_eval_gap:.6f} + {a1_eval_gap:.6f} * (C_T/T), R^2={r2_line_eval_gap:.4f}\n")

    with open(plane_path, "a") as f:
        f.write("\n=== Speed-limit (risk-gap) fit: y ≈ c0 + c1*T^{-1/2} + c2*(C_T/T) ===\n")
        f.write(f"alpha_theory = {alpha_theory:.6f}\n")
        f.write(f"c0 = {c0:.6f}, c1 = {c1:.6f}, c2 = {c2:.6f}, R^2 = {r2_full:.4f}\n")
        f.write("--- Partial-residual collapse (remove T^(-1/2)) ---\n")
        f.write(f"y_res ≈ {a0_res:.6f} + {a1_res:.6f}*(C_T/T), R^2 = {r2_res:.4f}\n")
        f.write("--- Ratio ρ_T = y / (T^{-1/2}+C_T/T) ---\n")
        f.write(f"mean={rho_stats['mean']:.4f}, std={rho_stats['std']:.4f}, "
                f"min={rho_stats['min']:.4f}, max={rho_stats['max']:.4f}\n")

    with open(plane_path, "a") as f:
        f.write("\n=== Ablations on risk gap (theory form) ===\n")
        f.write("Full: y ≈ c0 + c1*T^(-1/2) + c2*(C_T/T)\n")
        f.write(f"c0={c0:.6f}, c1={c1:.6f}, c2={c2:.6f}, R^2={r2_full:.4f}\n")
        f.write(f"Drop T^(-1/2): R^2={r2_noT:.4f}, ΔR^2={delta_R2_noT:.4f}, F={F_noT:.2f}\n")
        f.write(f"Drop C_T/T:    R^2={r2_noC:.4f}, ΔR^2={delta_R2_noC:.4f}, F={F_noC:.2f}\n")
        f.write(f"Partial residual (y - c0 - c1*T^(-1/2)) vs C_T/T: a0={a0_res:.6f}, a1={a1_res:.6f}, R^2={r2_res:.4f}\n")
        f.write("Per-T residual slopes (y_res vs C_T/T):\n")
        for Tval in sorted(perT):
            f.write(f"  T={Tval}: a1={perT[Tval]['a1']:.6f}, R^2={perT[Tval]['R2']:.4f}\n")


    # raw csv
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "T","C_exo_total","C_exo_ratio","gamma","seed","regime",
            "err_T","sum_dt","sum_kappa","gen_gap_traj","traj_emp_risk","traj_pop_risk",
            "pop_mse_initial","pop_mse_final"
        ])

        writer.writeheader(); writer.writerows(raw_rows)

    # summaries
    def mean_se(vals):
        n = len(vals); m = float(np.mean(vals)) if n else float("nan")
        se = float(np.std(vals, ddof=1)/np.sqrt(max(n,1))) if n>1 else float("nan")
        return m, se, n

    group = defaultdict(
        lambda: {
            "err_T": [],
            "sum_dt": [],
            "sum_kappa": [],
            "gen_gap_traj": [],
            "traj_emp_risk": [],
            "traj_pop_risk": [],
            "pop_mse_initial": [],
            "pop_mse_final": [],
            "T": None,
            "ratio": None,
        }
    )
    for r in raw_rows:
        key = (r["T"], r["C_exo_ratio"], r["gamma"])
        g = group[key]
        g["err_T"].append(r["err_T"])
        g["sum_dt"].append(r["sum_dt"])
        g["sum_kappa"].append(r["sum_kappa"])
        g["gen_gap_traj"].append(r["gen_gap_traj"])
        g["traj_emp_risk"].append(r["traj_emp_risk"])
        g["traj_pop_risk"].append(r["traj_pop_risk"])
        g["pop_mse_initial"].append(r["pop_mse_initial"])
        g["pop_mse_final"].append(r["pop_mse_final"])
        g["T"] = r["T"]; g["ratio"] = r["C_exo_ratio"]

    summary_rows = []
    for (Tval, ratio, gamma), g in sorted(group.items()):
        m_err, se_err, n = mean_se(g["err_T"])
        m_dt,  se_dt,  _ = mean_se(g["sum_dt"])
        m_kap, se_kap, _ = mean_se(g["sum_kappa"])
        m_gap, se_gap, _ = mean_se(g["gen_gap_traj"])
        m_emp, se_emp, _ = mean_se(g["traj_emp_risk"])
        m_pop, se_pop, _ = mean_se(g["traj_pop_risk"])
        m_init, se_init, _ = mean_se(g["pop_mse_initial"])
        m_final, se_final, _ = mean_se(g["pop_mse_final"])
        summary_rows.append({
            "T": int(Tval), "C_exo_ratio": float(ratio), "gamma": float(gamma), "regime": "mixed",
            "mean_err_T": m_err, "se_err_T": se_err, "n": int(n),
            "mean_sum_dt": m_dt, "se_sum_dt": se_dt,
            "mean_sum_kappa": m_kap, "se_sum_kappa": se_kap,
            "mean_gen_gap_traj": m_gap, "se_gen_gap_traj": se_gap,
            "mean_traj_emp_risk": m_emp, "se_traj_emp_risk": se_emp,
            "mean_traj_pop_risk": m_pop, "se_traj_pop_risk": se_pop,
            "mean_pop_mse_initial": m_init, "se_pop_mse_initial": se_init,
            "mean_pop_mse_final": m_final, "se_pop_mse_final": se_final,
            "mean_C_over_T(alpha_opt)": float((m_dt + alpha_opt*m_kap) / Tval)
        })

    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["T","C_exo_ratio","gamma","regime","mean_err_T","se_err_T","n",
                        "mean_sum_dt","se_sum_dt","mean_sum_kappa","se_sum_kappa",
                        "mean_gen_gap_traj","se_gen_gap_traj",
                        "mean_traj_emp_risk","se_traj_emp_risk",
                        "mean_traj_pop_risk","se_traj_pop_risk",
                        "mean_pop_mse_initial","se_pop_mse_initial",
                        "mean_pop_mse_final","se_pop_mse_final",
                        "mean_C_over_T(alpha_opt)"]
        )

        writer.writeheader(); writer.writerows(summary_rows)

    traj_pop_initial_vals = [r["traj_pop_initial"] for r in rows if "traj_pop_initial" in r]
    traj_pop_final_vals = [r["traj_pop_final"] for r in rows if "traj_pop_final" in r]

    meta = {
        "figure": "NN Additivity & budget scaling",
        "d_in": cfg.d_in,
        "m_feat": m_feat,
        "nn_architecture": {
            "input_dim": int(cfg.d_in),
            "hidden_dim": int(cfg.hidden_dim),
            "output_dim": 1,
        },
        "T_grid": list(map(int, cfg.T_grid)),
        "seeds": list(map(int, cfg.seeds)), "k_policy": float(cfg.k_policy), "sigma": float(cfg.sigma),
        "C_exo_ratios": list(map(float, cfg.drift_ratios)), "gamma_grid": list(map(float, cfg.gamma_grid)),
        "plane_fit": {"b0": b0, "b_s": b_s, "b1": b1, "b2": b2, "R2": r2},
        "risk_gap_plane": {"b0": b0g, "b_s": bsg, "b1": b1g, "b2": b2g, "R2": r2_gap_plane},
        "risk_gap_plane_fit": {"b0": b0g_fit, "b_s": bsg_fit, "b1": b1g_fit, "b2": b2g_fit, "R2": r2_gap_fit},
        "alpha_opt": float(alpha_opt),
        "alpha_abs_gap": float(alpha_abs),
        "alpha_gap_fit": float(alpha_gap_fit),
        "collapse_fit": {"a0": float(a0), "a1": float(a1), "R2": float(r2_line)},
        "collapse_gap_abs": {"a0": float(a0_plus), "a1": float(a1_plus), "R2": float(r2_line_plus)},
        "speed_limit_full": {
            "alpha_theory": float(alpha_theory),
            "c0": c0, "c1": c1, "c2": c2, "R2": r2_full
        },
        "ablation_drop_T": {
            "a0": a0_noT, "a1": a1_noT, "R2": r2_noT,
            "delta_R2": delta_R2_noT, "F": float(F_noT)
        },
        "ablation_drop_C": {
            "a0": a0_noC, "a1": a1_noC, "R2": r2_noC,
            "delta_R2": delta_R2_noC, "F": float(F_noC)
        },
        "partial_residual": {"a0": a0_res, "a1": a1_res, "R2": r2_res},
        "perT_slopes": {str(t): perT[t] for t in perT},
        "rho_stats": rho_stats,
         "holdout": holdout_info,
        "alpha_opt_fit": float(alpha_opt_fit),
        "eval_collapse_err":  {"a0": float(a0_eval), "a1": float(a1_eval), "R2": float(r2_line_eval)},
        "eval_collapse_gap":  {"a0": float(a0_eval_gap), "a1": float(a1_eval_gap), "R2": float(r2_line_eval_gap)}
    }
    meta["traj_pop_mse_initial_mean"] = float(np.mean(traj_pop_initial_vals)) if traj_pop_initial_vals else float("nan")
    meta["traj_pop_mse_final_mean"] = float(np.mean(traj_pop_final_vals)) if traj_pop_final_vals else float("nan")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balanced NN drift experiment")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN,
        help="Hidden layer width for the learner MLP (default: 128).",
    )
    args = parser.parse_args()
    main(hidden_dim=args.hidden_dim)
