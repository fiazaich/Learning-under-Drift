import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# ----------------------- Data structures & logging -----------------------

@dataclass
class RunStats:
    err_T: float                # || mu_hat_T - theta_T ||_2
    gap_abs: float              # |R_T - R| as defined below (final-model definition)
    sum_dt: float               # sum_t || v_exo_t ||_2
    sum_kappa: float            # sum_t gamma * || theta_t - mu_hat_t ||_2
    fisher_path_len: float      # sum of Fisher step lengths (optional per run)
    traj_emp_risk: float        # trajectory-averaged empirical loss
    traj_pop_risk: float        # trajectory-averaged population loss
    traj_gap: float             # |traj_emp_risk - traj_pop_risk|
    pop_mse_initial: float      # population MSE at t=1
    pop_mse_final: float        # population MSE at t=T

# ------------------------------ Utilities --------------------------------

def cholesky_sampler(Sigma: np.ndarray, rng: np.random.Generator):
    """Return function to sample N(theta, Sigma) using one Cholesky."""
    L = np.linalg.cholesky(Sigma)
    d = Sigma.shape[0]
    def sample(theta: np.ndarray, size: int = None) -> np.ndarray:
        if size is None:
            z = rng.standard_normal(d)
            return theta + L @ z
        else:
            z = rng.standard_normal((size, d))
            return theta + z @ L.T
    return sample

def throttle_path_length(vec: np.ndarray, remaining_budget: float) -> np.ndarray:
    """
    Scale 'vec' (shape (d,)) so that its norm contributes appropriately to a
    remaining L2 budget. If remaining_budget <= 0, return zeros.
    """
    if remaining_budget <= 0:
        return np.zeros_like(vec)
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return np.zeros_like(vec)
    # Clamp to not exceed remaining budget in one step
    scale = min(remaining_budget / n, 1.0)
    return vec * scale

def fit_linear_plane(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ordinary least squares: y ~ X @ coef.
    Returns (coef, R^2).
    """
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return coef, r2

def fisher_step_length(delta: np.ndarray, Sigma_inv: np.ndarray) -> float:
    """sqrt( delta^T Sigma^{-1} delta )."""
    return float(np.sqrt(delta @ Sigma_inv @ delta))

# --------------------------- Core simulation -----------------------------

def simulate_one_run(
    T: int,
    d: int,
    Sigma: np.ndarray,
    regime: str,         # "iid", "exo", "endogenous", or "mixed"
    C_exo: float,        # total exogenous path budget used when regime includes exogenous drift
    gamma: float,        # action-to-environment gain
    policy_k: float,     # u_t = -k * muhat_t  (used when endogenous actions are present)
    seed: int
) -> RunStats:
    """
    Single trajectory simulation matching your scaffolding. Returns final stats.
    We compute both err_T and |R_T - R|:
      - err_T = || mu_hat_T - theta_T ||_2
      - R     = E_{x ~ N(theta_T, Sigma)} || x - mu_hat_T ||_2^2  (Monte Carlo)
      - R_T   = (1/T) * sum_{t=1..T} E_{x ~ N(theta_t, Sigma)} || x - mu_hat_T ||_2^2
    """
    rng = np.random.default_rng(seed)
    sampler = cholesky_sampler(Sigma, rng)
    Sigma_inv = np.linalg.inv(Sigma)

    theta = np.zeros(d)
    muhat = np.zeros(d)

    # Precompute exogenous vectors for regimes with drift budget.
    exo_vecs = np.zeros((T, d))
    if regime in ("exo", "mixed"):
        raw = rng.normal(size=(T, d))
        norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12
        unit = raw / norms
        exo_vecs = unit * (C_exo / T)

    sum_dt = 0.0
    sum_kappa = 0.0
    fisher_path_len = 0.0

    # Keep theta history for R_T
    thetas = np.zeros((T, d))
    trSigma = float(np.trace(Sigma))
    train_losses = []
    pop_losses = []

    for t in range(1, T + 1):
        muhat_prev = muhat.copy()

        # sample x_t ~ N(theta_t, Sigma)
        x_t = sampler(theta)

        train_losses.append(float(np.sum((muhat_prev - x_t) ** 2)))
        pop_losses.append(float(trSigma + np.sum((theta - muhat_prev) ** 2)))

        # online sample mean (eta_t = 1/t)
        muhat = muhat_prev + (x_t - muhat_prev) / t

        # choose action
        u = np.zeros(d)
        if regime in ("endogenous", "mixed"):
            u = -policy_k * muhat

        # exogenous vector
        v_exo = exo_vecs[t - 1] if regime in ("exo", "mixed") else np.zeros(d)

        # environment evolves
        theta_next = theta + v_exo + gamma * u

        # log path terms
        d_t = np.linalg.norm(v_exo)
        kappa_t = abs(gamma) * float(np.sqrt((u @ Sigma_inv @ u)))

        sum_dt += d_t
        sum_kappa += kappa_t

        # Fisher step length
        fisher_path_len += fisher_step_length(theta_next - theta, Sigma_inv)

        thetas[t - 1] = theta
        theta = theta_next

    # Final error
    err_T = float(np.linalg.norm(muhat - theta))

    # Compute R and R_T via Monte Carlo (frozen environment vs path-average)
    # R = E || X - muhat ||^2, X ~ N(theta_T, Sigma)
    N_mc = 5000
    test = sampler(theta, size=N_mc)
    R = float(np.mean(np.sum((test - muhat) ** 2, axis=1)))
    


    # R_T = (1/T) sum_t E || X - muhat ||^2, X ~ N(theta_t, Sigma)
    # Monte Carlo per time can be expensive; instead use identity:
    # For Gaussian X ~ N(m, Sigma), E||X - a||^2 = trace(Sigma) + ||m - a||^2
    trSigma = float(np.trace(Sigma))
    # R = trSigma + np.sum((theta - muhat)**2)
    diffs = thetas - muhat  # broadcast
    R_T = float(np.mean(trSigma + np.sum(diffs * diffs, axis=1)))

    gap_abs = abs(R_T - R)
    traj_emp_risk = float(np.mean(train_losses)) if train_losses else float("nan")
    traj_pop_risk = float(np.mean(pop_losses)) if pop_losses else float("nan")
    traj_gap = float(abs(traj_emp_risk - traj_pop_risk))
    pop_mse_initial = float(pop_losses[0]) if pop_losses else float("nan")
    pop_mse_final = float(pop_losses[-1]) if pop_losses else float("nan")

    return RunStats(
        err_T=err_T,
        gap_abs=gap_abs,
        sum_dt=sum_dt,
        sum_kappa=sum_kappa,
        fisher_path_len=fisher_path_len,
        traj_emp_risk=traj_emp_risk,
        traj_pop_risk=traj_pop_risk,
        traj_gap=traj_gap,
        pop_mse_initial=pop_mse_initial,
        pop_mse_final=pop_mse_final,
    )

def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
