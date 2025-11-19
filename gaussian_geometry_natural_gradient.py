# Geometry / Natural-Gradient experiment with Fix A (aligned Fisher objective) and Fix B (equalized progress)
# Requires: common_sim.py in the same directory (uses fisher_step_length).

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

import scienceplots
plt.style.use(['science','ieee'])

import common_sim as cs  # fisher_step_length
from out_utils import create_results_dir


@dataclass
class SeedResult:
    seed: int
    mode: str
    cum_fisher_van: float
    steps_van: int
    cum_fisher_nat: float
    steps_nat: int
    J0: float
    Jfinal_van: float
    Jfinal_nat: float


@dataclass
class OutputPaths:
    csv: Path
    json: Path
    plot_prefix: Path


def build_output_paths(results_dir: Path, out_prefix: str, mode: str) -> OutputPaths:
    file_prefix = f"{out_prefix}_{mode}"
    return OutputPaths(
        csv=results_dir / f"{file_prefix}.csv",
        json=results_dir / f"{file_prefix}.json",
        plot_prefix=results_dir / file_prefix,
    )


# ----------------------------- Helpers -----------------------------

def paired_summary(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Return means/SEs for a, b, and paired difference (a-b), plus a simple t-stat."""
    assert a.shape == b.shape
    n = a.size
    def mean_se(x):
        m = float(np.mean(x))
        se = float(np.std(x, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        return m, se
    m_a, se_a = mean_se(a)
    m_b, se_b = mean_se(b)
    diff = a - b
    m_d, se_d = mean_se(diff)
    t_stat = m_d / (se_d + 1e-12) if (n > 1 and se_d > 0) else (float("inf") if m_d != 0 else 0.0)
    return {
        "mean_a": m_a, "se_a": se_a,
        "mean_b": m_b, "se_b": se_b,
        "mean_diff": m_d, "se_diff": se_d,
        "t_stat": t_stat,
        "n": float(n),
    }


def save_csv(per_seed: List[Dict[str, object]], path: str) -> None:
    import csv
    if not per_seed:
        return
    keys = list(per_seed[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in per_seed:
            w.writerow(row)


def plot_bars(summary: Dict[str, object], out_prefix: str) -> None:
    if not HAVE_PLT:
        print("[INFO] matplotlib not available; skip plot.")
        return

    cum = summary["cum_fisher_summary"]
    stp  = summary["steps_summary"]

    import scienceplots
    with plt.style.context(['science', 'ieee', {'text.usetex': False}]):
        # same physical size as your other figs; constrained layout handles spacing
        fig, axes = plt.subplots(1, 2, figsize=(3.2, 2.0), layout='constrained', sharey=False)

        labels  = ["Euclidean\n(flat-metric)", "Fisher–Rao\n(information-geometric)"]
        palette = ["#7c8a96", "#2f6b9a"]

        # lighter, science-y errorbar styling
        errkw = dict(ecolor='0.25', lw=0.8, capsize=2.0, capthick=0.8)

        def draw_panel(ax, means, errors, title, ylabel):
            ax.bar([0, 1], means, yerr=errors, color=palette, width=0.55,
                   edgecolor='none', error_kw=errkw)
            ax.set_xticks([0, 1], labels)
            ax.set_title(title, fontsize='small', pad=2.0)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=6)
            ax.grid(axis='y', which='major', alpha=0.45)
            ax.tick_params(axis='x', labelsize='x-small')  # two-line tick labels
            ax.tick_params(axis='y', labelsize='small')

        draw_panel(
            axes[0],
            [cum["mean_a"], cum["mean_b"]],
            [cum["se_a"],  cum["se_b"]],
            "Cumulative Fisher length",
            "path length",
        )
        draw_panel(
            axes[1],
            [stp["mean_a"], stp["mean_b"]],
            [stp["se_a"],  stp["se_b"]],
            "Steps to target",
            "",
        )

        # small caption-style note below the panels
        fig.text(0.5, -0.03,
                 f"paired Δ (steps): {stp['mean_diff']:.1f} ± {stp['se_diff']:.1f}",
                 ha='center', fontsize='x-small', color='0.3')

        for ext in ("pdf", "svg", "png"):
            fig.savefig(f"{out_prefix}.{ext}",
                        dpi=(600 if ext == "png" else None),
                        bbox_inches="tight")
        plt.close(fig)



# ----------------------------- Fix A -----------------------------
# Align objective to Fisher metric: J_F(theta) = 0.5 * theta^T G theta
# Policies:
#   Vanilla: u = -beta_van * (G @ theta)
#   Natural: u = -beta_nat * theta
# Fisher step length: sqrt(delta^T G delta)

def J_F(theta: np.ndarray, G: np.ndarray) -> float:
    return 0.5 * float(theta @ (G @ theta))


def run_fixA_once(
    theta0: np.ndarray,
    Sigma: np.ndarray,
    G: np.ndarray,
    beta_nat: float,
    beta_van: float,
    gamma: float,
    tol_factor: float,
    max_steps: int,
) -> Tuple[Tuple[float, int, float], Tuple[float, int, float], float]:
    """Return ((cumF_van, steps_van, Jfinal_van), (cumF_nat, steps_nat, Jfinal_nat), J0)."""
    Sigma_inv = np.linalg.inv(Sigma)

    def roll(policy: str) -> Tuple[float, int, float]:
        theta = theta0.copy()
        J0 = J_F(theta, G)
        target = J0 / tol_factor
        cumF = 0.0
        steps = 0
        while J_F(theta, G) > target and steps < max_steps:
            if policy == "vanilla":
                u = -beta_van * (G @ theta)
            else:
                u = -beta_nat * theta
            theta_next = theta + gamma * u
            delta = theta_next - theta
            cumF += cs.fisher_step_length(delta, Sigma_inv)
            theta = theta_next
            steps += 1
        return float(cumF), int(steps), float(J_F(theta, G))

    res_v = roll("vanilla")
    res_n = roll("natural")
    return res_v, res_n, float(J_F(theta0, G))


# ----------------------------- Fix B -----------------------------
# Keep Euclidean objective: J(theta) = 0.5 * ||theta||^2
# Directions:
#   Vanilla: d_v = -theta
#   Natural: d_n = -Sigma @ theta
# Choose per-step step size so J_{t+1} = rho * J_t exactly for both methods.

def J_E(theta: np.ndarray) -> float:
    return 0.5 * float(theta @ theta)


def positive_step_sizes(theta: np.ndarray,
                        d_eff: np.ndarray,
                        rho: float,
                        M: np.ndarray) -> List[float]:
    thMth = float(theta @ (M @ theta))
    if thMth <= 0:
        return []
    thMd  = float(theta @ (M @ d_eff))
    dMd   = float(d_eff @ (M @ d_eff))
    a, b, c = dMd, 2.0*thMd, (1.0 - rho)*thMth
    disc = b*b - 4.0*a*c
    if disc < 0 and disc > -1e-12:
        disc = 0.0
    if disc < 0 or a <= 0:
        return []
    sqrt_disc = math.sqrt(disc)
    beta1 = (-b + sqrt_disc) / (2.0*a)
    beta2 = (-b - sqrt_disc) / (2.0*a)
    return sorted([b for b in (beta1, beta2) if b > 0])


def run_fixB_once(
    theta0: np.ndarray,
    Sigma: np.ndarray,
    rho: float,
    gamma: float,
    tol_factor: float,
    max_steps: int,
    stop_on="E"
) -> Tuple[Tuple[float, int, float], Tuple[float, int, float], float]:
    G = np.linalg.inv(Sigma)
    Id = np.eye(theta0.size)

    def L_E(theta):
        return 0.5 * float(theta @ theta)

    def roll(policy: str) -> Tuple[float, int, float]:
        theta = theta0.copy()
        L0 = L_E(theta)
        target = L0 / tol_factor
        cumF = 0.0
        steps = 0

        while L_E(theta) > target and steps < max_steps:
            direction = -theta if policy == "vanilla" else -(Sigma @ theta)
            d_eff = gamma * direction
            betas = positive_step_sizes(theta, d_eff, rho, Id)
            L_prev = L_E(theta)
            success = False

            if betas:
                for beta in reversed(betas):  # try larger step first (better collapse)
                    delta = beta * d_eff
                    theta_try = theta + delta
                    L_try = L_E(theta_try)
                    if L_try <= rho * L_prev + 1e-10:
                        theta = theta_try
                        cumF += cs.fisher_step_length(delta, G)
                        steps += 1
                        success = True
                        break

            if not success:
                beta_base = 0.10 if policy == "vanilla" else 0.50
                beta = beta_base
                shrink = 1.0
                accepted = False
                for _ in range(20):
                    delta = shrink * beta * d_eff
                    theta_try = theta + delta
                    L_try = L_E(theta_try)
                    if L_try <= rho * L_prev + 1e-10:
                        theta = theta_try
                        cumF += cs.fisher_step_length(delta, G)
                        steps += 1
                        accepted = True
                        break
                    shrink *= 0.5
                if not accepted:
                    break

        return float(cumF), int(steps), float(L_E(theta))

    res_v = roll("vanilla")
    res_n = roll("natural")
    return res_v, res_n, float(L_E(theta0))


# ----------------------------- Driver -----------------------------

def run_experiment(
    mode: str,
    d: int,
    Sigma_diag: List[float],
    n_inits: int,
    gamma: float,
    tol_factor: float,
    max_steps: int,
    seed: int,
    beta_nat: float,
    beta_van: float,
    rho: float,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    Sigma = np.diag(np.asarray(Sigma_diag, dtype=float))
    G = np.linalg.inv(Sigma)

    results: List[SeedResult] = []

    for i in range(n_inits):
        theta0 = rng.standard_normal(d)
        if mode == "fixA":
            (cumF_v, steps_v, Jf_v), (cumF_n, steps_n, Jf_n), J0 = run_fixA_once(
                theta0, Sigma, G, beta_nat, beta_van, gamma, tol_factor, max_steps
            )
        else:
            (cumF_v, steps_v, Jf_v), (cumF_n, steps_n, Jf_n), J0 = run_fixB_once(
                theta0, Sigma, rho, gamma, tol_factor, max_steps
            )
        results.append(
            SeedResult(
                seed=int(seed + i),
                mode=mode,
                cum_fisher_van=float(cumF_v),
                steps_van=int(steps_v),
                cum_fisher_nat=float(cumF_n),
                steps_nat=int(steps_n),
                J0=float(J0),
                Jfinal_van=float(Jf_v),
                Jfinal_nat=float(Jf_n),
            )
        )

    # Aggregate
    cum_v = np.array([r.cum_fisher_van for r in results], dtype=float)
    cum_n = np.array([r.cum_fisher_nat for r in results], dtype=float)
    stp_v = np.array([r.steps_van for r in results], dtype=float)
    stp_n = np.array([r.steps_nat for r in results], dtype=float)

    cum_sum = paired_summary(cum_v, cum_n)
    stp_sum = paired_summary(stp_v, stp_n)

    summary = {
        "mode": mode,
        "Sigma_diag": Sigma_diag,
        "d": d,
        "n_inits": n_inits,
        "gamma": gamma,
        "tol_factor": tol_factor,
        "max_steps": max_steps,
        "seed": seed,
        "beta_nat": beta_nat,
        "beta_van": beta_van,
        "rho": rho,
        "cum_fisher_summary": cum_sum,
        "steps_summary": stp_sum,
    }

    return {"summary": summary, "per_seed": [asdict(r) for r in results]}


def main():
    ap = argparse.ArgumentParser(description="Figure 3 Geometry / Natural-Gradient experiment")
    ap.add_argument("--mode", choices=["fixA", "fixB"], default="fixA",
                    help="fixA: J_F(theta)=0.5 theta^T G theta; fixB: J=0.5||theta||^2 with equalized per-step drop.")
    ap.add_argument("--d", type=int, default=5, help="Dimensionality.")
    ap.add_argument("--Sigma_diag", type=str, default="0.1,1.0,5.0,0.5,2.0",
                    help="Comma-separated diagonal of Sigma (non-identity).")
    ap.add_argument("--n_inits", type=int, default=100, help="Number of paired initial thetas.")
    ap.add_argument("--gamma", type=float, default=1.0, help="Action-to-environment gain.")
    ap.add_argument("--tol_factor", type=float, default=100.0, help="Stop when J <= J0 / tol_factor.")
    ap.add_argument("--max_steps", type=int, default=10000, help="Safety cap on iterations.")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    ap.add_argument("--beta_nat", type=float, default=0.5, help="[fixA] Natural step size.")
    ap.add_argument("--beta_van", type=float, default=-1.0,
                    help="[fixA] Vanilla step size; default -1 means use 1/lambda_max(G).")
    ap.add_argument("--rho", type=float, default=0.9, help="[fixB] Per-step multiplicative drop for J.")
    ap.add_argument("--out_prefix", type=str, default="fig3_geometry", help="Output file prefix.")
    ap.add_argument("--plot", action="store_true", help="Save PNG bar chart (requires matplotlib).")
    args = ap.parse_args()
    results_dir = create_results_dir(f"gaussian_geometry_{args.mode}")

    Sigma_diag = [float(x.strip()) for x in args.Sigma_diag.split(",")]
    if len(Sigma_diag) != args.d:
        raise ValueError(f"Sigma_diag length {len(Sigma_diag)} must equal d={args.d}")
    Sigma = np.diag(np.asarray(Sigma_diag, dtype=float))
    G = np.linalg.inv(Sigma)

    beta_van = args.beta_van
    if args.mode == "fixA":
        # Stability/near-optimal choice if unset: beta_van = 1 / lambda_max(G)
        if beta_van <= 0:
            lam_max = float(np.linalg.eigvalsh(G).max())
            beta_van = 1.0 / lam_max

    out = run_experiment(
        mode=args.mode,
        d=args.d,
        Sigma_diag=Sigma_diag,
        n_inits=args.n_inits,
        gamma=args.gamma,
        tol_factor=args.tol_factor,
        max_steps=args.max_steps,
        seed=args.seed,
        beta_nat=args.beta_nat,
        beta_van=beta_van,
        rho=args.rho,
    )

    # Save
    paths = build_output_paths(results_dir, args.out_prefix, args.mode)
    save_csv(out["per_seed"], paths.csv)
    with open(paths.json, "w") as f:
        json.dump(out, f, indent=2)

    # Print
    s = out["summary"]
    cum = s["cum_fisher_summary"]
    stp = s["steps_summary"]
    print("\n=== Geometry / Natural-Gradient Experiment ===")
    print(f"Mode: {s['mode']}")
    print(f"Sigma diag: {s['Sigma_diag']}")
    print(f"d={s['d']}, n_inits={s['n_inits']}, gamma={s['gamma']}, tol_factor={s['tol_factor']}, "
          f"max_steps={s['max_steps']}, seed={s['seed']}")
    if s["mode"] == "fixA":
        print(f"[fixA] beta_nat={s['beta_nat']}, beta_van={s['beta_van']}")
    else:
        print(f"[fixB] rho={s['rho']} (exact per-step multiplicative drop)")

    print("\nCumulative Fisher Path Length (lower is better):")
    print(f"  Vanilla: mean={cum['mean_a']:.4f}  SE={cum['se_a']:.4f}")
    print(f"  Natural: mean={cum['mean_b']:.4f}  SE={cum['se_b']:.4f}")
    print(f"  Paired diff (Vanilla - Natural): mean={cum['mean_diff']:.4f}  SE={cum['se_diff']:.4f}  "
          f"t={cum['t_stat']:.3f}  n={int(cum['n'])}")

    print("\nSteps to Target:")
    print(f"  Vanilla: mean={stp['mean_a']:.2f}  SE={stp['se_a']:.2f}")
    print(f"  Natural: mean={stp['mean_b']:.2f}  SE={stp['se_b']:.2f}")
    print(f"  Paired diff (Vanilla - Natural): mean={stp['mean_diff']:.2f}  SE={stp['se_diff']:.2f}  "
          f"t={stp['t_stat']:.3f}  n={int(stp['n'])}")

    if args.plot:
        plot_bars(s, paths.plot_prefix)
        print(f"\nSaved: {paths.csv}, {paths.json}, {paths.plot_prefix}.pdf/.svg/.png")
    else:
        print(f"\nSaved: {paths.csv}, {paths.json}")


if __name__ == "__main__":
    main()
