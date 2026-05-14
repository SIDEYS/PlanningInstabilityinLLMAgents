import numpy as np
import pandas as pd
from scipy import stats

from . import config


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s2 = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    pooled = np.sqrt(s2)
    return 0.0 if pooled == 0 else (a.mean() - b.mean()) / pooled


def permutation_test(group_a, group_b, alternative="less",
                     n_perm=config.N_PERMUTATIONS, seed=config.RANDOM_SEED) -> dict:
    rng = np.random.default_rng(seed)
    a, b = np.asarray(group_a, dtype=float), np.asarray(group_b, dtype=float)
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)
    null = np.empty(n_perm)
    for k in range(n_perm):
        rng.shuffle(pooled)
        null[k] = pooled[:n_a].mean() - pooled[n_a:].mean()
    if alternative == "less":
        p = (np.sum(null <= obs) + 1) / (n_perm + 1)
    elif alternative == "greater":
        p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return {"observed_diff": float(obs), "p_value": float(p),
            "n_a": int(n_a), "n_b": int(len(b)),
            "cohens_d": float(cohens_d(a, b)), "alternative": alternative}


def bootstrap_ci(values, n_boot=config.N_BOOTSTRAP, ci=0.95,
                 seed=config.RANDOM_SEED) -> tuple:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    n = len(values)
    means = np.empty(n_boot)
    for k in range(n_boot):
        means[k] = rng.choice(values, size=n, replace=True).mean()
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def spearman_rank(task_sum, metric, alternative="less") -> dict:
    df = task_sum.copy()
    df["constraint_rank"] = df["family"].map(config.CONSTRAINT_RANK)
    rho, p_two = stats.spearmanr(df["constraint_rank"], df[metric])
    if np.isnan(rho):
        p = float("nan")
    elif alternative == "less":
        p = p_two / 2 if rho < 0 else 1 - p_two / 2
    elif alternative == "greater":
        p = p_two / 2 if rho > 0 else 1 - p_two / 2
    else:
        p = p_two
    return {"metric": metric, "rho": float(rho), "p_value": float(p),
            "alternative": alternative, "n": int(len(df))}


def fisher_z_test(rho1, n1, rho2, n2) -> dict:
    z1, z2 = np.arctanh(rho1), np.arctanh(rho2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"z": float(z), "p_value": float(p),
            "rho1": float(rho1), "rho2": float(rho2)}


def evaluate_h21(task_sum) -> dict:
    open_mask  = task_sum["family"].isin(["travel", "research"])
    close_mask = task_sum["family"].isin(["debugging", "study"])
    return {m: permutation_test(task_sum.loc[open_mask, m].to_numpy(),
                                task_sum.loc[close_mask, m].to_numpy(),
                                alternative="less")
            for m in ("action_sim", "seq_sim")}


def evaluate_h22(task_sum) -> dict:
    return {"action_sim": spearman_rank(task_sum, "action_sim", "less"),
            "seq_sim":    spearman_rank(task_sum, "seq_sim",    "less")}


def evaluate_h23(task_sum) -> dict:
    study_vals    = task_sum.loc[task_sum["family"] == "study",    "step_diff"].to_numpy()
    research_vals = task_sum.loc[task_sum["family"] == "research", "step_diff"].to_numpy()
    rho_J = spearman_rank(task_sum, "action_sim", alternative="less")
    rho_S = spearman_rank(task_sum, "step_diff",  alternative="greater")
    return {
        "study_vs_research_step_diff":
            permutation_test(study_vals, research_vals, alternative="greater"),
        "rho_jaccard":   rho_J,
        "rho_step_diff": rho_S,
        "fisher_z_dissociation":
            fisher_z_test(rho_J["rho"], rho_J["n"], rho_S["rho"], rho_S["n"]),
    }


def run_all_tests(task_sum_base, task_sum_tool) -> dict:
    return {
        "base": {"H2_1": evaluate_h21(task_sum_base),
                 "H2_2": evaluate_h22(task_sum_base),
                 "H2_3": evaluate_h23(task_sum_base)},
        "tool": {"H2_1": evaluate_h21(task_sum_tool),
                 "H2_2": evaluate_h22(task_sum_tool),
                 "H2_3": evaluate_h23(task_sum_tool)},
    }