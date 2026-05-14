"""Hypothesis tests for RQ2.

Original hypotheses (Reports 4-6):
  H2.1, H2.2, H2.3 — tested under both Jaccard and semantic similarity.

New hypothesis (Report 7):
  H2.4 (Metric Construct Validity) — semantic similarity will reveal stronger
        effects than Jaccard for H2.1 and H2.2, because Jaccard treats lexically
        distinct but semantically equivalent actions as different.

All tests one-tailed where pre-registered; 10k permutations / bootstraps.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import config


# ---------------- Effect size ----------------
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s2 = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    pooled = np.sqrt(s2)
    return 0.0 if pooled == 0 else (a.mean() - b.mean()) / pooled


# ---------------- Permutation test ----------------
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


# ---------------- Bootstrap CI ----------------
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


# ---------------- Spearman ----------------
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


# ---------------- Fisher z (compare two correlations) ----------------
def fisher_z_test(rho1, n1, rho2, n2) -> dict:
    z1, z2 = np.arctanh(rho1), np.arctanh(rho2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"z": float(z), "p_value": float(p),
            "rho1": float(rho1), "rho2": float(rho2)}


# ================================================================ #
#  Original hypotheses — now parameterized to take any sim metric  #
# ================================================================ #

def evaluate_h21(task_sum: pd.DataFrame, metric: str = "action_sim") -> dict:
    """H2.1 on a given similarity metric: open-ended < constrained."""
    open_mask  = task_sum["family"].isin(["travel", "research"])
    close_mask = task_sum["family"].isin(["debugging", "study"])
    return permutation_test(
        task_sum.loc[open_mask, metric].to_numpy(),
        task_sum.loc[close_mask, metric].to_numpy(),
        alternative="less",
    )


def evaluate_h22(task_sum: pd.DataFrame, metric: str = "action_sim") -> dict:
    """H2.2 on a given similarity metric: constraint rank negatively predicts sim."""
    return spearman_rank(task_sum, metric, alternative="less")


def evaluate_h23(task_sum: pd.DataFrame, sim_metric: str = "action_sim") -> dict:
    """H2.3 dissociation: rank predicts step_diff but not similarity.

    sim_metric controls which similarity metric defines rho_J.
    """
    study_vals    = task_sum.loc[task_sum["family"] == "study",    "step_diff"].to_numpy()
    research_vals = task_sum.loc[task_sum["family"] == "research", "step_diff"].to_numpy()

    rho_J = spearman_rank(task_sum, sim_metric, alternative="less")
    rho_S = spearman_rank(task_sum, "step_diff", alternative="greater")

    return {
        "study_vs_research_step_diff": permutation_test(
            study_vals, research_vals, alternative="greater"),
        "rho_jaccard":   rho_J,  # name kept for backward compat with Report 6
        "rho_step_diff": rho_S,
        "fisher_z_dissociation": fisher_z_test(
            rho_J["rho"], rho_J["n"], rho_S["rho"], rho_S["n"]),
        "sim_metric_used": sim_metric,
    }


# ================================================================ #
#  Report 7 — H2.4 (Metric Construct Validity)                      #
# ================================================================ #

def evaluate_h24(task_sum: pd.DataFrame) -> dict:
    """H2.4: Semantic similarity reveals stronger effects than Jaccard.

    Compares H2.1 and H2.2 test results between Jaccard (action_sim) and
    semantic similarity (semantic_sim). H2.4 is supported if semantic
    similarity produces a larger |Cohen's d| (for H2.1) and a more
    significant Spearman correlation (for H2.2).
    """
    h21_jaccard  = evaluate_h21(task_sum, metric="action_sim")
    h21_semantic = evaluate_h21(task_sum, metric="semantic_sim")
    h22_jaccard  = evaluate_h22(task_sum, metric="action_sim")
    h22_semantic = evaluate_h22(task_sum, metric="semantic_sim")

    # Direct quantitative comparison
    h21_d_improved = abs(h21_semantic["cohens_d"]) > abs(h21_jaccard["cohens_d"])
    h22_rho_improved = (
        h22_semantic["rho"] < h22_jaccard["rho"]  # more negative is "stronger" given one-tailed less
    )

    return {
        "h21_under_jaccard":   h21_jaccard,
        "h21_under_semantic":  h21_semantic,
        "h22_under_jaccard":   h22_jaccard,
        "h22_under_semantic":  h22_semantic,
        "h21_effect_size_improved_by_semantic": bool(h21_d_improved),
        "h22_correlation_improved_by_semantic": bool(h22_rho_improved),
    }


# ================================================================ #
#  Run everything                                                    #
# ================================================================ #

def run_all_tests(task_sum_base: pd.DataFrame, task_sum_tool: pd.DataFrame) -> dict:
    results = {}
    for label, ts in (("base", task_sum_base), ("tool", task_sum_tool)):
        results[label] = {
            # Original hypotheses on Jaccard (Report 6 results)
            "H2_1_jaccard": {
                "action_sim": evaluate_h21(ts, metric="action_sim"),
                "seq_sim":    evaluate_h21(ts, metric="seq_sim"),
            },
            "H2_2_jaccard": {
                "action_sim": evaluate_h22(ts, metric="action_sim"),
                "seq_sim":    evaluate_h22(ts, metric="seq_sim"),
            },
            "H2_3_jaccard": evaluate_h23(ts, sim_metric="action_sim"),

            # Same hypotheses re-tested on semantic similarity (Report 7)
            "H2_1_semantic": evaluate_h21(ts, metric="semantic_sim"),
            "H2_2_semantic": evaluate_h22(ts, metric="semantic_sim"),
            "H2_3_semantic": evaluate_h23(ts, sim_metric="semantic_sim"),

            # H2.4 direct comparison
            "H2_4": evaluate_h24(ts),
        }
    return results