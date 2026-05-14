"""All figures for Report 7. Saves PNGs to figures/.
Existing Report 6 figures regenerated identically; new figures added
for the semantic-similarity analysis (H2.4).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config
from .statistical_tests import bootstrap_ci


def _ordered_families(df):
    return [f for f in config.FAMILY_ORDER if f in df["family"].unique()]


def _bar_colors(families):
    return [config.FAMILY_COLORS[f] for f in families]


def family_bar_with_ci(task_sum, metric, title, ylabel, out_path):
    families = _ordered_families(task_sum)
    means, los, his = [], [], []
    for fam in families:
        vals = task_sum.loc[task_sum["family"] == fam, metric].to_numpy()
        means.append(vals.mean())
        lo, hi = bootstrap_ci(vals)
        los.append(lo); his.append(hi)
    err_lo = np.array(means) - np.array(los)
    err_hi = np.array(his) - np.array(means)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(families, means, yerr=[err_lo, err_hi], capsize=6,
                  color=_bar_colors(families))
    if "sim" in metric or "match" in metric or "agree" in metric:
        ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Task family (most -> least constrained)")
    ax.set_title(title)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.2f}",
                ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def step_diff_boxplot(pairwise, title, out_path):
    families = _ordered_families(pairwise)
    data = [pairwise.loc[pairwise["family"] == f, "step_diff"].to_numpy()
            for f in families]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, labels=families, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], _bar_colors(families)):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel("|steps_A - steps_B|")
    ax.set_xlabel("Task family")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def constraint_rank_scatter(task_sum, metric, title, ylabel, out_path, seed=0):
    rng = np.random.default_rng(seed)
    df = task_sum.copy()
    df["rank"] = df["family"].map(config.CONSTRAINT_RANK)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fam, sub in df.groupby("family"):
        jitter = rng.uniform(-0.08, 0.08, len(sub))
        ax.scatter(sub["rank"] + jitter, sub[metric],
                   color=config.FAMILY_COLORS[fam], label=fam, s=60, alpha=0.85)
    z = np.polyfit(df["rank"], df[metric], 1)
    xs = np.linspace(0.8, 4.2, 50)
    ax.plot(xs, np.polyval(z, xs), "k--", alpha=0.5, label="OLS fit")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["debugging", "travel", "study", "research"])
    ax.set_xlabel("Constraint rank (1 = most constrained)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def jaccard_vs_semantic_grouped(task_sum, planner_label, out_path):
    """Side-by-side bar chart: Jaccard (action_sim) vs semantic_sim per family.

    This is the headline H2.4 figure: visualizes how much higher semantic
    similarity scores are than Jaccard across families.
    """
    families = _ordered_families(task_sum)
    jac_means, sem_means = [], []
    jac_err_lo, jac_err_hi, sem_err_lo, sem_err_hi = [], [], [], []
    for fam in families:
        jvals = task_sum.loc[task_sum["family"] == fam, "action_sim"].to_numpy()
        svals = task_sum.loc[task_sum["family"] == fam, "semantic_sim"].to_numpy()
        jm, sm = jvals.mean(), svals.mean()
        jlo, jhi = bootstrap_ci(jvals)
        slo, shi = bootstrap_ci(svals)
        jac_means.append(jm); sem_means.append(sm)
        jac_err_lo.append(jm - jlo); jac_err_hi.append(jhi - jm)
        sem_err_lo.append(sm - slo); sem_err_hi.append(shi - sm)

    x = np.arange(len(families)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w/2, jac_means, w, yerr=[jac_err_lo, jac_err_hi],
           capsize=4, label="Jaccard (action_sim)", color="#4C78A8")
    ax.bar(x + w/2, sem_means, w, yerr=[sem_err_lo, sem_err_hi],
           capsize=4, label="Semantic similarity", color="#E45756")
    ax.set_xticks(x); ax.set_xticklabels(families)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean task-level similarity")
    ax.set_xlabel("Task family (most -> least constrained)")
    ax.set_title(f"Jaccard vs semantic similarity by family ({planner_label})")
    ax.legend(loc="lower right")
    for i, (jm, sm) in enumerate(zip(jac_means, sem_means)):
        ax.text(x[i] - w/2, jm + 0.025, f"{jm:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + w/2, sm + 0.025, f"{sm:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def generate_all_figures(pw_base, ts_base, fs_base,
                         pw_tool, ts_tool, fs_tool, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Report 6 figures (regenerated identically) ----------
    family_bar_with_ci(ts_base, "action_sim",
        "Action-set similarity by family (Base, 95% bootstrap CI)",
        "Mean Jaccard similarity", out_dir / "action_jaccard_base.png")
    family_bar_with_ci(ts_tool, "action_sim",
        "Action-set similarity by family (Tool-aware, 95% bootstrap CI)",
        "Mean Jaccard similarity", out_dir / "action_jaccard_tool.png")
    family_bar_with_ci(ts_base, "seq_sim",
        "Sequence similarity by family (Base, 95% bootstrap CI)",
        "Mean sequence similarity", out_dir / "sequence_similarity_base.png")
    family_bar_with_ci(ts_tool, "seq_sim",
        "Sequence similarity by family (Tool-aware, 95% bootstrap CI)",
        "Mean sequence similarity", out_dir / "sequence_similarity_tool.png")
    step_diff_boxplot(pw_base,
        "Step-count difference by family (Base)",
        out_dir / "step_count_diff_base.png")
    step_diff_boxplot(pw_tool,
        "Step-count difference by family (Tool-aware)",
        out_dir / "step_count_diff_tool.png")
    if "tool_agree" in fs_tool.columns:
        family_bar_with_ci(ts_tool, "tool_agree",
            "Tool agreement by family (Tool-aware, 95% bootstrap CI)",
            "Mean tool agreement", out_dir / "tool_agreement_tool.png")
    constraint_rank_scatter(ts_base, "action_sim",
        "Constraint rank vs action-set similarity (Base)",
        "Mean task-level Jaccard similarity",
        out_dir / "rank_vs_jaccard_base.png")
    constraint_rank_scatter(ts_base, "step_diff",
        "Constraint rank vs step-count difference (Base)",
        "Mean task-level step-count difference",
        out_dir / "rank_vs_stepdiff_base.png")

    # ---------- Report 7 additions: semantic similarity ----------
    family_bar_with_ci(ts_base, "semantic_sim",
        "Semantic similarity by family (Base, 95% bootstrap CI)",
        "Mean semantic similarity", out_dir / "semantic_sim_base.png")
    family_bar_with_ci(ts_tool, "semantic_sim",
        "Semantic similarity by family (Tool-aware, 95% bootstrap CI)",
        "Mean semantic similarity", out_dir / "semantic_sim_tool.png")

    # Headline H2.4 figure: Jaccard vs Semantic side by side
    jaccard_vs_semantic_grouped(ts_base, "Base planner",
        out_dir / "jaccard_vs_semantic_base.png")
    jaccard_vs_semantic_grouped(ts_tool, "Tool-aware planner",
        out_dir / "jaccard_vs_semantic_tool.png")

    # Constraint-rank scatter under semantic similarity
    constraint_rank_scatter(ts_base, "semantic_sim",
        "Constraint rank vs semantic similarity (Base)",
        "Mean task-level semantic similarity",
        out_dir / "rank_vs_semantic_base.png")