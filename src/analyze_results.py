from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def plot_family_bar(df: pd.DataFrame, metric: str, filename: str, ylabel: str) -> None:
    family_means = df.groupby("family", as_index=False)[metric].mean()

    plt.figure(figsize=(8, 5))
    plt.bar(family_means["family"], family_means[metric])
    plt.xlabel("Task family")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} by task family")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()


def plot_task_box(pairwise_df: pd.DataFrame, metric: str, filename: str, ylabel: str) -> None:
    families = sorted(pairwise_df["family"].unique())
    data = [pairwise_df[pairwise_df["family"] == fam][metric].dropna().values for fam in families]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=families)
    plt.xlabel("Task family")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} distribution by family")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)
    plt.close()


def main() -> None:
    planner_type = input("Enter planner type ('base' or 'tool'): ").strip().lower()

    pairwise_df = pd.read_csv(DATA_DIR / f"pairwise_metrics_{planner_type}.csv")
    summary_df = pd.read_csv(DATA_DIR / f"task_summary_{planner_type}.csv")

    plot_family_bar(
        summary_df,
        metric="action_jaccard_mean",
        filename=f"action_jaccard_{planner_type}.png",
        ylabel="Mean action-set similarity",
    )

    plot_family_bar(
        summary_df,
        metric="sequence_similarity_mean",
        filename=f"sequence_similarity_{planner_type}.png",
        ylabel="Mean sequence similarity",
    )

    plot_task_box(
        pairwise_df,
        metric="step_count_diff",
        filename=f"step_count_diff_{planner_type}.png",
        ylabel="Step count difference",
    )

    if planner_type == "tool":
        plot_family_bar(
            summary_df,
            metric="tool_agreement_mean",
            filename="tool_agreement_tool.png",
            ylabel="Mean tool agreement",
        )

    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()