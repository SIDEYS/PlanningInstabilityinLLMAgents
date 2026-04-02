import itertools
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from rapidfuzz.distance import Levenshtein


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def get_actions(plan: dict[str, Any]) -> list[str]:
    return [step["action"] for step in plan["steps"]]


def get_action_set(plan: dict[str, Any]) -> set[str]:
    return set(get_actions(plan))


def get_action_tool_pairs(plan: dict[str, Any]) -> list[tuple[str, str]]:
    pairs = []
    for step in plan["steps"]:
        action = step.get("action", "")
        tool = step.get("tool", "none")
        pairs.append((action, tool))
    return pairs


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def sequence_similarity(a: list[str], b: list[str]) -> float:
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = Levenshtein.distance(a, b)
    return 1.0 - (dist / max_len)


def step_count_diff(a: list[str], b: list[str]) -> int:
    return abs(len(a) - len(b))


def exact_plan_match(a: dict[str, Any], b: dict[str, Any]) -> int:
    return int(a == b)


def tool_agreement(a: dict[str, Any], b: dict[str, Any]) -> Optional[float]:
    a_steps = a["steps"]
    b_steps = b["steps"]

    if "tool" not in a_steps[0] or "tool" not in b_steps[0]:
        return None

    n = min(len(a_steps), len(b_steps))
    if n == 0:
        return 1.0

    agree = 0
    for i in range(n):
        if a_steps[i].get("tool") == b_steps[i].get("tool"):
            agree += 1

    return agree / n


def main() -> None:
    planner_type = input("Enter planner type ('base' or 'tool'): ").strip().lower()
    input_path = DATA_DIR / f"runs_{planner_type}.jsonl"

    rows = read_jsonl(input_path)
    rows = [r for r in rows if r["status"] == "ok"]

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["task_id"], []).append(row)

    pairwise_rows = []

    for task_id, group in grouped.items():
        group = sorted(group, key=lambda x: x["paraphrase_id"])

        for a, b in itertools.combinations(group, 2):
            plan_a = a["plan"]
            plan_b = b["plan"]

            actions_a = get_actions(plan_a)
            actions_b = get_actions(plan_b)

            pairwise_rows.append(
                {
                    "task_id": task_id,
                    "family": a["family"],
                    "complexity": a["complexity"],
                    "planner": planner_type,
                    "p1": a["paraphrase_id"],
                    "p2": b["paraphrase_id"],
                    "action_jaccard": jaccard_similarity(set(actions_a), set(actions_b)),
                    "sequence_similarity": sequence_similarity(actions_a, actions_b),
                    "step_count_diff": step_count_diff(actions_a, actions_b),
                    "exact_plan_match": exact_plan_match(plan_a, plan_b),
                    "tool_agreement": tool_agreement(plan_a, plan_b),
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_out = DATA_DIR / f"pairwise_metrics_{planner_type}.csv"
    pairwise_df.to_csv(pairwise_out, index=False)

    summary_df = (
        pairwise_df.groupby(["task_id", "family", "complexity", "planner"], as_index=False)
        .agg(
            action_jaccard_mean=("action_jaccard", "mean"),
            sequence_similarity_mean=("sequence_similarity", "mean"),
            step_count_diff_mean=("step_count_diff", "mean"),
            exact_plan_match_rate=("exact_plan_match", "mean"),
            tool_agreement_mean=("tool_agreement", "mean"),
        )
    )

    summary_out = DATA_DIR / f"task_summary_{planner_type}.csv"
    summary_df.to_csv(summary_out, index=False)

    print(f"Saved pairwise metrics to: {pairwise_out}")
    print(f"Saved task summary to: {summary_out}")


if __name__ == "__main__":
    main()