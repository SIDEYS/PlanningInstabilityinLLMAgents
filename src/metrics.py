import json
from itertools import combinations
from pathlib import Path

import pandas as pd


# ---------------- Core distance functions ----------------
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def edit_distance(seq_a: list, seq_b: list) -> int:
    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def sequence_similarity(seq_a: list, seq_b: list) -> float:
    if not seq_a and not seq_b:
        return 1.0
    max_len = max(len(seq_a), len(seq_b))
    if max_len == 0:
        return 1.0
    return 1.0 - edit_distance(seq_a, seq_b) / max_len


def tool_agreement(plan_a: list, plan_b: list) -> float:
    n = min(len(plan_a), len(plan_b))
    if n == 0:
        return 1.0 if not plan_a and not plan_b else 0.0
    matches = sum(
        1 for i in range(n)
        if plan_a[i].get("tool") == plan_b[i].get("tool")
    )
    return matches / n


def _actions_of(steps: list) -> list:
    return [s.get("action", "") for s in steps]


def compare_plans(steps_a: list, steps_b: list, has_tools: bool) -> dict:
    actions_a = _actions_of(steps_a)
    actions_b = _actions_of(steps_b)
    out = {
        "action_sim":  jaccard(set(actions_a), set(actions_b)),
        "seq_sim":     sequence_similarity(actions_a, actions_b),
        "step_diff":   abs(len(steps_a) - len(steps_b)),
        "exact_match": int(steps_a == steps_b),
    }
    if has_tools:
        out["tool_agree"] = tool_agreement(steps_a, steps_b)
    return out


# ---------------- Loaders / aggregators ----------------
def load_runs(path: Path) -> dict:
    by_task: dict = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_task.setdefault(rec["task_id"], {})[rec["paraphrase_idx"]] = rec
    return by_task


def compute_pairwise(runs_path: Path, has_tools: bool) -> pd.DataFrame:
    by_task = load_runs(runs_path)
    rows = []
    for task_id, paras in by_task.items():
        family = next(iter(paras.values()))["family"]
        ids = sorted(paras.keys())
        for i, j in combinations(ids, 2):
            m = compare_plans(paras[i]["steps"], paras[j]["steps"], has_tools)
            rows.append({"task_id": task_id, "family": family,
                         "p_i": i, "p_j": j, **m})
    return pd.DataFrame(rows)


def task_summary(pairwise: pd.DataFrame, has_tools: bool) -> pd.DataFrame:
    cols = ["action_sim", "seq_sim", "step_diff", "exact_match"]
    if has_tools:
        cols.append("tool_agree")
    return pairwise.groupby(["task_id", "family"])[cols].mean().reset_index()


def family_summary(task_sum: pd.DataFrame, has_tools: bool) -> pd.DataFrame:
    cols = ["action_sim", "seq_sim", "step_diff", "exact_match"]
    if has_tools:
        cols.append("tool_agree")
    return task_sum.groupby("family")[cols].mean().reset_index()