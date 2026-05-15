#!/usr/bin/env python3
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
from src.metrics import (
    jaccard, sequence_similarity, semantic_action_similarity,
    _actions_of, tool_agreement,
)
from src.statistical_tests import spearman_rank, fisher_z_test

load_dotenv()

TEMPERATURE = 0.7
N_SAMPLES   = 3

# 2 tasks per family — chosen to span the stability range seen in the pilot.
# T1/T4/T8 had perfect stability; T3/T5/T12 showed meaningful instability.
SUBSET_IDS  = {
    "travel":    ["T1", "T3"],
    "research":  ["T4", "T5"],
    "debugging": ["T7", "T8"],
    "study":     ["T10", "T12"],
}
ALL_SUBSET_IDS = [tid for ids in SUBSET_IDS.values() for tid in ids]

RUNS_BASE_PATH = config.DATA_DIR / "runs_base_temp07.jsonl"
RUNS_TOOL_PATH = config.DATA_DIR / "runs_tool_temp07.jsonl"


# ------------------------------------------------------------------ #
#  Prompt loading (inline to avoid import issues)                     #
# ------------------------------------------------------------------ #
def _load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file missing: {path}")
    return path.read_text(encoding="utf-8").strip()


# ------------------------------------------------------------------ #
#  Plan generation at temperature = 0.7                               #
# ------------------------------------------------------------------ #
def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


def _call_llm(client, system_prompt, user_prompt,
              temperature=TEMPERATURE, max_retries=3) -> dict:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=config.MODEL_NAME,
                temperature=temperature,
                max_completion_tokens=config.MAX_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed: {last_err}")


def _load_completed(path: Path) -> set:
    """Resume key: (task_id, paraphrase_idx, sample_idx)."""
    completed = set()
    if not path.exists():
        return completed
    with path.open("r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                completed.add((rec["task_id"], rec["paraphrase_idx"],
                               rec["sample_idx"]))
            except json.JSONDecodeError:
                continue
    return completed


def generate_temp_plans(benchmark: list, planner: str, output_path: Path,
                        resume: bool = True) -> None:
    """Generate N_SAMPLES plans per (task, paraphrase) at TEMPERATURE."""
    if planner == "base":
        system_prompt = _load_prompt(config.PROMPT_BASE)
    elif planner == "tool":
        system_prompt = _load_prompt(config.PROMPT_TOOL)
    else:
        raise ValueError(f"Unknown planner: {planner!r}")

    completed = _load_completed(output_path) if resume else set()
    if completed:
        print(f"[temp{TEMPERATURE}/{planner}] Resuming — "
              f"{len(completed)} records already done.")

    subset = [t for t in benchmark if t["task_id"] in ALL_SUBSET_IDS]

    todo = []
    for task in subset:
        for para_idx, paraphrase in enumerate(task["paraphrases"]):
            for s in range(N_SAMPLES):
                if (task["task_id"], para_idx, s) not in completed:
                    todo.append((task["task_id"], task["family"],
                                 para_idx, paraphrase, s))

    total = len(subset) * 4 * N_SAMPLES
    if not todo:
        print(f"[temp{TEMPERATURE}/{planner}] Nothing to do "
              f"({total} records already complete).")
        return

    print(f"[temp{TEMPERATURE}/{planner}] Generating {len(todo)}/{total} plans "
          f"-> {output_path.name}")
    client = _client()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a") as f:
        for task_id, family, para_idx, paraphrase, sample_idx in tqdm(
                todo, desc=f"temp{TEMPERATURE}-{planner}"):
            try:
                plan  = _call_llm(client, system_prompt, paraphrase)
                steps = plan.get("steps", [])
            except Exception as e:
                print(f"  [skip] {task_id} p{para_idx} s{sample_idx}: {e}")
                continue

            record = {
                "task_id":        task_id,
                "family":         family,
                "paraphrase_idx": para_idx,
                "sample_idx":     sample_idx,
                "paraphrase":     paraphrase,
                "planner":        planner,
                "temperature":    TEMPERATURE,
                "steps":          steps,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"[temp{TEMPERATURE}/{planner}] Done.")


# ------------------------------------------------------------------ #
#  Pairwise metrics — averaged across samples                          #
# ------------------------------------------------------------------ #
def _compare(steps_a, steps_b, has_tools) -> dict:
    actions_a = _actions_of(steps_a)
    actions_b = _actions_of(steps_b)
    out = {
        "action_sim":   jaccard(set(actions_a), set(actions_b)),
        "seq_sim":      sequence_similarity(actions_a, actions_b),
        "semantic_sim": semantic_action_similarity(actions_a, actions_b),
        "step_diff":    float(abs(len(steps_a) - len(steps_b))),
        "exact_match":  float(steps_a == steps_b),
    }
    if has_tools:
        out["tool_agree"] = tool_agreement(steps_a, steps_b)
    return out


def load_temp_runs(path: Path) -> dict:
    """task_id -> paraphrase_idx -> sample_idx -> record."""
    data: dict = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            (data
             .setdefault(rec["task_id"], {})
             .setdefault(rec["paraphrase_idx"], {})
             [rec["sample_idx"]]) = rec
    return data


def compute_pairwise_temp(runs_path: Path, has_tools: bool) -> pd.DataFrame:
    """Pairwise metrics per sample, then averaged across samples.

    For each (task, paraphrase_i, paraphrase_j) pair: compute metrics for
    each sample, average across N_SAMPLES samples. This gives one row per
    paraphrase pair per task, directly comparable to the main study.
    """
    by_task = load_temp_runs(runs_path)
    rows = []
    for task_id, paras in by_task.items():
        family = next(iter(next(iter(paras.values())).values()))["family"]
        para_ids = sorted(paras.keys())

        for i, j in combinations(para_ids, 2):
            sample_metrics = []
            for s in range(N_SAMPLES):
                if s not in paras.get(i, {}) or s not in paras.get(j, {}):
                    continue
                m = _compare(paras[i][s]["steps"], paras[j][s]["steps"],
                             has_tools)
                sample_metrics.append(m)
            if not sample_metrics:
                continue

            # Average across samples
            avg = {k: float(np.mean([sm[k] for sm in sample_metrics]))
                   for k in sample_metrics[0]}
            rows.append({"task_id": task_id, "family": family,
                         "p_i": i, "p_j": j, **avg})

    return pd.DataFrame(rows)


def task_summary_from_pairwise(pw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in pw.columns
                   if c not in ("task_id", "family", "p_i", "p_j")]
    return pw.groupby(["task_id", "family"])[metric_cols].mean().reset_index()


# ------------------------------------------------------------------ #
#  H2.5 dissociation test                                              #
# ------------------------------------------------------------------ #
def test_dissociation(task_sum: pd.DataFrame, label: str,
                      sim_metric: str) -> dict:
    rho_J = spearman_rank(task_sum, sim_metric,  alternative="less")
    rho_S = spearman_rank(task_sum, "step_diff", alternative="greater")
    fz    = fisher_z_test(rho_J["rho"], rho_J["n"],
                          rho_S["rho"], rho_S["n"])

    sig  = "✅" if fz["p_value"] < 0.10 else "❌"
    print(f"\n  [{label} | {sim_metric}]")
    print(f"  ρ_J ({sim_metric:<12}) = {rho_J['rho']:+.3f}  p = {rho_J['p_value']:.3f}")
    print(f"  ρ_S (step_diff)       = {rho_S['rho']:+.3f}  p = {rho_S['p_value']:.3f}")
    print(f"  Fisher z = {fz['z']:.3f},  p = {fz['p_value']:.3f}  {sig}")
    return {"rho_J": rho_J, "rho_S": rho_S, "fisher_z": fz}


# ------------------------------------------------------------------ #
#  Comparison: print temp=0 subset vs temp=0.7                         #
# ------------------------------------------------------------------ #
def load_t0_subset(has_tools: bool) -> pd.DataFrame:
    fname = ("pairwise_metrics_tool.csv" if has_tools
             else "pairwise_metrics_base.csv")
    df = pd.read_csv(config.DATA_DIR / fname)
    return df[df["task_id"].isin(ALL_SUBSET_IDS)].copy()


def print_comparison(ts_t0: pd.DataFrame, ts_t07: pd.DataFrame,
                     label: str) -> None:
    metrics = [m for m in ("action_sim", "seq_sim", "semantic_sim", "step_diff")
               if m in ts_t0.columns and m in ts_t07.columns]
    print(f"\n  {label} — family means (temp=0  /  temp=0.7)")
    print(f"  {'Family':<12}" + "".join(f"  {m[:12]:<22}" for m in metrics))
    for fam in config.FAMILY_ORDER:
        r0  = ts_t0[ts_t0["family"] == fam]
        r07 = ts_t07[ts_t07["family"] == fam]
        if r0.empty or r07.empty:
            continue
        row = f"  {fam:<12}"
        for m in metrics:
            v0  = r0[m].mean()
            v07 = r07[m].mean()
            row += f"  {v0:.3f} / {v07:.3f}        "
        print(row)


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #
def main():
    with config.BENCHMARK_PATH.open() as f:
        benchmark = json.load(f)

    print("=" * 60)
    print("PHASE B: Temperature Robustness Experiment (H2.5)")
    print("=" * 60)
    print(f"Temperature : {TEMPERATURE}")
    print(f"Samples/para: {N_SAMPLES}")
    print(f"Task subset : {ALL_SUBSET_IDS}")
    print(f"API calls   : {len(ALL_SUBSET_IDS)} × 4 × 2 × {N_SAMPLES} = "
          f"{len(ALL_SUBSET_IDS) * 4 * 2 * N_SAMPLES}")

    # ---- Generate plans ----
    generate_temp_plans(benchmark, "base", RUNS_BASE_PATH)
    generate_temp_plans(benchmark, "tool", RUNS_TOOL_PATH)

    # ---- Compute pairwise metrics ----
    print("\nComputing pairwise metrics (semantic included — ~1 min)...")
    pw_base_07 = compute_pairwise_temp(RUNS_BASE_PATH, has_tools=False)
    pw_tool_07 = compute_pairwise_temp(RUNS_TOOL_PATH, has_tools=True)

    ts_base_07 = task_summary_from_pairwise(pw_base_07)
    ts_tool_07 = task_summary_from_pairwise(pw_tool_07)

    pw_base_07.to_csv(config.DATA_DIR / "pairwise_metrics_base_temp07.csv",
                      index=False)
    pw_tool_07.to_csv(config.DATA_DIR / "pairwise_metrics_tool_temp07.csv",
                      index=False)
    ts_base_07.to_csv(config.DATA_DIR / "task_summary_base_temp07.csv",
                      index=False)
    ts_tool_07.to_csv(config.DATA_DIR / "task_summary_tool_temp07.csv",
                      index=False)

    # ---- Load temp=0 subset for comparison ----
    pw_base_t0 = load_t0_subset(has_tools=False)
    pw_tool_t0 = load_t0_subset(has_tools=True)
    ts_base_t0 = task_summary_from_pairwise(pw_base_t0)
    ts_tool_t0 = task_summary_from_pairwise(pw_tool_t0)

    # ---- Comparison tables ----
    print_comparison(ts_base_t0, ts_base_07, "Base planner")
    print_comparison(ts_tool_t0, ts_tool_07, "Tool-aware planner")

    # ---- H2.5 tests ----
    print("\n" + "=" * 60)
    print("H2.5: DISSOCIATION TEST AT TEMPERATURE = 0.7")
    print("=" * 60)
    print("\n--- Base planner, temp=0.7 ---")
    h25_base_jac = test_dissociation(ts_base_07, "Base / temp=0.7", "action_sim")
    h25_base_sem = test_dissociation(ts_base_07, "Base / temp=0.7", "semantic_sim")

    print("\n--- Tool-aware planner, temp=0.7 ---")
    h25_tool_jac = test_dissociation(ts_tool_07, "Tool / temp=0.7", "action_sim")
    h25_tool_sem = test_dissociation(ts_tool_07, "Tool / temp=0.7", "semantic_sim")

    print("\n--- Reference: same 8 tasks at temp=0 ---")
    ref_base_jac = test_dissociation(ts_base_t0, "Base / temp=0.0", "action_sim")
    ref_base_sem = test_dissociation(ts_base_t0, "Base / temp=0.0", "semantic_sim")
    ref_tool_jac = test_dissociation(ts_tool_t0, "Tool / temp=0.0", "action_sim")
    ref_tool_sem = test_dissociation(ts_tool_t0, "Tool / temp=0.0", "semantic_sim")

    # ---- Save results ----
    h25_all = {
        "temperature": TEMPERATURE,
        "n_samples":   N_SAMPLES,
        "subset_task_ids": ALL_SUBSET_IDS,
        "temp07": {
            "base": {"jaccard": h25_base_jac, "semantic": h25_base_sem},
            "tool": {"jaccard": h25_tool_jac, "semantic": h25_tool_sem},
        },
        "temp00_reference": {
            "base": {"jaccard": ref_base_jac, "semantic": ref_base_sem},
            "tool": {"jaccard": ref_tool_jac, "semantic": ref_tool_sem},
        },
    }
    with (config.DATA_DIR / "h25_results.json").open("w") as f:
        json.dump(h25_all, f, indent=2)

    # ---- Verdict ----
    print("\n" + "=" * 60)
    print("H2.5 VERDICT")
    print("=" * 60)
    p_t0  = ref_tool_sem["fisher_z"]["p_value"]
    p_t07 = h25_tool_sem["fisher_z"]["p_value"]
    print(f"\nTool / Semantic — Fisher z p-values:")
    print(f"  Full 32 tasks, temp=0.0 : p = 0.007  (Phase A)")
    print(f"  8-task subset, temp=0.0 : p = {p_t0:.3f}")
    print(f"  8-task subset, temp=0.7 : p = {p_t07:.3f}")
    if p_t07 < 0.10:
        print("\n  ✅ H2.5 SUPPORTED — dissociation persists under stochastic decoding.")
    else:
        print("\n  ❌ H2.5 NOT SUPPORTED — dissociation does not replicate at temp=0.7.")
    print(f"\nResults saved -> data/h25_results.json")


if __name__ == "__main__":
    main()