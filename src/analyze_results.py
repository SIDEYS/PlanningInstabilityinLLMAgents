import json
import pandas as pd

from . import config
from .metrics import compute_pairwise, family_summary, task_summary
from .statistical_tests import run_all_tests
from .visualize import generate_all_figures


def _print_section(title):
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def main():
    runs_base = config.DATA_DIR / "runs_base.jsonl"
    runs_tool = config.DATA_DIR / "runs_tool.jsonl"

    if not runs_base.exists() or not runs_tool.exists():
        raise FileNotFoundError(
            "data/runs_base.jsonl or data/runs_tool.jsonl missing. "
            "Run scripts/run_all_experiments.py first.")

    pw_base = compute_pairwise(runs_base, has_tools=False)
    pw_tool = compute_pairwise(runs_tool, has_tools=True)
    pw_base.to_csv(config.DATA_DIR / "pairwise_metrics_base.csv", index=False)
    pw_tool.to_csv(config.DATA_DIR / "pairwise_metrics_tool.csv", index=False)

    ts_base = task_summary(pw_base, has_tools=False)
    ts_tool = task_summary(pw_tool, has_tools=True)
    ts_base.to_csv(config.DATA_DIR / "task_summary_base.csv", index=False)
    ts_tool.to_csv(config.DATA_DIR / "task_summary_tool.csv", index=False)

    fs_base = family_summary(ts_base, has_tools=False)
    fs_tool = family_summary(ts_tool, has_tools=True)

    fs_base["planner"] = "base"
    fs_tool["planner"] = "tool"
    pd.concat([fs_base, fs_tool], ignore_index=True) \
        .to_csv(config.DATA_DIR / "results_summary.csv", index=False)

    pd.options.display.float_format = "{:.3f}".format
    _print_section("Family summary (Base)")
    print(fs_base.to_string(index=False))
    _print_section("Family summary (Tool-aware)")
    print(fs_tool.to_string(index=False))

    test_results = run_all_tests(ts_base, ts_tool)
    with (config.DATA_DIR / "statistical_results.json").open("w") as f:
        json.dump(test_results, f, indent=2)

    _print_section("Hypothesis tests")
    print(json.dumps(test_results, indent=2))

    generate_all_figures(pw_base, ts_base, fs_base,
                         pw_tool, ts_tool, fs_tool,
                         config.FIGURES_DIR)
    print(f"\nFigures -> {config.FIGURES_DIR}")
    print(f"Data    -> {config.DATA_DIR}")


if __name__ == "__main__":
    main()