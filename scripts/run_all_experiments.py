import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.analyze_results import main as run_analysis
from src.generate_plans import generate_all_plans


def main():
    with config.BENCHMARK_PATH.open("r") as f:
        benchmark = json.load(f)

    n_paras = len(benchmark[0]["paraphrases"]) if benchmark else 0
    print(f"Loaded benchmark: {len(benchmark)} tasks x {n_paras} paraphrases.")

    generate_all_plans(benchmark, "base",
                       config.DATA_DIR / "runs_base.jsonl", resume=True)
    generate_all_plans(benchmark, "tool",
                       config.DATA_DIR / "runs_tool.jsonl", resume=True)

    run_analysis()


if __name__ == "__main__":
    main()