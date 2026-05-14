import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src import config
from src.generate_plans import generate_all_plans

with config.BENCHMARK_PATH.open() as f:
    benchmark = json.load(f)

test_subset = [t for t in benchmark if t["task_id"] == "T1"]
generate_all_plans(test_subset, "base",
                   config.DATA_DIR / "smoke_test_runs.jsonl", resume=False)

print("\nSmoke test output:")
with (config.DATA_DIR / "smoke_test_runs.jsonl").open() as f:
    for line in f:
        rec = json.loads(line)
        print(f"  {rec['task_id']} p{rec['paraphrase_idx']}: {len(rec['steps'])} steps")