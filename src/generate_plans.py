import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_benchmark(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_json(text: str) -> Dict[str, Any]:
    """
    Tries to parse JSON robustly, including accidental markdown fences.
    """
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        lines = text.splitlines()
        if lines and lines[0].lower().strip() == "json":
            lines = lines[1:]
        text = "\n".join(lines).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1:
        text = text[first:last + 1]

    return json.loads(text)


def validate_plan(plan: Dict[str, Any], planner_type: str) -> bool:
    if "steps" not in plan or not isinstance(plan["steps"], list):
        return False

    for step in plan["steps"]:
        if not isinstance(step, dict):
            return False
        if "step_id" not in step or "action" not in step:
            return False
        if planner_type == "tool" and "tool" not in step:
            return False
    return True


def call_model(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = response.choices[0].message.content
    plan = extract_json(text)
    return plan


def main() -> None:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")

    client = OpenAI(api_key=api_key)

    benchmark = load_benchmark(DATA_DIR / "benchmark_tasks.json")

    planner_type = input("Enter planner type ('base' or 'tool'): ").strip().lower()
    if planner_type not in {"base", "tool"}:
        raise ValueError("Planner type must be 'base' or 'tool'")

    system_prompt_path = (
        PROMPTS_DIR / "system_prompt_base.txt"
        if planner_type == "base"
        else PROMPTS_DIR / "system_prompt_tool.txt"
    )
    system_prompt = load_text(system_prompt_path)

    output_path = DATA_DIR / f"runs_{planner_type}.jsonl"

    records = []
    total_runs = sum(len(task["paraphrases"]) for task in benchmark)

    progress = tqdm(total=total_runs, desc=f"Generating {planner_type} plans")

    for task in benchmark:
        for idx, prompt in enumerate(task["paraphrases"], start=1):
            record = {
                "task_id": task["task_id"],
                "family": task["family"],
                "complexity": task["complexity"],
                "planner": planner_type,
                "paraphrase_id": idx,
                "prompt": prompt,
            }

            try:
                plan = call_model(client, model, system_prompt, prompt)

                if not validate_plan(plan, planner_type):
                    raise ValueError("Plan failed schema validation")

                record["plan"] = plan
                record["status"] = "ok"

            except Exception as e:
                record["plan"] = None
                record["status"] = "error"
                record["error"] = str(e)

            records.append(record)
            progress.update(1)
            time.sleep(0.3)

    progress.close()

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()