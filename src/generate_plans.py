import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from . import config

load_dotenv()


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env first.")
    return OpenAI(api_key=api_key)


def _load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file missing: {path}")
    return path.read_text(encoding="utf-8").strip()


def _call_llm(client: OpenAI, system_prompt: str, user_prompt: str,
              max_retries: int = 3) -> dict:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=config.MODEL_NAME,
                temperature=config.TEMPERATURE,
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
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def _load_completed(path: Path) -> set:
    completed = set()
    if not path.exists():
        return completed
    with path.open("r") as f:
        for line in f:
            try:
                rec = json.loads(line)
                completed.add((rec["task_id"], rec["paraphrase_idx"]))
            except json.JSONDecodeError:
                continue
    return completed


def generate_all_plans(benchmark: dict, planner: str, output_path: Path,
                       resume: bool = True) -> None:
    if planner == "base":
        system_prompt = _load_prompt(config.PROMPT_BASE)
    elif planner == "tool":
        system_prompt = _load_prompt(config.PROMPT_TOOL)
    else:
        raise ValueError(f"Unknown planner: {planner!r}")

    completed = _load_completed(output_path) if resume else set()
    if completed:
        print(f"[{planner}] Resuming: {len(completed)} records already complete.")

    todo = []
    for task in benchmark:
        task_id = task["task_id"]
        for idx, paraphrase in enumerate(task["paraphrases"]):
            if (task_id, idx) not in completed:
                todo.append((task_id, task["family"], idx, paraphrase))

    if not todo:
        print(f"[{planner}] Nothing to do. All {len(completed)} records exist.")
        return

    print(f"[{planner}] Generating {len(todo)} new plans -> {output_path.name}")
    client = _client()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a") as f:
        for task_id, family, idx, paraphrase in tqdm(todo, desc=f"gen-{planner}"):
            try:
                plan = _call_llm(client, system_prompt, paraphrase)
                steps = plan.get("steps", [])
            except Exception as e:
                print(f"  [skip] {task_id} p{idx}: {e}")
                continue

            record = {
                "task_id":        task_id,
                "family":         family,
                "paraphrase_idx": idx,
                "paraphrase":     paraphrase,
                "planner":        planner,
                "steps":          steps,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

    print(f"[{planner}] Done. {output_path}")