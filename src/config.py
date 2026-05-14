"""Project-wide configuration. Single source of truth for paths and constants."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------- Paths ----------------
DATA_DIR        = PROJECT_ROOT / "data"
FIGURES_DIR     = PROJECT_ROOT / "figures"
PROMPTS_DIR     = PROJECT_ROOT / "prompts"
BENCHMARK_PATH  = DATA_DIR / "benchmark_tasks.json"
PROMPT_BASE     = PROMPTS_DIR / "system_prompt_base.txt"
PROMPT_TOOL     = PROMPTS_DIR / "system_prompt_tool.txt"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Model ----------------
MODEL_NAME  = os.getenv("MODEL_NAME", "gpt-5.4")
TEMPERATURE = 0.0
MAX_TOKENS  = 1024

# ---------------- Tools ----------------
TOOL_VOCAB = ["search_web", "read_docs", "code_editor", "calendar", "maps", "none"]

# ---------------- Task families ----------------
CONSTRAINT_RANK = {"debugging": 1, "travel": 2, "study": 3, "research": 4}
FAMILY_ORDER    = ["debugging", "travel", "study", "research"]
FAMILY_COLORS   = {
    "debugging": "#4C78A8",
    "travel":    "#F58518",
    "study":     "#54A24B",
    "research":  "#E45756",
}

# ---------------- Statistics ----------------
N_PERMUTATIONS = 10_000
N_BOOTSTRAP    = 10_000
ALPHA          = 0.05
RANDOM_SEED    = 42