# 📐 Planning Instability in LLM-Based Agents under Semantic Prompt Reformulation

> **COMPSCI 602 — Project Report 3**
> University of Massachusetts Amherst

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Project-orange)](.)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Phenomena](#-phenomena)
- [Benchmark](#-benchmark)
- [Metrics](#-metrics)
- [Results](#-results)
- [Setup](#-setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [References](#-references)

---

## 🔍 Overview

This project studies **planning instability** in LLM-based agents under **semantic prompt reformulation**. When multiple prompts express the same underlying goal, the agent may still produce structurally different plans — varying in action choice, action order, step count, and tool assignment.

**Central question:**
> *Do semantically equivalent prompts lead to structurally consistent plans, or do small wording changes produce meaningful planning divergence?*

We evaluate two planner configurations across a 12-task benchmark spanning four task families:

| Planner | Description |
|---|---|
| **Base planner** | Generates an ordered sequence of actions only |
| **Tool-aware planner** | Generates ordered (action, tool) pairs |

All experiments use a fixed GPT-based model at **temperature = 0** with identical system prompts and structured JSON output to enable precise metric computation.

---

## 🔬 Phenomena

The study decomposes planning instability into four targeted sub-phenomena:

| ID | Phenomenon | Question |
|---|---|---|
| **P1** | Action-set instability | Do paraphrases produce different *sets* of actions? |
| **P2** | Sequence-order instability | Are the same actions produced in different *orders*? |
| **P3** | Step-count instability | Does the *number of steps* change across paraphrases? |
| **P4** | Tool-assignment instability | Do paraphrases change which *tools* are assigned? |

### Prior Expectations

Before running experiments, the following expectations were recorded:

1. Semantically equivalent prompts should usually produce similar high-level action sets.
2. Action *order* will vary more than action *presence*.
3. Tool-aware planning will be **less** stable than action-only planning (more decisions to make).
4. Complex tasks will show larger instability than simple tasks.
5. Open-ended tasks (travel, research) will be less stable than constrained tasks (debugging).
6. Some reformulations will cause the agent to add or remove intermediate steps even when the goal is unchanged.

> **Key finding:** Expectation (3) was **not supported** — tool-aware planning dramatically *improved* stability, acting as implicit regularization.

---

## 📊 Benchmark

A 12-task benchmark spanning four task families, each with **4 semantically equivalent paraphrases** (yielding 6 pairwise comparisons per task):

### Task Families

#### ✈️ Travel Planning (T1–T3)
| ID | Task |
|---|---|
| T1 | Plan a two-day Boston trip for a student on a budget |
| T2 | Plan a weekend trip to New York City with low-cost food and transit |
| T3 | Plan a one-day Amherst-to-Boston itinerary using only public transportation |

#### 📚 Literature Review / Research Planning (T4–T6)
| ID | Task |
|---|---|
| T4 | Create a literature review plan on transformer model pruning |
| T5 | Create a literature review plan on deceptive review detection using NLP |
| T6 | Create a research plan for surveying LLM-based agents |

#### 🐛 Programming / Debugging (T7–T9)
| ID | Task |
|---|---|
| T7 | Plan how to debug a Python import error in a project environment |
| T8 | Plan how to fix a failing REST API integration in a web app |
| T9 | Plan how to debug a React page that renders blank |

#### 📅 Study / Scheduling (T10–T12)
| ID | Task |
|---|---|
| T10 | Create a one-week study plan for a machine learning exam |
| T11 | Create a 10-day preparation plan for a coding interview |
| T12 | Create a study schedule for three final exams in one week |

### Paraphrase Example

Each task has four paraphrases that share the same goal but differ in surface wording:

```
P1: "Plan a two-day Boston trip for a student on a budget."
P2: "Create a low-cost two-day itinerary for a student visiting Boston."
P3: "Help me organize a budget-friendly two-day visit to Boston as a student."
P4: "Build a cheap two-day Boston travel plan for a student traveler."
```

---

## 📏 Metrics

All metrics are computed **pairwise** across all 4 paraphrases of a task (6 pairs), then averaged at the task-family level.

| Metric | Formula / Method | Phenomenon |
|---|---|---|
| **Action Set Similarity** | Jaccard: $J(A,B) = \|A \cap B\| / \|A \cup B\|$ | P1 |
| **Sequence Similarity** | Normalized edit distance between action sequences | P2 |
| **Step Count Difference** | Absolute difference in number of steps | P3 |
| **Tool Agreement** | Exact-match proportion over aligned tools | P4 |
| **Exact Plan Match Rate** | 1 if all actions, tools, and order match exactly | Headline |

---

## 📈 Results

### Base Planner

| Task Family | Action Sim. | Seq. Sim. | Step Diff | Exact Match |
|---|---|---|---|---|
| Debugging | 0.917 | 0.939 | 0.17 | 0.67 |
| Research | 0.787 | 0.789 | 0.33 | 0.56 |
| Study | 0.866 | 0.866 | 0.78 | 0.28 |
| Travel | 0.881 | 0.900 | 0.22 | 0.56 |

### Tool-Aware Planner

| Task Family | Action Sim. | Seq. Sim. | Step Diff | Exact Match | Tool Agree. |
|---|---|---|---|---|---|
| Debugging | 1.00 | 1.00 | 0.00 | 0.50 | 0.88 |
| Research | 0.91 | 0.90 | 0.33 | 0.28 | 0.80 |
| Study | 0.93 | 0.91 | 0.00 | 0.56 | 0.87 |
| Travel | 0.94 | 0.97 | 0.00 | 0.83 | 1.00 |

### Key Observations

- 🟢 **Task family is the strongest predictor of instability.** Debugging > Travel ≈ Study > Research.
- 🟢 **Tool constraints act as implicit regularization.** The tool-aware planner outperforms the base planner across all metrics.
- 🔴 **Exact plan match is low even when individual metrics are high.** Plans can share actions and ordering but still differ in minor ways.
- 🔴 **Research tasks are the hardest to stabilize** — retaining the most action-set, sequence, and tool variability even under the tool-aware planner.
- ⚠️ **Step-count instability is most pronounced in study tasks** (base mean diff = 0.78), where plan granularity shifts with phrasing.

---

## ⚙️ Setup

### Requirements

```bash
Python >= 3.10
```

### Install dependencies

```bash
pip install openai pandas numpy
```

### API Key

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

---

## 🚀 Usage

### 1. Generate plans

Run the planner on all 12 tasks × 4 paraphrases:

```bash
python generate_plans.py --planner base --output data/plans_base.json
python generate_plans.py --planner tool --output data/plans_tool.json
```

### 2. Compute pairwise metrics

```bash
python compute_metrics.py --input data/plans_base.json --output results/pairwise_metrics_base.csv
python compute_metrics.py --input data/plans_tool.json --output results/pairwise_metrics_tool.csv
```

### 3. Aggregate by task family

```bash
python summarize.py --input results/pairwise_metrics_base.csv --output results/task_summary_base.csv
python summarize.py --input results/pairwise_metrics_tool.csv --output results/task_summary_tool.csv
```

### 4. Generate figures

```bash
python visualize.py --base results/task_summary_base.csv --tool results/task_summary_tool.csv
```

### Example plan output (JSON)

```json
{
  "task_id": "T1",
  "paraphrase": 1,
  "planner": "tool",
  "steps": [
    {"step_id": 1, "action": "search_transportation", "tool": "web_search"},
    {"step_id": 2, "action": "find_budget_hotels",    "tool": "booking_api"},
    {"step_id": 3, "action": "build_itinerary",       "tool": "calendar"}
  ]
}
```

---

## 📁 Project Structure

```
planning-instability/
│
├── data/
│   ├── tasks.json                  # 12 tasks with 4 paraphrases each
│   ├── plans_base.json             # Generated plans (base planner)
│   └── plans_tool.json             # Generated plans (tool-aware planner)
│
├── results/
│   ├── pairwise_metrics_base.csv   # Pairwise metrics, base planner
│   ├── pairwise_metrics_tool.csv   # Pairwise metrics, tool-aware planner
│   ├── task_summary_base.csv       # Family-level aggregates, base
│   └── task_summary_tool.csv       # Family-level aggregates, tool
│
├── figures/
│   ├── action_jaccard_base.png
│   ├── action_jaccard_tool.png
│   ├── sequence_similarity_base.png
│   ├── sequence_similarity_tool.png
│   ├── step_count_diff_base.png
│   ├── step_count_diff_tool.png
│   └── tool_agreement_tool.png
│
├── report/
│   └── COMPSCI602_Project_Report3.pdf
│
├── generate_plans.py               # Prompt the LLM and collect structured plans
├── compute_metrics.py              # Pairwise similarity computation
├── summarize.py                    # Aggregate metrics by task family
├── visualize.py                    # Generate all figures
└── README.md
```

---

## 📖 References

1. Yao, S., Zhao, J., Yu, D., et al. (2023). **ReAct: Synergizing Reasoning and Acting in Language Models.** *ICLR 2023.*
2. Liu, X., Yu, H., Zhang, H., et al. (2025). **AgentBench: Evaluating LLMs as Agents.** *arXiv:2308.03688.*
3. Farn, N., & Shin, R. (2023). **ToolTalk: Evaluating Tool-Usage in a Conversational Setting.** *arXiv:2311.10775.*
4. Valmeekam, K., Marquez, M., Olmo, A., et al. (2023). **PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change.** *NeurIPS 2023.*
5. Rabinovich, E., & Anaby-Tavor, A. (2025). **On the Robustness of Agentic Function Calling.** *TrustNLP @ ACL 2025.*

---

## 👤 Author

**Siddharth Bhangale**
College of Information and Computer Science
University of Massachusetts Amherst
`sbhangale@umass.edu`

---

*COMPSCI 602 — Spring 2026*
