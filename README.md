# stateful-swe-eval

Evaluates whether [Stateful](https://stateful.dev) memory improves AI agent performance on real-world software engineering tasks from [SWE-bench Verified](https://swe-bench.github.io/).

## What it measures

Runs the same SWE-bench tasks in three conditions and compares:

| Mode | Description |
|---|---|
| `cold` | Agent has no prior context — empty Stateful namespace |
| `warm_git` | Stateful pre-warmed via git ingest of the repo's PR history |
| `warm_live` | Git ingest + sequential live sessions (each task feeds the next) |

**Metrics:** resolved rate (%), tokens per task, cost per task, turns per task.

## Setup

```bash
# 1. Clone
git clone https://github.com/MrAbh1/stateful-swe-eval
cd stateful-swe-eval

# 2. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Fill in ANTHROPIC_API_KEY, STATEFUL_API_KEY, STATEFUL_BASE_URL, GITHUB_TOKEN

# 4. Make sure Docker is running (for full evaluation)
docker info
```

## Run

```bash
# Quick smoke test — 5 tasks, local eval (no Docker)
python run_eval.py --tasks 5 --mode cold warm_live --local-eval

# Full run on specific repos with Docker evaluation
python run_eval.py \
  --repos django/django astropy/astropy \
  --tasks 20 \
  --mode cold warm_git warm_live

# Re-generate report from saved results
python run_eval.py --report-only --run-id 2026-03-08_14-00-00
```

## How it works

```
task_loader.py      — loads SWE-bench Verified tasks, groups by repo
swe_agent.py        — Claude agent with bash + view_file + edit_file tools
stateful_harness.py — wraps agent with Stateful session/start + session/end
evaluate.py         — applies patches and runs SWE-bench Docker evaluation
run_eval.py         — CLI orchestrator (cold vs warm conditions)
report.py           — ASCII comparison table + per-repo breakdown + verdict
sdk/client.py       — Stateful HTTP SDK (zero-dependency stdlib)
results/            — per-run predictions, results JSON, report
```

### The key experiment

SWE-rebench tasks come from repos with multiple real issues. The most compelling signal is **sequential**: run issue #1 cold, Stateful captures that session, run issue #2 with the memory of issue #1 injected. Later tasks should resolve at a higher rate and lower token cost because the agent skips dead ends it already tried.

## Results

Results are saved per run under `results/<run-id>/`:
- `tasks.json` — task manifest
- `predictions_<mode>.jsonl` — raw patches
- `results_<mode>.json` — per-task resolved + token stats
- `report.txt` — comparison table
