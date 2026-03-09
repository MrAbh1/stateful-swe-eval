"""
run_eval.py — CLI entry point for the Stateful SWE-bench evaluation.

Examples:

  # Quick smoke test (5 tasks, local eval, no Docker)
  python run_eval.py --tasks 5 --mode cold warm_live --local-eval

  # Full run on specific repos (Docker eval)
  python run_eval.py \\
    --repos django/django astropy/astropy \\
    --tasks 20 \\
    --mode cold warm_git warm_live

  # Re-run report only (skip agent, use saved results)
  python run_eval.py --report-only --run-id my-run-2026-03-08
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from task_loader import load_tasks, group_by_repo
from swe_agent import SWEAgent, AgentResult
from stateful_harness import StatefulHarness
from evaluate import save_predictions, run_swebench_evaluation, run_local_evaluation
from report import generate_report, print_report

DEFAULT_REPOS = [
    "django/django",
    "astropy/astropy",
    "sympy/sympy",
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
]

RESULTS_DIR = Path("results")


def parse_args():
    p = argparse.ArgumentParser(description="Stateful SWE-bench evaluation harness")
    p.add_argument("--repos", nargs="+", default=None,
                   help="GitHub repos to evaluate (default: 5 popular repos)")
    p.add_argument("--tasks", type=int, default=10,
                   help="Max tasks to run (default: 10)")
    p.add_argument("--mode", nargs="+", default=["cold", "warm_live"],
                   choices=["cold", "warm_git", "warm_live"],
                   help="Experiment modes to run (default: cold warm_live)")
    p.add_argument("--min-issues", type=int, default=2,
                   help="Min issues per repo for sequential experiment (default: 2)")
    p.add_argument("--local-eval", action="store_true",
                   help="Use local pytest evaluation instead of Docker (faster, less accurate)")
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel Docker evaluation workers (default: 2)")
    p.add_argument("--run-id", default=None,
                   help="Unique run identifier (default: timestamp)")
    p.add_argument("--report-only", action="store_true",
                   help="Skip agent execution, re-generate report from saved results")
    p.add_argument("--github-token", default=None,
                   help="GitHub token for git ingest (or set GITHUB_TOKEN env var)")
    return p.parse_args()


def run_condition(
    mode: str,
    tasks: list,
    run_dir: Path,
    github_token: str,
    local_eval: bool,
    workers: int,
) -> list[AgentResult]:
    """Run all tasks for a single mode (cold / warm_git / warm_live)."""
    print(f"\n{'='*60}")
    print(f"  MODE: {mode.upper()}  ({len(tasks)} tasks)")
    print(f"{'='*60}")

    harness = StatefulHarness(
        mode=mode,
        org_suffix="eval",
    )

    # Pre-warm repos for warm modes
    if mode != "cold" and github_token:
        repos = list({t.repo for t in tasks})
        for repo in repos:
            harness.prewarm_repo(repo, github_token, wait=True)

    results: list[AgentResult] = []
    by_repo = group_by_repo(tasks)

    with tempfile.TemporaryDirectory(prefix="swe-repos-") as repo_base:
        for repo, repo_tasks in by_repo.items():
            print(f"\n  Repo: {repo} ({len(repo_tasks)} tasks)")
            for i, task in enumerate(repo_tasks):
                print(f"\n  [{i+1}/{len(repo_tasks)}] {task.instance_id}")
                t0 = time.time()

                # Checkout repo at the task's base commit
                try:
                    repo_path = StatefulHarness.checkout_repo(
                        repo, task.base_commit, repo_base
                    )
                except Exception as e:
                    print(f"    Checkout failed: {e}")
                    results.append(AgentResult(instance_id=task.instance_id,
                                               patch="", error=str(e)))
                    continue

                # Run agent
                try:
                    result = harness.run_task(task, repo_path)
                    elapsed = time.time() - t0
                    cost = estimate_cost(result.input_tokens, result.output_tokens)
                    print(f"    turns={result.turns}  tools={result.tool_calls}  "
                          f"in={result.input_tokens}  out={result.output_tokens}  "
                          f"cost=${cost:.4f}  time={elapsed:.1f}s  "
                          f"patch={'yes' if result.patch else 'no'}")
                    results.append(result)
                except Exception as e:
                    print(f"    Agent error: {e}")
                    results.append(AgentResult(instance_id=task.instance_id,
                                               patch="", error=str(e)))

        # Save predictions
        preds_path = run_dir / f"predictions_{mode}.jsonl"
        save_predictions(results, preds_path)
        print(f"\n  Predictions saved → {preds_path}")

        # Evaluate patches
        print(f"\n  Running evaluation ({'local' if local_eval else 'Docker'}) ...")
        if local_eval:
            resolved = run_local_evaluation(results, tasks, repo_base)
        else:
            eval_out = run_dir / f"eval_output_{mode}"
            resolved = run_swebench_evaluation(preds_path, tasks, eval_out, workers)

        # Attach resolved flag to results
        for r in results:
            r.resolved = resolved.get(r.instance_id, False)
            status = "✓ RESOLVED" if r.resolved else "✗ failed"
            print(f"    {r.instance_id}: {status}")

    # Save full results JSON
    results_path = run_dir / f"results_{mode}.json"
    _save_results(results, results_path)
    print(f"\n  Results saved → {results_path}")

    return results


def _save_results(results: list[AgentResult], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "instance_id": r.instance_id,
            "resolved": r.resolved,
            "patch": r.patch,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "tool_calls": r.tool_calls,
            "turns": r.turns,
            "error": r.error,
        }
        for r in results
    ]
    path.write_text(json.dumps(data, indent=2))


def _load_results(path: Path) -> list[AgentResult]:
    data = json.loads(path.read_text())
    results = []
    for d in data:
        r = AgentResult(instance_id=d["instance_id"], patch=d.get("patch", ""))
        r.resolved = d.get("resolved")
        r.input_tokens = d.get("input_tokens", 0)
        r.output_tokens = d.get("output_tokens", 0)
        r.tool_calls = d.get("tool_calls", 0)
        r.turns = d.get("turns", 0)
        r.error = d.get("error", "")
        results.append(r)
    return results


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    # claude-sonnet-4-5 pricing (approximate)
    return (input_tokens / 1_000_000) * 3.0 + (output_tokens / 1_000_000) * 15.0


def main():
    args = parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}  →  {run_dir}")

    github_token = args.github_token or os.environ.get("GITHUB_TOKEN", "")
    repos = args.repos or DEFAULT_REPOS

    if not args.report_only:
        # Load tasks
        print(f"\nLoading SWE-bench tasks (repos={repos}, max={args.tasks}) ...")
        tasks = load_tasks(
            repos=repos,
            max_tasks=args.tasks,
            min_issues_per_repo=args.min_issues,
        )
        print(f"Loaded {len(tasks)} tasks across {len({t.repo for t in tasks})} repos")

        if not tasks:
            print("No tasks found. Try --repos django/django or remove --min-issues constraint.")
            sys.exit(1)

        # Save task manifest
        manifest_path = run_dir / "tasks.json"
        manifest_path.write_text(json.dumps(
            [{"instance_id": t.instance_id, "repo": t.repo} for t in tasks], indent=2
        ))

        # Run each mode
        all_results: dict[str, list[AgentResult]] = {}
        for mode in args.mode:
            mode_results = run_condition(
                mode=mode,
                tasks=tasks,
                run_dir=run_dir,
                github_token=github_token,
                local_eval=args.local_eval,
                workers=args.workers,
            )
            all_results[mode] = mode_results
    else:
        # Load saved results
        all_results = {}
        for mode in args.mode:
            path = run_dir / f"results_{mode}.json"
            if path.exists():
                all_results[mode] = _load_results(path)
            else:
                print(f"No saved results for mode={mode} at {path}")

    # Generate report
    report = generate_report(all_results)
    report_path = run_dir / "report.txt"
    report_path.write_text(report)
    print_report(report)
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
