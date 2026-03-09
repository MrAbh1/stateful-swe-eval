"""
evaluate.py — Run SWE-bench evaluation on collected patches.

Takes AgentResult objects, writes them as a predictions JSONL,
then calls the swebench harness to determine pass/fail via Docker.
"""

from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from swe_agent import AgentResult
from task_loader import SWETask


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def results_to_predictions(results: list[AgentResult]) -> list[dict]:
    """Convert AgentResult list to the format swebench expects."""
    return [
        {
            "instance_id": r.instance_id,
            "model_patch": r.patch,
            "model_name_or_path": "stateful-swe-agent",
        }
        for r in results
    ]


def save_predictions(results: list[AgentResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for pred in results_to_predictions(results):
            f.write(json.dumps(pred) + "\n")
    return path


# ---------------------------------------------------------------------------
# Docker-based evaluation via swebench
# ---------------------------------------------------------------------------

def run_swebench_evaluation(
    predictions_path: str | Path,
    tasks: list[SWETask],
    output_dir: str | Path,
    max_workers: int = 4,
) -> dict[str, bool]:
    """
    Run swebench's Docker evaluation harness.

    Returns a dict mapping instance_id → resolved (True/False).
    Requires Docker to be running.
    """
    from swebench.harness.run_evaluation import main as swe_eval_main

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write task instances to a temp file (swebench needs them)
    instances = [_task_to_swebench_dict(t) for t in tasks]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")
        instances_path = f.name

    try:
        swe_eval_main(
            dataset_name="princeton-nlp/SWE-bench_Verified",
            split="test",
            instance_ids=[t.instance_id for t in tasks],
            predictions_path=str(predictions_path),
            max_workers=max_workers,
            force_rebuild=False,
            cache_level="instance",
            clean=False,
            open_file_limit=4096,
            run_id="stateful-eval",
            timeout=1800,
        )
    finally:
        os.unlink(instances_path)

    # Parse results from output dir
    return _parse_results(output_dir)


def _parse_results(output_dir: Path) -> dict[str, bool]:
    """Parse swebench evaluation output into instance_id → resolved."""
    results = {}
    results_file = output_dir / "results" / "results.json"
    if results_file.exists():
        data = json.loads(results_file.read_text())
        for instance_id, info in data.items():
            results[instance_id] = info.get("resolved", False)
    return results


def _task_to_swebench_dict(task: SWETask) -> dict:
    return {
        "instance_id": task.instance_id,
        "repo": task.repo,
        "version": task.version,
        "base_commit": task.base_commit,
        "problem_statement": task.problem_statement,
        "hints_text": task.hints_text,
        "patch": task.patch,
        "test_patch": task.test_patch,
        "FAIL_TO_PASS": json.dumps(task.fail_to_pass),
        "PASS_TO_PASS": json.dumps(task.pass_to_pass),
    }


# ---------------------------------------------------------------------------
# Lightweight local evaluation (no Docker — for smoke testing)
# ---------------------------------------------------------------------------

def _extract_test_files(test_patch: str) -> list[str]:
    """Extract b/ file paths from a unified diff test_patch."""
    import re
    if not test_patch:
        return []
    paths = []
    for line in test_patch.splitlines():
        m = re.match(r'^diff --git a/(\S+) b/(\S+)', line)
        if m:
            paths.append(m.group(2))
    return paths


def run_local_evaluation(
    results: list[AgentResult],
    tasks: list[SWETask],
    repo_base_dir: str | Path,
) -> dict[str, bool]:
    """
    Apply each patch to a local repo clone and run the fail_to_pass tests.
    Cheaper than Docker — use for quick iteration, not final benchmarks.
    """
    import subprocess

    task_map = {t.instance_id: t for t in tasks}
    resolved = {}

    for result in results:
        task = task_map.get(result.instance_id)
        if not task:
            continue

        repo_dir = Path(repo_base_dir) / task.repo.replace("/", "__")
        if not repo_dir.exists():
            resolved[result.instance_id] = False
            continue

        # Reset to base commit (discard any agent changes first)
        subprocess.run(["git", "reset", "--hard", task.base_commit],
                       cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "clean", "-fdx"],
                       cwd=repo_dir, capture_output=True)

        # Apply patch
        if not result.patch.strip():
            resolved[result.instance_id] = False
            continue

        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(result.patch)
            patch_file = f.name

        apply = subprocess.run(
            ["git", "apply", patch_file],
            cwd=repo_dir, capture_output=True
        )
        os.unlink(patch_file)

        if apply.returncode != 0:
            resolved[result.instance_id] = False
            continue

        # Apply test patch
        if task.test_patch:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
                f.write(task.test_patch)
                test_patch_file = f.name
            subprocess.run(["git", "apply", test_patch_file],
                           cwd=repo_dir, capture_output=True)
            os.unlink(test_patch_file)

        # Run fail_to_pass tests
        if task.fail_to_pass:
            # Build -k expression matching any of the test function names
            test_names = task.fail_to_pass[:10]  # cap for speed
            k_expr = " or ".join(test_names)
            # Extract test file paths from the test_patch diff headers
            test_files = _extract_test_files(task.test_patch)
            if test_files:
                test_targets = " ".join(test_files)
                cmd = f"python -m pytest {test_targets} -k '{k_expr}' -x -q 2>&1"
            else:
                cmd = f"python -m pytest -k '{k_expr}' -x -q 2>&1"
            test_run = subprocess.run(
                cmd, shell=True, cwd=repo_dir,
                capture_output=True, text=True, timeout=180
            )
            resolved[result.instance_id] = test_run.returncode == 0
        else:
            resolved[result.instance_id] = bool(result.patch.strip())

    return resolved
