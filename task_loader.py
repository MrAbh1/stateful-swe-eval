"""
task_loader.py — Load and filter SWE-bench tasks.

Uses the verified split of SWE-bench (princeton-nlp/SWE-bench_Verified)
which is the same pool SWE-rebench draws from. Filters to repos that have
multiple issues so sequential warm-start experiments are meaningful.
"""

from __future__ import annotations
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# SWE-bench dataset — loaded lazily so the import doesn't slow down the CLI
_DATASET: list[dict] | None = None


@dataclass
class SWETask:
    instance_id: str          # e.g. "django__django-12345"
    repo: str                 # e.g. "django/django"
    version: str              # repo version tag
    problem_statement: str    # the GitHub issue text
    hints_text: str           # additional hints (may be empty)
    base_commit: str          # commit to apply patch on top of
    patch: str                # gold patch (for reference — not shown to agent)
    test_patch: str           # test patch applied before evaluation
    fail_to_pass: list[str]   # tests that must go from fail → pass
    pass_to_pass: list[str]   # tests that must remain passing


def _load_dataset(split: str = "test") -> list[dict]:
    global _DATASET
    if _DATASET is None:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split=split)
        _DATASET = list(ds)
    return _DATASET


def load_tasks(
    repos: Optional[list[str]] = None,
    max_tasks: Optional[int] = None,
    min_issues_per_repo: int = 2,
    split: str = "test",
) -> list[SWETask]:
    """
    Load SWE-bench tasks, optionally filtered by repo.

    Args:
        repos: e.g. ["django/django", "astropy/astropy"]. None = all repos.
        max_tasks: cap on total tasks returned.
        min_issues_per_repo: only include repos with at least this many issues
            (ensures warm-start experiments have enough sequential tasks).
        split: dataset split — "test" (default) or "dev".

    Returns:
        List of SWETask, sorted by repo then chronological order.
    """
    raw = _load_dataset(split)

    # Group by repo
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for item in raw:
        by_repo[item["repo"]].append(item)

    # Filter repos
    if repos:
        by_repo = {r: v for r, v in by_repo.items() if r in repos}

    # Drop repos with too few issues for sequential experiments
    by_repo = {r: v for r, v in by_repo.items() if len(v) >= min_issues_per_repo}

    # Flatten and sort: within each repo keep original order (chronological)
    all_tasks: list[dict] = []
    for repo_tasks in by_repo.values():
        all_tasks.extend(repo_tasks)

    if max_tasks:
        all_tasks = all_tasks[:max_tasks]

    return [_to_task(t) for t in all_tasks]


def _to_task(raw: dict) -> SWETask:
    return SWETask(
        instance_id=raw["instance_id"],
        repo=raw["repo"],
        version=raw.get("version", ""),
        problem_statement=raw["problem_statement"],
        hints_text=raw.get("hints_text", ""),
        base_commit=raw["base_commit"],
        patch=raw.get("patch", ""),
        test_patch=raw.get("test_patch", ""),
        fail_to_pass=json.loads(raw.get("FAIL_TO_PASS", "[]")),
        pass_to_pass=json.loads(raw.get("PASS_TO_PASS", "[]")),
    )


def group_by_repo(tasks: list[SWETask]) -> dict[str, list[SWETask]]:
    """Group tasks by repo — useful for sequential warm-start experiments."""
    grouped: dict[str, list[SWETask]] = defaultdict(list)
    for t in tasks:
        grouped[t.repo].append(t)
    return dict(grouped)


def repo_github_url(repo: str) -> str:
    return f"https://github.com/{repo}"
