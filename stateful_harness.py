"""
stateful_harness.py — Wraps SWEAgent with the Stateful session lifecycle.

Three experiment modes:
  cold       — empty Stateful namespace, no context injected
  warm_git   — repo pre-warmed via git ingest, context injected at session start
  warm_live  — git ingest + sequential live sessions (each task feeds the next)

Usage:
    harness = StatefulHarness(mode="warm_live")
    harness.prewarm_repo("django/django", github_token="ghp_...")
    result = harness.run_task(task, repo_path="/tmp/django")
"""

from __future__ import annotations
import os
import time
import subprocess
from pathlib import Path
from typing import Optional

from sdk.client import StatefulClient, StatefulError
from swe_agent import SWEAgent, AgentResult
from task_loader import SWETask, repo_github_url

ORG_ID_PREFIX = "org-swe"


class StatefulHarness:
    def __init__(
        self,
        mode: str = "warm_live",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        org_suffix: str = "eval",
    ):
        """
        Args:
            mode: "cold" | "warm_git" | "warm_live"
            org_suffix: appended to org ID so cold/warm runs are isolated.
                cold      → org-swe-eval-cold
                warm_git  → org-swe-eval-warm-git
                warm_live → org-swe-eval-warm-live
        """
        assert mode in ("cold", "warm_git", "warm_live"), f"Unknown mode: {mode}"
        self.mode = mode
        self.org_id = f"{ORG_ID_PREFIX}-{org_suffix}-{mode.replace('_', '-')}"

        self.client = StatefulClient(
            base_url=base_url or os.environ.get("STATEFUL_BASE_URL", "http://localhost:8000"),
            api_key=api_key or os.environ.get("STATEFUL_API_KEY", ""),
        )

        # Track ingest jobs so we can wait for them
        self._ingest_jobs: dict[str, str] = {}  # repo → job_id

    # ------------------------------------------------------------------
    # Pre-warming
    # ------------------------------------------------------------------

    def prewarm_repo(self, repo: str, github_token: str, wait: bool = True) -> str | None:
        """
        Kick off git ingest for a repo to bootstrap memory from its PR history.
        Returns job_id. Only runs for warm_git and warm_live modes.
        """
        if self.mode == "cold":
            return None

        if repo in self._ingest_jobs:
            return self._ingest_jobs[repo]

        print(f"  [stateful] ingesting {repo} ...")
        try:
            resp = self.client._post("/ingest/repo", {
                "org_id": self.org_id,
                "repo_url": repo_github_url(repo),
                "github_token": github_token,
                "team_id": f"team-{repo.replace('/', '-')}",
            })
            job_id = resp.get("job_id", "")
            self._ingest_jobs[repo] = job_id

            if wait and job_id:
                self._wait_for_ingest(job_id)

            return job_id
        except StatefulError as e:
            print(f"  [stateful] ingest error: {e}")
            return None

    def _wait_for_ingest(self, job_id: str, timeout: int = 300, poll: int = 5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                status = self.client._get(f"/ingest/repo/{job_id}?org_id={self.org_id}")
                state = status.get("status", "")
                print(f"  [stateful] ingest {job_id}: {state} "
                      f"({status.get('sessions_created', 0)} sessions)")
                if state in ("completed", "failed"):
                    return
            except StatefulError:
                pass
            time.sleep(poll)
        print(f"  [stateful] ingest {job_id}: timed out after {timeout}s")

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def run_task(self, task: SWETask, repo_path: str | Path) -> AgentResult:
        """
        Run the agent on a single task with the appropriate Stateful wrapping.
        """
        context_string = ""
        session_id = None
        injected_ids: list[str] = []

        # --- Session start ---
        if self.mode != "cold":
            try:
                start = self.client.start_session(
                    org_id=self.org_id,
                    user_id="swe-agent",
                    team_id=f"team-{task.repo.replace('/', '-')}",
                    ticket_name=f"{task.instance_id}: {task.problem_statement[:120]}",
                    pipeline_names=[task.repo],
                    git_repos=[repo_github_url(task.repo)],
                )
                session_id = start.get("session_id")
                context_string = start.get("context_string", "")
                injected_ids = [
                    s.get("session_id", s.get("id", ""))
                    for s in start.get("sessions", [])
                    if isinstance(s, dict)
                ]
                if context_string:
                    print(f"  [stateful] warm context loaded ({len(injected_ids)} prior sessions)")
                else:
                    print(f"  [stateful] no prior context (first task for this repo)")
            except StatefulError as e:
                print(f"  [stateful] session/start error: {e}")

        # --- Run agent ---
        agent = SWEAgent(repo_path)
        result = agent.run(task, context_string=context_string)

        # --- Buffer events ---
        if session_id and self.mode in ("warm_live",):
            for ev in result.events:
                try:
                    self.client.event(
                        session_id=session_id,
                        org_id=self.org_id,
                        user_id="swe-agent",
                        event_type="message",
                        content=ev["content"][:4000],
                    )
                except StatefulError:
                    pass

        # --- Session end ---
        if session_id and self.mode in ("warm_live",):
            try:
                end = self.client.end_session(
                    session_id=session_id,
                    org_id=self.org_id,
                    user_id="swe-agent",
                    team_id=f"team-{task.repo.replace('/', '-')}",
                    ticket_name=f"{task.instance_id}: {task.problem_statement[:120]}",
                    pipeline_names=[task.repo],
                    git_repos=[repo_github_url(task.repo)],
                    context_session_ids=injected_ids,
                )
                print(f"  [stateful] session ended — "
                      f"{end.get('key_terms_indexed', 0)} terms indexed")
            except StatefulError as e:
                print(f"  [stateful] session/end error: {e}")

        return result

    # ------------------------------------------------------------------
    # Repo checkout helpers
    # ------------------------------------------------------------------

    @staticmethod
    def checkout_repo(repo: str, commit: str, target_dir: str | Path) -> Path:
        """
        Clone repo at a specific commit into target_dir.
        Uses a blobless clone (--filter=blob:none) for speed — fetches all
        history metadata but downloads file blobs on demand.
        Returns path to the repo root.
        """
        target = Path(target_dir) / repo.replace("/", "__")
        url = f"https://github.com/{repo}.git"

        if target.exists():
            # Wipe any leftover changes from the previous task BEFORE fetching
            subprocess.run(["git", "reset", "--hard", "HEAD"],
                           cwd=target, capture_output=True)
            subprocess.run(["git", "clean", "-fdx"],
                           cwd=target, capture_output=True)
            subprocess.run(
                ["git", "fetch", "--filter=blob:none", "origin"],
                cwd=target, capture_output=True
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--filter=blob:none", url, str(target)],
                check=True, capture_output=True
            )

        # Checkout the exact commit then do a final hard reset + clean
        subprocess.run(
            ["git", "checkout", commit],
            cwd=target, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "reset", "--hard", commit],
            cwd=target, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "clean", "-fdx"],
            cwd=target, check=True, capture_output=True
        )
        return target
