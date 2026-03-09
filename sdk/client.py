"""
StatefulClient — thin Python SDK for the Stateful memory API.

Zero dependencies beyond the stdlib. Drop this file anywhere and it works.

Usage:
    from sdk.client import StatefulClient

    client = StatefulClient(base_url="http://localhost:8000")

    # Start a session — get a session_id and prior context
    start = client.start_session(
        org_id="org-001",
        user_id="user-alice",
        team_id="team-payments",
        ticket_name="Fix JWT refresh 401s",
        packages=["jwt", "redis"],
    )
    session_id = start["session_id"]
    context_string = start["context_string"]   # inject into Claude system prompt

    # Buffer events during the session
    client.event(session_id, "org-001", "user-alice", "message", "user said: ...")
    client.event(session_id, "org-001", "user-alice", "message", "agent said: ...")

    # Finalize — extracts structure, upserts to Pinecone
    result = client.end_session(
        session_id=session_id,
        org_id="org-001",
        user_id="user-alice",
        team_id="team-payments",
        ticket_name="Fix JWT refresh 401s",
        packages=["jwt", "redis"],
    )
    print(result["digest"])
"""

import json
import urllib.error
import urllib.request
from typing import Optional


class StatefulError(Exception):
    """Raised when the Stateful server returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class StatefulClient:
    """
    Thin wrapper over the Stateful HTTP API.

    All methods return the parsed JSON response dict.
    Raises StatefulError on non-2xx responses.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(
        self,
        org_id: str,
        user_id: str,
        team_id: str,
        ticket_name: str = "",
        ticket_id: str = "",
        pipeline_names: list = None,
        git_repos: list = None,
        packages: list = None,
        top_n: int = 5,
    ) -> dict:
        """
        Start a new agent session.

        Generates a session_id and queries Stateful for prior context
        relevant to the current task. Returns both so the caller can
        inject context_string into their agent's system prompt and pass
        session_id to subsequent event() and end_session() calls.

        Returns:
            {
                "session_id": str,
                "context_string": str,      # inject into system prompt
                "prior_sessions_used": int,
                "sessions": [...],          # full session records
                "canonical_stores": [...],  # established patterns
            }
        """
        return self._post("/session/start", {
            "org_id": org_id,
            "user_id": user_id,
            "team_id": team_id,
            "ticket_name": ticket_name,
            "ticket_id": ticket_id,
            "pipeline_names": pipeline_names or [],
            "git_repos": git_repos or [],
            "packages": packages or [],
            "top_n": top_n,
        })

    def event(
        self,
        session_id: str,
        org_id: str,
        user_id: str,
        event_type: str,
        content: str,
    ) -> dict:
        """
        Buffer a raw session event.

        Call this for every meaningful turn in the session:
        - event_type="message" for user/agent messages
        - event_type="tool_call" for tool invocations
        - event_type="tool_result" for tool outputs

        Returns:
            {"status": "ok", "event_id": str}
        """
        return self._post("/session/event", {
            "session_id": session_id,
            "org_id": org_id,
            "user_id": user_id,
            "event_type": event_type,
            "content": content,
        })

    def end_session(
        self,
        session_id: str,
        org_id: str,
        user_id: str,
        team_id: str,
        ticket_id: str = "",
        ticket_name: str = "",
        pipeline_names: list = None,
        git_repos: list = None,
        packages: list = None,
        context_session_ids: list = None,
    ) -> dict:
        """
        Finalize a session.

        Claude extracts structured fields (digest, solutions, failure_causes,
        key_terms) from the buffered events, writes them to SQLite, and
        upserts 5 named vectors to Pinecone for future retrieval.

        context_session_ids: session IDs that were injected as context at session
            start (returned by /session/start). Passing these records an implicit
            citation — those sessions contributed to this session's warm start.

        Returns:
            {
                "status": "finalized",
                "session_id": str,
                "digest": str,
                "extracted": {...},
                "key_terms_indexed": int,
                "references_tracked": int,
            }
        """
        return self._post("/session/end", {
            "session_id": session_id,
            "org_id": org_id,
            "user_id": user_id,
            "team_id": team_id,
            "ticket_id": ticket_id,
            "ticket_name": ticket_name,
            "pipeline_names": pipeline_names or [],
            "git_repos": git_repos or [],
            "packages": packages or [],
            "context_session_ids": context_session_ids or [],
        })

    def get_session(self, session_id: str) -> dict:
        """Fetch a session record by ID."""
        return self._get(f"/session/{session_id}")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        org_id: str,
        user_id: str,
        team_id: str,
        ticket_name: str = "",
        pipeline_names: list = None,
        git_repos: list = None,
        packages: list = None,
        top_n: int = 5,
    ) -> dict:
        """
        Multi-signal search for relevant prior sessions.

        Use this when you want context without starting a formal session
        (e.g. answering a quick question, building a search UI).

        Returns:
            {
                "sessions": [...],
                "canonical_stores": [...],
                "context_string": str,
                "total_results": int,
            }
        """
        return self._post("/query/", {
            "org_id": org_id,
            "user_id": user_id,
            "team_id": team_id,
            "ticket_name": ticket_name,
            "pipeline_names": pipeline_names or [],
            "git_repos": git_repos or [],
            "packages": packages or [],
            "top_n": top_n,
        })

    # ------------------------------------------------------------------
    # Git events
    # ------------------------------------------------------------------

    def git_event(
        self,
        org_id: str,
        user_id: str,
        repo: str,
        event_type: str,
        files_changed: list = None,
        pipelines: list = None,
        packages: list = None,
    ) -> dict:
        """
        Ingest a git event (push, pr_merged, code_review).

        Triggers a staleness check — if any active sessions touch the
        same files/pipelines/packages, they are marked deprecated.

        Returns:
            {"status": "ok", "deprecated_sessions": int}
        """
        return self._post("/git/event", {
            "org_id": org_id,
            "user_id": user_id,
            "repo": repo,
            "event_type": event_type,
            "files_changed": files_changed or [],
            "pipelines": pipelines or [],
            "packages": packages or [],
        })

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """Check that the Stateful server is reachable."""
        return self._get("/health")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass  # nothing to clean up — stateless HTTP client

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url=self.base_url + path,
            data=data,
            method="POST",
            headers=self._headers(),
        )
        return self._send(req)

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(
            url=self.base_url + path,
            method="GET",
            headers=self._headers(),
        )
        return self._send(req)

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _send(self, req: urllib.request.Request) -> dict:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode("utf-8")).get("detail", str(e))
            except Exception:
                detail = str(e)
            raise StatefulError(e.code, detail) from e
        except urllib.error.URLError as e:
            raise StatefulError(0, f"Connection failed: {e.reason}") from e
