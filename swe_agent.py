"""
swe_agent.py — Claude agent that attempts to solve a SWE-bench task.

The agent has three tools:
  - bash        : run shell commands inside the task repo
  - view_file   : read a file (with optional line range)
  - edit_file   : apply a targeted string replacement to a file

Token usage and tool call counts are tracked per run and returned alongside
the generated patch.
"""

from __future__ import annotations
import os
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import anthropic

from task_loader import SWETask

MODEL = "claude-sonnet-4-5"
MAX_TURNS = 30
MAX_TOKENS_PER_TURN = 4096

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "bash",
        "description": (
            "Run a bash command in the repository root. "
            "Use for: listing files, running tests, grepping, git commands. "
            "Avoid interactive commands. Output is truncated at 8000 chars."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."}
            },
            "required": ["command"],
        },
    },
    {
        "name": "view_file",
        "description": "Read a file from the repository, optionally limiting to a line range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path from repo root."},
                "start_line": {"type": "integer", "description": "First line to read (1-indexed)."},
                "end_line": {"type": "integer", "description": "Last line to read (inclusive)."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file with new content. "
            "old_str must match the file exactly (including whitespace). "
            "Use view_file first to confirm the exact text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path from repo root."},
                "old_str": {"type": "string", "description": "Exact string to replace."},
                "new_str": {"type": "string", "description": "Replacement string."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    instance_id: str
    patch: str                      # git diff output
    resolved: Optional[bool] = None # filled in by evaluate.py
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    turns: int = 0
    error: str = ""
    events: list[dict] = field(default_factory=list)  # for Stateful session/event


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SWEAgent:
    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path)
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def run(self, task: SWETask, context_string: str = "") -> AgentResult:
        """
        Attempt to solve task. Returns AgentResult with patch + token stats.
        context_string is injected into the system prompt by the Stateful harness.
        """
        system = self._build_system_prompt(task, context_string)
        user_message = self._build_user_message(task)

        messages = [{"role": "user", "content": user_message}]
        result = AgentResult(instance_id=task.instance_id)

        # Track events for Stateful
        result.events.append({"role": "user_message", "content": user_message})

        for turn in range(MAX_TURNS):
            result.turns = turn + 1

            response = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS_PER_TURN,
                system=system,
                tools=TOOLS,
                messages=messages,
            )

            result.input_tokens += response.usage.input_tokens
            result.output_tokens += response.usage.output_tokens

            # Accumulate assistant message
            messages.append({"role": "assistant", "content": response.content})

            # Build agent response text for Stateful event
            agent_text = " ".join(
                b.text for b in response.content if hasattr(b, "text")
            )
            if agent_text:
                result.events.append({"role": "agent_response", "content": agent_text})

            # Check stop condition
            if response.stop_reason == "end_turn":
                break

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result.tool_calls += 1
                tool_output = self._dispatch_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_output[:8000],
                })

            if not tool_results:
                break

            messages.append({"role": "user", "content": tool_results})

        # Generate patch
        result.patch = self._get_patch()
        return result

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, name: str, inputs: dict) -> str:
        if name == "bash":
            return self._bash(inputs["command"])
        elif name == "view_file":
            return self._view_file(
                inputs["path"],
                inputs.get("start_line"),
                inputs.get("end_line"),
            )
        elif name == "edit_file":
            return self._edit_file(inputs["path"], inputs["old_str"], inputs["new_str"])
        return f"Unknown tool: {name}"

    def _bash(self, command: str) -> str:
        try:
            out = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=60,
            )
            output = (out.stdout + out.stderr).strip()
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 60s"
        except Exception as e:
            return f"Error: {e}"

    def _view_file(self, path: str, start: Optional[int], end: Optional[int]) -> str:
        full = self.repo_path / path
        if not full.exists():
            return f"Error: {path} not found"
        lines = full.read_text(errors="replace").splitlines()
        if start or end:
            s = (start or 1) - 1
            e = end or len(lines)
            lines = lines[s:e]
        numbered = "\n".join(f"{i+1:4d}  {l}" for i, l in enumerate(lines))
        return numbered[:8000]

    def _edit_file(self, path: str, old_str: str, new_str: str) -> str:
        full = self.repo_path / path
        if not full.exists():
            return f"Error: {path} not found"
        content = full.read_text(errors="replace")
        if old_str not in content:
            return (
                f"Error: old_str not found verbatim in {path}. "
                "Use view_file to check exact whitespace/indentation."
            )
        full.write_text(content.replace(old_str, new_str, 1))
        return f"Edited {path} successfully."

    def _get_patch(self) -> str:
        try:
            result = subprocess.run(
                "git diff",
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )
            return result.stdout.strip()
        except Exception as e:
            return f"# Error generating patch: {e}"

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_system_prompt(self, task: SWETask, context_string: str) -> str:
        parts = [
            textwrap.dedent(f"""\
                You are an expert software engineer working on the {task.repo} repository.
                Your goal is to fix the issue described by the user by editing the source code.

                Rules:
                - Only modify files in the repository (do not install packages).
                - After making changes, run the relevant tests to verify your fix.
                - When you are confident the fix is correct, stop (do not say "done" — just stop calling tools).
                - Be concise. Avoid unnecessary exploration.
            """)
        ]
        if context_string:
            parts.append(
                "--- PRIOR SESSION MEMORY (from Stateful) ---\n"
                + context_string
                + "\n--- END PRIOR SESSION MEMORY ---"
            )
        return "\n\n".join(parts)

    def _build_user_message(self, task: SWETask) -> str:
        msg = f"Fix the following issue in {task.repo}:\n\n{task.problem_statement}"
        if task.hints_text:
            msg += f"\n\nHints:\n{task.hints_text}"
        return msg
