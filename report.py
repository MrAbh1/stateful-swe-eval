"""
report.py — Generate comparison report across cold / warm_git / warm_live conditions.

Outputs an ASCII table showing:
  - Resolved rate (%)
  - Avg tokens per task (input + output)
  - Avg cost per task ($)
  - Avg turns per task
  - Delta vs cold baseline
"""

from __future__ import annotations
from swe_agent import AgentResult

# claude-sonnet-4-5 pricing per million tokens
PRICE_INPUT_PER_M  = 3.00
PRICE_OUTPUT_PER_M = 15.00


def _cost(r: AgentResult) -> float:
    return (r.input_tokens / 1e6) * PRICE_INPUT_PER_M + \
           (r.output_tokens / 1e6) * PRICE_OUTPUT_PER_M


def _stats(results: list[AgentResult]) -> dict:
    valid = [r for r in results if r.error == ""]
    if not valid:
        return {}
    resolved = [r for r in valid if r.resolved]
    return {
        "n":             len(valid),
        "resolved":      len(resolved),
        "resolve_rate":  len(resolved) / len(valid) * 100,
        "avg_input_tok": sum(r.input_tokens for r in valid) / len(valid),
        "avg_output_tok":sum(r.output_tokens for r in valid) / len(valid),
        "avg_tokens":    sum(r.input_tokens + r.output_tokens for r in valid) / len(valid),
        "avg_cost":      sum(_cost(r) for r in valid) / len(valid),
        "total_cost":    sum(_cost(r) for r in valid),
        "avg_turns":     sum(r.turns for r in valid) / len(valid),
        "avg_tools":     sum(r.tool_calls for r in valid) / len(valid),
    }


def generate_report(all_results: dict[str, list[AgentResult]]) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("  STATEFUL SWE-BENCH EVALUATION REPORT")
    lines.append("=" * 72)

    # Compute stats per mode
    stats: dict[str, dict] = {}
    for mode, results in all_results.items():
        stats[mode] = _stats(results)

    cold_stats = stats.get("cold", {})

    # Header
    col_w = 14
    header = f"{'Metric':<26}" + "".join(f"{m.upper():>{col_w}}" for m in all_results)
    if "cold" in all_results and len(all_results) > 1:
        header += f"{'Δ vs cold':>{col_w}}"
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))

    def row(label, key, fmt):
        vals = [stats[m].get(key, 0) for m in all_results]
        line = f"{label:<26}"
        for v in vals:
            line += f"{fmt.format(v):>{col_w}}"
        if "cold" in all_results and len(all_results) > 1 and cold_stats:
            modes = list(all_results.keys())
            # Delta = last non-cold mode vs cold
            non_cold = [m for m in modes if m != "cold"]
            if non_cold:
                best = stats[non_cold[-1]].get(key, 0)
                cold_val = cold_stats.get(key, 0)
                delta = best - cold_val
                sign = "+" if delta > 0 else ""
                line += f"{sign + fmt.format(delta):>{col_w}}"
        lines.append(line)

    row("Tasks run",              "n",              "{:.0f}")
    row("Resolved",               "resolved",       "{:.0f}")
    row("Resolve rate (%)",       "resolve_rate",   "{:.1f}%")
    row("Avg tokens / task",      "avg_tokens",     "{:,.0f}")
    row("  → input tokens",       "avg_input_tok",  "{:,.0f}")
    row("  → output tokens",      "avg_output_tok", "{:,.0f}")
    row("Avg cost / task ($)",    "avg_cost",       "${:.4f}")
    row("Total cost ($)",         "total_cost",     "${:.3f}")
    row("Avg turns / task",       "avg_turns",      "{:.1f}")
    row("Avg tool calls / task",  "avg_tools",      "{:.1f}")

    lines.append("")
    lines.append("=" * 72)

    # Per-repo breakdown
    lines.append("")
    lines.append("  PER-REPO BREAKDOWN")
    lines.append("-" * 72)

    # Collect all repos
    all_repos: set[str] = set()
    for results in all_results.values():
        for r in results:
            repo = r.instance_id.rsplit("__", 1)[0].replace("__", "/") if "__" in r.instance_id else "unknown"
            all_repos.add(repo)

    for repo in sorted(all_repos):
        lines.append(f"\n  {repo}")
        for mode, results in all_results.items():
            repo_results = [
                r for r in results
                if r.instance_id.startswith(repo.replace("/", "__"))
            ]
            if not repo_results:
                continue
            resolved = sum(1 for r in repo_results if r.resolved)
            avg_cost = sum(_cost(r) for r in repo_results) / len(repo_results)
            avg_tokens = sum(r.input_tokens + r.output_tokens for r in repo_results) / len(repo_results)
            lines.append(
                f"    {mode:<12} {resolved}/{len(repo_results)} resolved  "
                f"avg {avg_tokens:,.0f} tok  avg ${avg_cost:.4f}"
            )

    lines.append("")
    lines.append("=" * 72)

    # Verdict
    lines.append("")
    lines.append("  VERDICT")
    lines.append("-" * 72)

    if "cold" in stats and stats["cold"]:
        cold_rate = stats["cold"]["resolve_rate"]
        warm_modes = [m for m in stats if m != "cold" and stats[m]]
        if warm_modes:
            best_mode = max(warm_modes, key=lambda m: stats[m]["resolve_rate"])
            warm_rate = stats[best_mode]["resolve_rate"]
            delta_rate = warm_rate - cold_rate

            cold_cost = stats["cold"]["avg_cost"]
            warm_cost = stats[best_mode]["avg_cost"]
            delta_cost_pct = ((warm_cost - cold_cost) / cold_cost * 100) if cold_cost else 0

            cold_tok = stats["cold"]["avg_tokens"]
            warm_tok = stats[best_mode]["avg_tokens"]
            delta_tok_pct = ((warm_tok - cold_tok) / cold_tok * 100) if cold_tok else 0

            lines.append(
                f"  Best warm mode ({best_mode}) vs cold:\n"
                f"    Resolve rate:  {cold_rate:.1f}% → {warm_rate:.1f}%  "
                f"({'+'if delta_rate>=0 else ''}{delta_rate:.1f}pp)\n"
                f"    Avg tokens:    {cold_tok:,.0f} → {warm_tok:,.0f}  "
                f"({'+'if delta_tok_pct>=0 else ''}{delta_tok_pct:.1f}%)\n"
                f"    Avg cost:      ${cold_cost:.4f} → ${warm_cost:.4f}  "
                f"({'+'if delta_cost_pct>=0 else ''}{delta_cost_pct:.1f}%)"
            )

            if delta_rate >= 10:
                verdict = "STRONG POSITIVE SIGNAL ✓"
            elif delta_rate >= 5:
                verdict = "POSITIVE SIGNAL ✓"
            elif delta_rate >= 0:
                verdict = "MARGINAL / NO REGRESSION"
            else:
                verdict = "NEGATIVE — investigate"

            lines.append(f"\n  Verdict: {verdict}")
    else:
        lines.append("  Run cold + warm modes together to see comparison.")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def print_report(report: str):
    print("\n" + report)
