"""AI multi-agent debate pipeline for XAUUSD regime classification.

Adapted from alphalens-v2's 7-agent debate architecture (pipeline.py).

Architecture:
  1. Format RegimeContext features into domain-specific context strings
  2. Run 4 Analysts in sequence (fast brain, <=512 max_tokens each)
  3. Run Bull + Bear Researchers for n_debate_rounds (slow brain)
  4. Run Judge Agent with debate transcript (slow brain)
  5. Parse structured output into DebateResult

LLM backends (auto-detected):
  1. Claude Code CLI (``claude -p``) — preferred, uses Opus 4.6 / Sonnet 4.6
  2. Anthropic API — fallback when CLI is unavailable
  3. Template — returns TRANSITION regime with no LLM cost

Context hygiene rules:
  - Analysts receive ONLY their domain's features
  - Researchers receive all 4 analyst views + full feature summary
  - Judge receives debate transcript + compact feature summary (not raw analyst views)
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Literal

from smc.ai.debate.prompts import (
    ANALYST_SYSTEM_PROMPTS,
    BEAR_RESEARCHER_SYSTEM,
    BULL_RESEARCHER_SYSTEM,
    DOMAIN_FEATURES,
    JUDGE_SYSTEM,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM backend selection — Anthropic API (preferred) > Claude CLI > fallback
# ---------------------------------------------------------------------------

_FAST_MODEL_CLI = "sonnet"  # Analysts
_SLOW_MODEL_CLI = "opus"    # Researchers + Judge

BackendType = Literal["anthropic", "claude_cli", "auto"]


def _has_claude_cli() -> bool:
    """Check if ``claude`` CLI is available on PATH."""
    return shutil.which("claude") is not None


def _anthropic_chat(
    client: Any,
    system: str,
    user: str,
    max_tokens: int,
    model: str,
) -> tuple[str, int, float]:
    """Call Anthropic API and return (content, total_tokens, cost_usd).

    Uses prompt caching for system prompts (they're reused across all calls
    in a single debate run).
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    content = response.content[0].text if response.content else ""
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    total_tokens = input_tokens + output_tokens

    # Estimate cost (Sonnet: $3/$15 per MTok, Opus: $15/$75 per MTok)
    if "sonnet" in model.lower():
        cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000
    else:
        cost = (input_tokens * 15.0 + output_tokens * 75.0) / 1_000_000

    return content, total_tokens, cost


def _claude_cli_chat(
    system: str,
    user: str,
    max_tokens: int,
    model: str = _SLOW_MODEL_CLI,
) -> tuple[str, int, float]:
    """Call Claude Code pipe mode and return (content, 0_tokens, 0_cost).

    Token count and cost are unavailable from pipe mode.
    """
    prompt = f"{system}\n\n---\n\n{user}"
    claude_path = shutil.which("claude") or shutil.which("claude.cmd") or "claude"
    cmd = [claude_path, "-p", "--model", model]
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("claude -p timed out after 120s")
    except FileNotFoundError:
        raise RuntimeError("claude CLI not found on PATH")

    if result.returncode != 0:
        stderr = result.stderr[:300] if result.stderr else "(no stderr)"
        raise RuntimeError(f"claude -p exit {result.returncode}: {stderr}")

    return result.stdout.strip(), 0, 0.0


def _chat(
    system: str,
    user: str,
    max_tokens: int,
    *,
    brain: Literal["fast", "slow"] = "slow",
    backend: BackendType = "auto",
    anthropic_client: Any = None,
    anthropic_model: str = "claude-sonnet-4-6",
) -> tuple[str, int, float]:
    """Unified chat dispatch — Claude CLI (preferred) > Anthropic API > error.

    brain="fast" -> Sonnet 4.6 (Analysts)
    brain="slow" -> Opus 4.6 (Researchers + Judge)
    """
    # Determine model for each backend
    if brain == "fast":
        cli_model = _FAST_MODEL_CLI
        api_model = anthropic_model if "sonnet" in anthropic_model.lower() else "claude-sonnet-4-6"
    else:
        cli_model = _SLOW_MODEL_CLI
        # For slow brain, always use Opus regardless of config
        api_model = "claude-opus-4-6"

    # Try Claude CLI first (preferred — matches alphalens-v2 + user confirmation)
    if backend in ("claude_cli", "auto") and _has_claude_cli():
        return _claude_cli_chat(system, user, max_tokens, cli_model)

    # Fallback to Anthropic API
    if backend in ("anthropic", "auto") and anthropic_client is not None:
        return _anthropic_chat(anthropic_client, system, user, max_tokens, api_model)

    raise RuntimeError(
        "No LLM backend available. Install Claude Code CLI or set "
        "SMC_ANTHROPIC_API_KEY."
    )


# ---------------------------------------------------------------------------
# Feature context formatters
# ---------------------------------------------------------------------------


def _format_domain_context(
    features: dict[str, Any],
    domain: str,
) -> str:
    """Build the user-message context for a single Analyst agent.

    Only includes features that belong to this domain (least-privilege).
    """
    domain_keys = DOMAIN_FEATURES.get(domain, [])
    lines = [f"=== {domain.upper()} DOMAIN FEATURES ==="]

    available = 0
    for key in domain_keys:
        value = features.get(key)
        if value is not None:
            lines.append(f"  {key}: {value}")
            available += 1
        else:
            lines.append(f"  {key}: (unavailable)")

    if available == 0:
        lines.append("  (no features available for this domain)")

    return "\n".join(lines)


def _format_full_feature_summary(features: dict[str, Any]) -> str:
    """Compact feature summary for Researchers and Judge."""
    lines = ["=== FULL FEATURE SUMMARY ==="]

    # Group by domain
    for domain, keys in DOMAIN_FEATURES.items():
        domain_lines = []
        for key in keys:
            value = features.get(key)
            if value is not None:
                domain_lines.append(f"  {key}: {value}")
        if domain_lines:
            lines.append(f"\n[{domain.upper()}]")
            lines.extend(domain_lines)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON parsers with fallback
# ---------------------------------------------------------------------------


def _parse_analyst_json(content: str, domain: str) -> dict[str, Any]:
    """Parse analyst JSON output, with fallback on parse error."""
    try:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data: dict[str, Any] = json.loads(text.strip())
        return {
            "domain": domain,
            "regime_vote": str(data.get("regime_vote", "TRANSITION")),
            "confidence": float(data.get("confidence", 0.5)),
            "reasoning": str(data.get("reasoning", "(no reasoning)")),
        }
    except Exception as exc:
        logger.warning(
            "Failed to parse %s analyst JSON: %s — raw: %s",
            domain, exc, content[:200],
        )
        return {
            "domain": domain,
            "regime_vote": "TRANSITION",
            "confidence": 0.3,
            "reasoning": f"(parse error) {content[:150]}",
        }


def _parse_judge_json(content: str) -> dict[str, Any]:
    """Parse Judge JSON output, with fallback."""
    try:
        text = content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data: dict[str, Any] = json.loads(text.strip())
        return {
            "regime": str(data.get("regime", "TRANSITION")),
            "confidence": float(data.get("confidence", 0.5)),
            "decisive_factors": tuple(data.get("decisive_factors", [])),
            "reasoning": str(data.get("reasoning", "(no reasoning)")),
        }
    except Exception as exc:
        logger.warning(
            "Failed to parse Judge JSON: %s — raw: %s",
            exc, content[:200],
        )
        return {
            "regime": "TRANSITION",
            "confidence": 0.3,
            "decisive_factors": (),
            "reasoning": f"(parse error) {content[:150]}",
        }


# ---------------------------------------------------------------------------
# Debate result type (plain dict — will be converted to frozen models by caller)
# ---------------------------------------------------------------------------


def _fallback_result(elapsed: float) -> dict[str, Any]:
    """Return a fallback result when no LLM is available."""
    return {
        "regime": "TRANSITION",
        "confidence": 0.3,
        "decisive_factors": (),
        "reasoning": "LLM not configured — defaulting to TRANSITION regime.",
        "analyst_views": (),
        "debate_rounds": (),
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_regime_debate(
    features: dict[str, Any],
    *,
    backend: BackendType = "auto",
    anthropic_client: Any = None,
    anthropic_model: str = "claude-sonnet-4-6",
    n_debate_rounds: int = 2,
    fast_max_tokens: int = 512,
    slow_max_tokens: int = 1024,
    on_step: Any | None = None,
) -> dict[str, Any]:
    """Run the full 7-agent debate pipeline for regime classification.

    Parameters
    ----------
    features:
        Flat dictionary of all regime context features. Keys match
        ``DOMAIN_FEATURES`` in prompts.py. Values may be None if
        unavailable.
    backend:
        LLM backend selection. "auto" tries Anthropic API first,
        then Claude CLI, then returns fallback result.
    anthropic_client:
        An initialized ``anthropic.Anthropic`` client instance.
        Required for "anthropic" backend. Ignored for "claude_cli".
    anthropic_model:
        Model ID for Anthropic API calls (fast brain). Slow brain
        always uses Opus regardless of this setting.
    n_debate_rounds:
        Number of bull-vs-bear debate rounds. Default 2.
    fast_max_tokens:
        Max tokens for Analyst agents (fast brain).
    slow_max_tokens:
        Max tokens for Researcher and Judge agents (slow brain).
    on_step:
        Optional callback: on_step(stage, detail, data).
        Stages: "analyst", "bull", "bear", "judge", "done".

    Returns
    -------
    dict[str, Any]
        Result dictionary with keys: regime, confidence, decisive_factors,
        reasoning, analyst_views, debate_rounds, total_tokens, total_cost_usd,
        elapsed_seconds.
    """
    start = time.monotonic()
    total_tokens = 0
    total_cost = 0.0

    # ------------------------------------------------------------------
    # Detect LLM backend availability (CLI preferred, API fallback)
    # ------------------------------------------------------------------
    has_cli = backend in ("claude_cli", "auto") and _has_claude_cli()
    has_api = anthropic_client is not None and backend in ("anthropic", "auto")

    if not has_cli and not has_api:
        logger.warning("No LLM backend available — returning fallback")
        return _fallback_result(time.monotonic() - start)

    # ------------------------------------------------------------------
    # Phase 1: Analyst Layer (4 agents, fast brain, domain-isolated)
    # ------------------------------------------------------------------
    analyst_views: list[dict[str, Any]] = []
    for domain in ("trend", "zone", "macro", "risk"):
        user_ctx = _format_domain_context(features, domain)
        try:
            content, tokens, cost = _chat(
                system=ANALYST_SYSTEM_PROMPTS[domain],
                user=user_ctx,
                max_tokens=fast_max_tokens,
                brain="fast",
                backend=backend,
                anthropic_client=anthropic_client,
                anthropic_model=anthropic_model,
            )
            total_tokens += tokens
            total_cost += cost
            view = _parse_analyst_json(content, domain)
        except Exception as exc:
            logger.warning("Analyst [%s] failed: %s", domain, exc)
            view = {
                "domain": domain,
                "regime_vote": "TRANSITION",
                "confidence": 0.3,
                "reasoning": f"(LLM error) {exc}",
            }
        analyst_views.append(view)
        logger.debug(
            "Analyst [%s]: vote=%s conf=%.2f",
            domain, view["regime_vote"], view["confidence"],
        )
        if on_step:
            on_step("analyst", f"{domain}: {view['regime_vote']} ({view['confidence']:.0%})", view)

    # ------------------------------------------------------------------
    # Phase 2: Researcher Layer (Bull + Bear, slow brain, n rounds)
    # ------------------------------------------------------------------
    feature_summary = _format_full_feature_summary(features)
    analyst_summary = "\n\n".join(
        f"[{v['domain'].upper()} ANALYST — vote={v['regime_vote']}, "
        f"confidence={v['confidence']:.2f}]\n{v['reasoning']}"
        for v in analyst_views
    )
    researcher_ctx = f"{analyst_summary}\n\n{feature_summary}"

    debate_rounds: list[dict[str, Any]] = []
    bull_history: list[str] = []
    bear_history: list[str] = []

    for round_num in range(1, n_debate_rounds + 1):
        # Build round-specific context with prior round history
        round_prefix = ""
        if round_num > 1:
            round_prefix = (
                "\n=== PRIOR DEBATE ROUNDS ===\n"
                + "\n---\n".join(
                    f"Round {i + 1}:\nBULL: {b}\nBEAR: {r}"
                    for i, (b, r) in enumerate(zip(bull_history, bear_history))
                )
                + f"\n\nNow respond for Round {round_num}. "
                "Do NOT repeat features you already cited in prior rounds.\n"
            )

        bull_user = researcher_ctx + round_prefix
        try:
            bull_content, bull_tokens, bull_cost = _chat(
                system=BULL_RESEARCHER_SYSTEM,
                user=bull_user,
                max_tokens=slow_max_tokens,
                brain="slow",
                backend=backend,
                anthropic_client=anthropic_client,
                anthropic_model=anthropic_model,
            )
            total_tokens += bull_tokens
            total_cost += bull_cost
        except Exception as exc:
            logger.warning("Bull researcher round %d failed: %s", round_num, exc)
            bull_content = f"BULLISH: (LLM error) {exc}"

        bear_user = researcher_ctx + round_prefix + f"\n\nBull's argument:\n{bull_content}"
        try:
            bear_content, bear_tokens, bear_cost = _chat(
                system=BEAR_RESEARCHER_SYSTEM,
                user=bear_user,
                max_tokens=slow_max_tokens,
                brain="slow",
                backend=backend,
                anthropic_client=anthropic_client,
                anthropic_model=anthropic_model,
            )
            total_tokens += bear_tokens
            total_cost += bear_cost
        except Exception as exc:
            logger.warning("Bear researcher round %d failed: %s", round_num, exc)
            bear_content = f"BEARISH: (LLM error) {exc}"

        bull_history.append(bull_content)
        bear_history.append(bear_content)
        rnd = {
            "round_num": round_num,
            "bull_argument": bull_content,
            "bear_argument": bear_content,
        }
        debate_rounds.append(rnd)
        logger.debug("Debate round %d complete", round_num)
        if on_step:
            on_step("bull", f"R{round_num}: {bull_content[:80]}...", rnd)
            on_step("bear", f"R{round_num}: {bear_content[:80]}...", rnd)

    # ------------------------------------------------------------------
    # Phase 3: Judge Agent (slow brain, debate transcript + features)
    # Per context hygiene: does NOT read raw analyst views
    # ------------------------------------------------------------------
    debate_transcript = "\n\n".join(
        f"--- Round {r['round_num']} ---\n"
        f"BULL: {r['bull_argument']}\n\n"
        f"BEAR: {r['bear_argument']}"
        for r in debate_rounds
    )
    judge_ctx = (
        f"=== DEBATE TRANSCRIPT ===\n{debate_transcript}\n\n"
        f"=== COMPACT FEATURE SUMMARY ===\n{feature_summary}"
    )

    try:
        judge_content, judge_tokens, judge_cost = _chat(
            system=JUDGE_SYSTEM,
            user=judge_ctx,
            max_tokens=slow_max_tokens,
            brain="slow",
            backend=backend,
            anthropic_client=anthropic_client,
            anthropic_model=anthropic_model,
        )
        total_tokens += judge_tokens
        total_cost += judge_cost
        verdict = _parse_judge_json(judge_content)
        if on_step:
            on_step("judge", f"{verdict['regime']} ({verdict['confidence']:.0%})", verdict)
    except Exception as exc:
        logger.warning("Judge agent failed: %s", exc)
        verdict = {
            "regime": "TRANSITION",
            "confidence": 0.3,
            "decisive_factors": (),
            "reasoning": f"(Judge LLM error) {exc}",
        }

    elapsed = time.monotonic() - start
    if on_step:
        on_step("done", f"{elapsed:.0f}s, ${total_cost:.3f}", None)

    logger.info(
        "Debate complete: regime=%s conf=%.2f tokens=%d cost=$%.3f elapsed=%.1fs",
        verdict["regime"],
        verdict["confidence"],
        total_tokens,
        total_cost,
        elapsed,
    )

    return {
        "regime": verdict["regime"],
        "confidence": verdict["confidence"],
        "decisive_factors": tuple(verdict.get("decisive_factors", ())),
        "reasoning": verdict["reasoning"],
        "analyst_views": tuple(
            {
                "domain": v["domain"],
                "regime_vote": v["regime_vote"],
                "confidence": v["confidence"],
                "reasoning": v["reasoning"],
            }
            for v in analyst_views
        ),
        "debate_rounds": tuple(
            {
                "round_num": r["round_num"],
                "bull_argument": r["bull_argument"],
                "bear_argument": r["bear_argument"],
            }
            for r in debate_rounds
        ),
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "elapsed_seconds": elapsed,
    }
