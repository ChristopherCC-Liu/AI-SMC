"""AI Direction Engine — determines XAUUSD directional bias for trade filtering.

Sprint 7 core principle: "AI judges direction, SMC finds entry timing."

This module answers ONE question: should we be looking for longs, shorts, or
staying flat?  The Direction Engine does NOT generate trade signals — it provides
a directional filter that the SMC entry pipeline uses to gate setups.

Fallback chain (each step tried in order):
    1. Memory cache (TTL-based, ~4h)
    2. File cache (parquet, for backtest mode)
    3. AI direction debate (3-agent: Macro → News → Judge)
    4. SMA50 + DXY deterministic fallback
    5. Neutral default

LLM backends (auto-detected, same pattern as debate/pipeline.py):
    1. Claude Code CLI (``claude -p``) — preferred
    2. Anthropic API — fallback
    3. SMA fallback — always available, no LLM cost

Usage::

    engine = DirectionEngine()
    direction = engine.get_direction(h4_df=h4_data)
    if direction.direction == "bullish":
        # only consider long setups
        ...
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import polars as pl

from smc.ai.direction_prompts import (
    DIRECTION_JUDGE_SYSTEM,
    DIRECTION_MACRO_ANALYST_SYSTEM,
    DIRECTION_NEWS_ANALYST_SYSTEM,
)
from smc.ai.models import AIDirection, ExternalContext, H4TechnicalContext

logger = logging.getLogger(__name__)

__all__ = ["DirectionEngine"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SMA50_PERIOD = 50
_SWING_LOOKBACK = 10
_TREND_SLOPE_FLAT_THRESHOLD = 0.02  # % slope below which SMA50 is "flat"

_FAST_MODEL_CLI = "sonnet"
_SLOW_MODEL_CLI = "opus"


# ---------------------------------------------------------------------------
# H4 Technical Context Extraction
# ---------------------------------------------------------------------------


def _compute_sma(closes: list[float], period: int) -> float | None:
    """Simple moving average of the last ``period`` values."""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def extract_h4_context(h4_df: pl.DataFrame | None) -> H4TechnicalContext:
    """Compute H4 technical features for direction confirmation.

    Returns a frozen ``H4TechnicalContext`` with SMA50 direction, slope,
    and swing structure counts.  Returns a neutral context if data is
    insufficient.
    """
    if h4_df is None or h4_df.is_empty() or len(h4_df) < _SMA50_PERIOD + 5:
        return H4TechnicalContext(
            sma50_direction="flat",
            sma50_slope=0.0,
            higher_highs=0,
            lower_lows=0,
            bar_count=0 if h4_df is None or h4_df.is_empty() else len(h4_df),
        )

    closes = h4_df["close"].to_list()

    # SMA50 direction and slope
    sma_now = _compute_sma(closes, _SMA50_PERIOD)
    sma_5ago = _compute_sma(closes[:-5], _SMA50_PERIOD)

    if sma_now is None or sma_5ago is None or sma_now <= 0:
        direction: Literal["up", "down", "flat"] = "flat"
        slope = 0.0
    else:
        slope = (sma_now - sma_5ago) / sma_now * 100.0 / 5.0
        if slope > _TREND_SLOPE_FLAT_THRESHOLD:
            direction = "up"
        elif slope < -_TREND_SLOPE_FLAT_THRESHOLD:
            direction = "down"
        else:
            direction = "flat"

    # Count HH/LL from swing structure
    highs = h4_df["high"].to_list()
    lows = h4_df["low"].to_list()
    n_bars = min(len(highs), _SWING_LOOKBACK * 3)

    local_highs: list[float] = []
    local_lows: list[float] = []

    for i in range(2, n_bars - 2):
        idx = len(highs) - n_bars + i
        if (highs[idx] >= highs[idx - 1] and highs[idx] >= highs[idx - 2]
                and highs[idx] >= highs[idx + 1] and highs[idx] >= highs[idx + 2]):
            local_highs.append(highs[idx])
        if (lows[idx] <= lows[idx - 1] and lows[idx] <= lows[idx - 2]
                and lows[idx] <= lows[idx + 1] and lows[idx] <= lows[idx + 2]):
            local_lows.append(lows[idx])

    hh_count = 0
    for i in range(1, min(len(local_highs), _SWING_LOOKBACK)):
        if local_highs[-(i)] > local_highs[-(i + 1)]:
            hh_count += 1

    ll_count = 0
    for i in range(1, min(len(local_lows), _SWING_LOOKBACK)):
        if local_lows[-(i)] < local_lows[-(i + 1)]:
            ll_count += 1

    return H4TechnicalContext(
        sma50_direction=direction,
        sma50_slope=round(slope, 4),
        higher_highs=hh_count,
        lower_lows=ll_count,
        bar_count=len(h4_df),
    )


# ---------------------------------------------------------------------------
# LLM Chat (same pattern as debate/pipeline.py)
# ---------------------------------------------------------------------------


def _has_claude_cli() -> bool:
    """Check if ``claude`` CLI is available on PATH."""
    return shutil.which("claude") is not None


def _claude_cli_chat(
    system: str,
    user: str,
    model: str = _SLOW_MODEL_CLI,
    timeout: int = 120,
) -> str:
    """Call Claude Code pipe mode and return response text."""
    prompt = f"{system}\n\n---\n\n{user}"
    claude_path = shutil.which("claude") or shutil.which("claude.cmd") or "claude"
    cmd = [claude_path, "-p", "--model", model]
    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr[:300] if result.stderr else "(no stderr)"
        raise RuntimeError(f"claude -p exit {result.returncode}: {stderr}")
    return result.stdout.strip()


def _anthropic_chat(
    client: Any,
    system: str,
    user: str,
    model: str,
    max_tokens: int = 512,
) -> str:
    """Call Anthropic API and return response text."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text if response.content else ""


def _parse_direction_json(content: str) -> dict[str, Any]:
    """Parse a direction JSON response, stripping markdown fences."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    data: dict[str, Any] = json.loads(text.strip())
    return {
        "direction": str(data.get("direction", "neutral")),
        "confidence": float(data.get("confidence", 0.3)),
        "key_factors": list(data.get("key_factors", data.get("key_drivers", []))),
        "reasoning": str(data.get("reasoning", "(no reasoning)")),
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_macro_context(
    external_ctx: ExternalContext | None,
    h4_ctx: H4TechnicalContext | None = None,
) -> str:
    """Build user-message context for the Macro Analyst."""
    if external_ctx is None:
        if h4_ctx is None:
            return (
                "=== MACRO DATA ===\n"
                "  (all macro data unavailable)\n"
                "  source_quality: unavailable"
            )
        return "\n".join([
            "=== MACRO DATA ===",
            "  (all macro data unavailable — macro-free mode)",
            "  source_quality: unavailable",
            "",
            "=== H4 TECHNICAL FALLBACK ===",
            f"  sma50_direction: {h4_ctx.sma50_direction}",
            f"  sma50_slope: {h4_ctx.sma50_slope:+.4f}%",
            f"  higher_highs: {h4_ctx.higher_highs}",
            f"  lower_lows: {h4_ctx.lower_lows}",
            f"  bar_count: {h4_ctx.bar_count}",
            "",
            "Instructions: In macro-free mode, derive a provisional directional read",
            "from H4 technical structure. You MAY return neutral if H4 is also flat,",
            "but default must be an evidence-based directional call from the technicals.",
            "Confidence: 0.3-0.45 if only technicals available; ≥0.5 only with 2+ signals aligned.",
        ])
    lines = [
        "=== MACRO DATA ===",
        f"  dxy_direction: {external_ctx.dxy_direction}",
        f"  dxy_value: {external_ctx.dxy_value}",
        f"  vix_level: {external_ctx.vix_level}",
        f"  vix_regime: {external_ctx.vix_regime}",
        f"  real_rate_10y: {external_ctx.real_rate_10y}",
        f"  cot_net_spec: {external_ctx.cot_net_spec}",
        f"  central_bank_stance: {external_ctx.central_bank_stance}",
        f"  source_quality: {external_ctx.source_quality}",
    ]
    return "\n".join(lines)


def _format_news_context(
    external_ctx: ExternalContext | None,
    h4_ctx: H4TechnicalContext | None = None,
) -> str:
    """Build user-message context for the News Analyst.

    In Sprint 7 MVP, news context is derived from the same ExternalContext.
    A dedicated news feed integration is planned for Sprint 8.
    """
    if external_ctx is None:
        if h4_ctx is None:
            return (
                "=== NEWS / SENTIMENT CONTEXT ===\n"
                "  (no recent events or sentiment data available)\n"
                "  Respond with neutral, confidence 0.3."
            )
        return "\n".join([
            "=== NEWS / SENTIMENT CONTEXT ===",
            "  (no recent events or sentiment data available — macro-free mode)",
            "",
            "=== H4 TECHNICAL FALLBACK ===",
            f"  sma50_direction: {h4_ctx.sma50_direction}",
            f"  sma50_slope: {h4_ctx.sma50_slope:+.4f}%",
            f"  higher_highs: {h4_ctx.higher_highs}",
            f"  lower_lows: {h4_ctx.lower_lows}",
            f"  bar_count: {h4_ctx.bar_count}",
            "",
            "Instructions: In macro-free mode, derive a provisional directional read",
            "from H4 technical structure. You MAY return neutral if H4 is also flat,",
            "but default must be an evidence-based directional call from the technicals.",
            "Confidence: 0.3-0.45 if only technicals available; ≥0.5 only with 2+ signals aligned.",
        ])
    lines = [
        "=== NEWS / SENTIMENT CONTEXT ===",
        f"  vix_level: {external_ctx.vix_level} ({external_ctx.vix_regime})",
        f"  central_bank_stance: {external_ctx.central_bank_stance}",
        f"  dxy_direction: {external_ctx.dxy_direction}",
        f"  source_quality: {external_ctx.source_quality}",
        "",
        "Note: Detailed news feed not yet integrated (Sprint 8).",
        "Assess directional sentiment from available macro indicators.",
    ]
    return "\n".join(lines)


def _format_judge_context(
    macro_view: dict[str, Any],
    news_view: dict[str, Any],
    h4_ctx: H4TechnicalContext,
) -> str:
    """Build user-message context for the Direction Judge."""
    lines = [
        "=== MACRO ANALYST VIEW ===",
        f"  direction: {macro_view['direction']}",
        f"  confidence: {macro_view['confidence']:.2f}",
        f"  key_factors: {macro_view['key_factors']}",
        f"  reasoning: {macro_view['reasoning']}",
        "",
        "=== NEWS ANALYST VIEW ===",
        f"  direction: {news_view['direction']}",
        f"  confidence: {news_view['confidence']:.2f}",
        f"  key_factors: {news_view['key_factors']}",
        f"  reasoning: {news_view['reasoning']}",
        "",
        "=== H4 TECHNICAL CONTEXT ===",
        f"  sma50_direction: {h4_ctx.sma50_direction}",
        f"  sma50_slope: {h4_ctx.sma50_slope:+.4f}%",
        f"  higher_highs: {h4_ctx.higher_highs}",
        f"  lower_lows: {h4_ctx.lower_lows}",
        f"  bar_count: {h4_ctx.bar_count}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DirectionEngine
# ---------------------------------------------------------------------------


class DirectionEngine:
    """AI-powered direction assessment with multi-level fallback.

    Fallback chain:
        1. In-memory TTL cache (4h default)
        2. File cache (parquet, for backtest)
        3. AI 3-agent debate (Macro + News + Judge)
        4. SMA50 + DXY deterministic fallback
        5. Neutral default

    Parameters
    ----------
    external_fetcher:
        Callable that returns ``ExternalContext``.  If None, the engine
        runs without macro data (SMA fallback only).
    cache_path:
        Path to a direction cache parquet file (backtest mode).
    cache_ttl_hours:
        How long in-memory cached directions remain valid.  Default 4h
        (aligned with H4 candle frequency).
    """

    def __init__(
        self,
        *,
        external_fetcher: Any | None = None,
        cache_path: Path | str | None = None,
        cache_ttl_hours: int = 4,
    ) -> None:
        self._external_fetcher = external_fetcher
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        self._mem_cache: dict[str, tuple[AIDirection, datetime]] = {}

        # File cache (lazy import to avoid circular dependency)
        self._file_cache: Any | None = None
        if cache_path is not None:
            from smc.ai.direction_cache import DirectionCacheLookup

            self._file_cache = DirectionCacheLookup(Path(cache_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_direction(
        self,
        h4_df: pl.DataFrame | None = None,
        bar_ts: datetime | None = None,
    ) -> AIDirection:
        """Get the current directional bias for XAUUSD.

        Fallback chain: memory cache -> file cache -> AI debate ->
        SMA fallback -> neutral default.

        Parameters
        ----------
        h4_df:
            H4 OHLCV DataFrame for technical context extraction.
        bar_ts:
            Timestamp of the current bar. Used for cache lookups
            and as the cache key. Defaults to now (UTC).
        """
        now = datetime.now(tz=timezone.utc)
        ts = bar_ts or now
        cache_key = ts.strftime("%Y-%m-%dT%H")

        # Step 1: Memory cache
        if cache_key in self._mem_cache:
            cached_dir, cached_at = self._mem_cache[cache_key]
            if (now - cached_at) < self._cache_ttl:
                logger.debug("Direction cache hit (memory): %s", cache_key)
                return cached_dir

        # Step 2: File cache (backtest mode)
        if self._file_cache is not None:
            file_hit = self._file_cache.lookup(ts)
            if file_hit is not None:
                logger.debug("Direction cache hit (file): %s", ts)
                self._mem_cache[cache_key] = (file_hit, now)
                return file_hit

        # Step 3: Try AI debate
        external_ctx = None
        if self._external_fetcher is not None:
            try:
                external_ctx = self._external_fetcher()
            except Exception:
                logger.warning("External context fetch failed", exc_info=True)

        h4_ctx = extract_h4_context(h4_df)

        direction = None
        try:
            direction = self._run_direction_debate(external_ctx, h4_ctx)
        except Exception:
            logger.warning("AI direction debate failed, using SMA fallback", exc_info=True)

        if direction is None:
            # Step 4: SMA + DXY fallback
            direction = self._sma_dxy_fallback(h4_ctx, external_ctx)

        # Macro-free guardrail: cap confidence when no macro evidence available
        if external_ctx is None or not self._has_any_evidence(external_ctx):
            direction = direction.model_copy(update={
                "confidence": min(direction.confidence, 0.5),
                "reasoning_tag": "macro_free_capped",
            })

        self._mem_cache[cache_key] = (direction, now)
        return direction

    # ------------------------------------------------------------------
    # AI Debate
    # ------------------------------------------------------------------

    def _run_direction_debate(
        self,
        external_ctx: ExternalContext | None,
        h4_ctx: H4TechnicalContext,
    ) -> AIDirection:
        """Run the 3-agent direction debate via Claude CLI or Anthropic API.

        Agents:
            1. Macro Analyst (Sonnet, fast) — fundamental factors
            2. News Analyst (Sonnet, fast) — events & sentiment
            3. Judge (Opus, slow) — synthesize with H4 technical context
        """
        start = time.monotonic()
        has_cli = _has_claude_cli()

        if not has_cli:
            raise RuntimeError("No LLM backend available for direction debate")

        # Phase 1: Macro Analyst
        macro_user = _format_macro_context(external_ctx, h4_ctx)
        try:
            macro_raw = _claude_cli_chat(
                DIRECTION_MACRO_ANALYST_SYSTEM, macro_user, _FAST_MODEL_CLI,
            )
            macro_view = _parse_direction_json(macro_raw)
        except Exception as exc:
            logger.warning("Macro analyst failed: %s", exc)
            macro_view = {
                "direction": "neutral", "confidence": 0.3,
                "key_factors": [], "reasoning": f"(LLM error) {exc}",
            }

        # Phase 2: News Analyst
        news_user = _format_news_context(external_ctx, h4_ctx)
        try:
            news_raw = _claude_cli_chat(
                DIRECTION_NEWS_ANALYST_SYSTEM, news_user, _FAST_MODEL_CLI,
            )
            news_view = _parse_direction_json(news_raw)
        except Exception as exc:
            logger.warning("News analyst failed: %s", exc)
            news_view = {
                "direction": "neutral", "confidence": 0.3,
                "key_factors": [], "reasoning": f"(LLM error) {exc}",
            }

        # Phase 3: Judge
        judge_user = _format_judge_context(macro_view, news_view, h4_ctx)
        try:
            judge_raw = _claude_cli_chat(
                DIRECTION_JUDGE_SYSTEM, judge_user, _SLOW_MODEL_CLI,
            )
            judge_result = _parse_direction_json(judge_raw)
        except Exception as exc:
            logger.warning("Direction judge failed: %s", exc)
            # Fallback: use macro view directly (fundamental > sentiment)
            judge_result = macro_view

        elapsed = time.monotonic() - start
        logger.info(
            "Direction debate: %s (conf=%.2f) in %.1fs",
            judge_result["direction"], judge_result["confidence"], elapsed,
        )

        return AIDirection(
            direction=judge_result["direction"],
            confidence=max(0.1, min(0.95, judge_result["confidence"])),
            key_drivers=tuple(judge_result.get("key_factors", judge_result.get("key_drivers", []))),
            reasoning=judge_result["reasoning"][:300],
            assessed_at=datetime.now(tz=timezone.utc),
            source="ai_debate",
            cost_usd=0.0,  # pipe mode doesn't report cost
        )

    # ------------------------------------------------------------------
    # Evidence check
    # ------------------------------------------------------------------

    @staticmethod
    def _has_any_evidence(ctx: ExternalContext) -> bool:
        """Return True iff at least one evidence field is non-null/non-unknown."""
        if ctx is None:
            return False
        checks = [
            ctx.dxy_direction not in (None, "flat", "unknown"),
            ctx.vix_level is not None and ctx.vix_level > 0,
            ctx.real_rate_10y is not None,
            ctx.cot_net_spec is not None,
            ctx.central_bank_stance not in (None, "neutral", "unknown"),
        ]
        return any(checks)

    # ------------------------------------------------------------------
    # SMA + DXY Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _sma_dxy_fallback(
        h4_ctx: H4TechnicalContext,
        external_ctx: ExternalContext | None,
    ) -> AIDirection:
        """Deterministic fallback: SMA50 direction + DXY inverse.

        Logic:
            - SMA50 up + DXY weakening → bullish (0.6)
            - SMA50 down + DXY strengthening → bearish (0.6)
            - SMA50 up + DXY not strengthening → bullish (0.5)
            - SMA50 down + DXY not weakening → bearish (0.5)
            - SMA50 flat or contradictory → neutral (0.3)
        """
        sma_dir = h4_ctx.sma50_direction
        dxy_dir = external_ctx.dxy_direction if external_ctx is not None else "flat"

        direction: Literal["bullish", "bearish", "neutral"]
        confidence: float
        drivers: list[str] = []
        reasoning_parts: list[str] = []

        if sma_dir == "up":
            direction = "bullish"
            drivers.append("sma50_up")
            reasoning_parts.append(f"H4 SMA50 trending up (slope {h4_ctx.sma50_slope:+.4f}%)")
            if dxy_dir == "weakening":
                confidence = 0.6
                drivers.append("dxy_weakening")
                reasoning_parts.append("DXY weakening confirms gold bullish bias")
            elif dxy_dir == "strengthening":
                # SMA up but DXY strong → conflicting, reduce confidence
                confidence = 0.4
                drivers.append("dxy_strengthening")
                reasoning_parts.append("DXY strengthening partially offsets bullish SMA")
            else:
                confidence = 0.5
                reasoning_parts.append("DXY flat — no macro confirmation")
        elif sma_dir == "down":
            direction = "bearish"
            drivers.append("sma50_down")
            reasoning_parts.append(f"H4 SMA50 trending down (slope {h4_ctx.sma50_slope:+.4f}%)")
            if dxy_dir == "strengthening":
                confidence = 0.6
                drivers.append("dxy_strengthening")
                reasoning_parts.append("DXY strengthening confirms gold bearish bias")
            elif dxy_dir == "weakening":
                confidence = 0.4
                drivers.append("dxy_weakening")
                reasoning_parts.append("DXY weakening partially offsets bearish SMA")
            else:
                confidence = 0.5
                reasoning_parts.append("DXY flat — no macro confirmation")
        else:
            direction = "neutral"
            confidence = 0.3
            drivers.append("sma50_flat")
            reasoning_parts.append("H4 SMA50 flat — no directional bias from technicals")

        return AIDirection(
            direction=direction,
            confidence=round(confidence, 4),
            key_drivers=tuple(drivers),
            reasoning=". ".join(reasoning_parts),
            assessed_at=datetime.now(tz=timezone.utc),
            source="sma_fallback",
            cost_usd=0.0,
        )

    # ------------------------------------------------------------------
    # Neutral default
    # ------------------------------------------------------------------

    @staticmethod
    def _neutral_default() -> AIDirection:
        """Return the safest default: neutral with low confidence."""
        return AIDirection(
            direction="neutral",
            confidence=0.3,
            key_drivers=("insufficient_data",),
            reasoning="Default fallback — insufficient data for direction assessment",
            assessed_at=datetime.now(tz=timezone.utc),
            source="neutral_default",
            cost_usd=0.0,
        )
