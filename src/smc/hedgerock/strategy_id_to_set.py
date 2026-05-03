"""Resolve a HedgeRock ``strategy_id`` to its corresponding ``.set`` parameters.

A ``strategy_id`` is a triple-token slug of the form
``{symbol_lc}_{tf_lc}_{regime_lc}`` — e.g.::

    xauusd_h1_trend_up      -> XAUUSD / H1 / TREND_UP
    eurusd_m15_consolidation-> EURUSD / M15 / CONSOLIDATION
    audcad_h4_transition    -> AUDCAD / H4 / TRANSITION

The function ``resolve_set_for_strategy`` walks a deterministic three-step
fallback chain to locate a ``.set`` file inside a directory of curated
HedgeRock parameter presets (default: ``HedgeRock/HedeRockEXAMPLEsets``):

    1. **Exact match**:   ``{symbol_lc}_{tf_lc}_{regime_lc}.set`` — the
       most specific preset for the current (symbol, tf, regime) triple.
    2. **Symbol fallback**:  ``real-{SYMBOL}.set`` — the curated baseline
       set the user shipped with HedgeRock for that instrument.
    3. **Universal fallback**: ``real-XAUUSD.set`` — XAUUSD is the only
       instrument both AI-SMC and HedgeRock are tuned for; using its
       ``.set`` as terminal fallback guarantees a non-empty result.

If even the universal fallback is missing (a deliberately broken setup),
``StrategyResolutionError`` is raised so the caller does not silently
push a default-zero envelope to the EA.

Format expectations (verified against
``$HEDGEROCK_SETS_DIR/*.set`` —
default ``$HOME/HedgeRock/HedeRockEXAMPLEsets/*.set``):

- Plain ASCII or UTF-8; **handles UTF-8 BOM** transparently.
- **CRLF or LF** line endings (``newline=None`` universal newline mode).
- Lines are ``key=value``; first ``=`` splits, remainder is value.
- The value may contain spaces (e.g. ``footer=        HedgeRock EA``);
  do NOT strip whitespace from the value.
- Comment lines starting with ``;`` or ``#`` are skipped (MT5 also accepts
  these though Aldo's exports do not contain any).
- Section header rows are MQL5 ``extern string`` separators of the form
  ``______NAME______=______NAME______`` — they are kept in the dict
  because callers may want to round-trip the file unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from smc.ai.models import MarketRegimeAI


__all__ = [
    "DEFAULT_SETS_DIR",
    "P0_DANGEROUS_DEFAULTS",
    "StrategyId",
    "StrategyResolutionError",
    "UNIVERSAL_FALLBACK_BASENAME",
    "audit_set_parameters",
    "load_set_file",
    "parse_strategy_id",
    "resolve_set_for_strategy",
]


# ---------------------------------------------------------------------------
# Defaults & sentinels
# ---------------------------------------------------------------------------


def _default_sets_dir() -> Path:
    raw = os.environ.get("HEDGEROCK_SETS_DIR")
    if raw:
        return Path(raw).expanduser()
    home = os.environ.get("HEDGEROCK_HOME")
    base = Path(home).expanduser() if home else Path.home() / "HedgeRock"
    return base / "HedeRockEXAMPLEsets"


DEFAULT_SETS_DIR: Final[Path] = _default_sets_dir()
"""Where the curated HedgeRock ``.set`` library lives on this
workstation. Overrideable via ``$HEDGEROCK_SETS_DIR`` (highest
priority) or ``$HEDGEROCK_HOME`` (root); defaults to
``$HOME/HedgeRock/HedeRockEXAMPLEsets``. Every public function also
takes a ``sets_dir`` argument so callers / tests can stay decoupled
from the workstation layout."""


UNIVERSAL_FALLBACK_BASENAME: Final[str] = "real-XAUUSD"
"""The terminal fallback file when nothing more specific exists.

XAUUSD is the only instrument both AI-SMC and HedgeRock are tuned for, so
using its ``.set`` as the universal default avoids ever returning empty.
"""


# Allowed regime tokens, lower-cased. Mirrors ``MarketRegimeAI`` in
# ``smc.ai.models`` — kept in sync via the parser below.
_VALID_REGIME_LCASE: Final[frozenset[str]] = frozenset(
    {"trend_up", "trend_down", "consolidation", "transition", "ath_breakout"}
)


# ---------------------------------------------------------------------------
# P0 risk-parameter audit (Phase 2 lead increment)
# ---------------------------------------------------------------------------

P0_DANGEROUS_DEFAULTS: Final[tuple[tuple[str, str, float | None, float | None, str], ...]] = (
    # (input_name, severity, low_threshold, high_threshold, rationale)
    #
    # ``low_threshold`` — value <= this is *safe*; greater is dangerous.
    # ``high_threshold`` — value >= this is *safe*; smaller is dangerous.
    # ``None`` on either side means "no bound on that side".
    #
    # The thresholds come from ``hedgerock-redflag-mapping.md §4`` (the
    # P0 ticket table) cross-referenced with ``hedgerock-risk-audit.md``
    # default-value chapters. These are the 6 inputs whose default
    # values are effectively "off" or "lethal" and that must be
    # narrowed before *any* serious backtest.
    (
        "MaxEquityDrawDown",
        "CRITICAL",
        0.10,
        None,
        "default 0.8 (=80% DD) is structurally an AGG tier; clamp <= 0.10",
    ),
    (
        "MaxOrderLoss",
        "CRITICAL",
        1_000.0,
        None,
        "default 1e10 disables single-trade loss cap; clamp <= 1000 USD",
    ),
    (
        "maxSpread",
        "HIGH",
        500.0,
        None,
        "default 100000 effectively disables spread filter; clamp <= 500",
    ),
    (
        "bailout",
        "HIGH",
        50.0,
        None,
        "default 100000 makes Bailotlots unreachable so PauseMultiple "
        "never fires; clamp <= 50",
    ),
    (
        "GearRH",
        "CRITICAL",
        0.0,
        None,
        "ANY non-zero value enables KC A6-2 'double-direction + martingale' "
        "behaviour that lost 13/14 months in KC backtests; only 0.0 is safe",
    ),
    (
        "EvaluationDelay",
        "HIGH",
        0.0,
        None,
        "non-zero EvaluationDelay skips news/order management/new entry "
        "(see lead-intel §3 modified by deep-dive §2.2); only 0.0 is "
        "safe in live mode",
    ),
)


def _coerce_float(raw: str) -> float | None:
    """Best-effort float parse for ``.set`` values.

    ``.set`` values can be ``"true"`` / ``"false"`` (boolean inputs),
    arbitrary strings (``EAOrderComment``), or numerics with optional
    trailing zeros. Only numerics participate in the audit; everything
    else returns ``None`` so the caller skips the check.
    """
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def audit_set_parameters(parameters: dict[str, str]) -> tuple[str, ...]:
    """Return human-readable warnings for any P0-dangerous values.

    Each entry in :data:`P0_DANGEROUS_DEFAULTS` is checked against the
    matching input in ``parameters`` (skipped if absent: the .set author
    just left the user's MT5 default in place, which we still flag in
    the EA itself but cannot detect from the .set).

    The string format is intentionally stable so callers — like
    ``short_backtest`` — can grep ``"CRITICAL"`` to filter out unsafe
    candidates without reaching into structured data.

    Returns:
        Tuple of strings of the form
        ``"<SEVERITY> <input>=<value>: <rationale>"``. Empty when no
        dangerous defaults are detected.
    """
    warnings: list[str] = []
    for name, severity, low, high, rationale in P0_DANGEROUS_DEFAULTS:
        raw_value = parameters.get(name)
        if raw_value is None:
            continue
        numeric = _coerce_float(raw_value)
        if numeric is None:
            # Non-numeric — e.g. a future enum-style override. Skip
            # rather than emit a noisy false-positive.
            continue
        is_dangerous = False
        if low is not None and numeric > low:
            is_dangerous = True
        if high is not None and numeric < high:
            is_dangerous = True
        if is_dangerous:
            warnings.append(
                f"{severity} {name}={raw_value}: {rationale}"
            )
    return tuple(warnings)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyId:
    """Parsed components of a ``strategy_id`` slug.

    The original raw slug is kept for round-trip / logging.
    """

    raw: str
    symbol: str  # uppercase, e.g. ``XAUUSD``
    timeframe: str  # uppercase, e.g. ``H1`` / ``M15``
    regime: MarketRegimeAI  # uppercase, e.g. ``TREND_UP``


class StrategyResolutionError(RuntimeError):
    """Raised when the fallback chain exhausts without finding a ``.set`` file."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_strategy_id(strategy_id: str) -> StrategyId:
    """Split ``strategy_id`` into ``(symbol, timeframe, regime)``.

    Acceptable shapes::

        xauusd_h1_trend_up
        eurusd_m15_consolidation
        audcad_h4_transition

    The function is *strict*:

    - The slug must split into 3 *or more* tokens by ``_``.
    - The leading token is the symbol, the next is the timeframe, the
      remainder joined by ``_`` is the regime — this lets ``trend_up`` /
      ``trend_down`` / ``ath_breakout`` parse correctly.
    - The regime token must be one of the documented :data:`_VALID_REGIME_LCASE`
      values (so a typo produces ``ValueError``, not a silent fallback).

    Raises:
        ValueError: If the slug is malformed or the regime is unknown.
    """
    if not strategy_id:
        raise ValueError("strategy_id cannot be empty")
    tokens = strategy_id.strip().lower().split("_")
    if len(tokens) < 3:
        raise ValueError(
            f"strategy_id must be 'symbol_tf_regime' with at least 3 tokens, "
            f"got {strategy_id!r}"
        )

    symbol_lc, tf_lc, *regime_parts = tokens
    if not symbol_lc or not tf_lc or not regime_parts:
        raise ValueError(f"strategy_id has empty token(s): {strategy_id!r}")

    regime_lc = "_".join(regime_parts)
    if regime_lc not in _VALID_REGIME_LCASE:
        raise ValueError(
            f"unknown regime token {regime_lc!r}; "
            f"expected one of {sorted(_VALID_REGIME_LCASE)}"
        )

    return StrategyId(
        raw=strategy_id,
        symbol=symbol_lc.upper(),
        timeframe=tf_lc.upper(),
        regime=regime_lc.upper(),  # type: ignore[return-value]
    )


# ---------------------------------------------------------------------------
# .set file parsing
# ---------------------------------------------------------------------------


def load_set_file(path: Path | str) -> dict[str, str]:
    """Parse a HedgeRock ``.set`` file and return ``{key: value}``.

    The file is read in **universal-newline mode** so CRLF and LF both
    work. A leading UTF-8 BOM (``﻿``) is stripped from the first
    key. Values are kept verbatim — leading/trailing spaces inside
    the value are preserved, only the trailing newline is removed.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if a non-empty line contains no ``=`` separator.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    result: dict[str, str] = {}
    with file_path.open("r", encoding="utf-8-sig", newline=None) as fh:
        for line_no, raw in enumerate(fh, start=1):
            # Strip newline only — never trim value whitespace.
            line = raw.rstrip("\r\n")
            if not line:
                continue
            stripped = line.lstrip()
            # MT5 .set comments (rare in Aldo's exports but valid).
            if stripped.startswith(";") or stripped.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(
                    f"{file_path}:{line_no}: expected 'key=value', got {line!r}"
                )
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(
                    f"{file_path}:{line_no}: empty key in {line!r}"
                )
            # Last write wins — MT5 itself behaves the same way.
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------


def _candidate_basenames(strat: StrategyId) -> tuple[str, str, str]:
    """Build the three-step lookup order, lowest-priority last.

    Returns:
        ``(exact, symbol, universal)`` tuple of basenames (no ``.set`` suffix).
    """
    exact = f"{strat.symbol.lower()}_{strat.timeframe.lower()}_{strat.regime.lower()}"
    symbol_default = f"real-{strat.symbol}"
    return exact, symbol_default, UNIVERSAL_FALLBACK_BASENAME


def _resolve_path(sets_dir: Path, basename: str) -> Path | None:
    """Return ``sets_dir/basename.set`` if it exists, else ``None``."""
    candidate = sets_dir / f"{basename}.set"
    return candidate if candidate.is_file() else None


@dataclass(frozen=True)
class ResolvedSet:
    """Outcome of a fallback-chain lookup.

    The ``warnings`` tuple is populated by
    :func:`audit_set_parameters` against :data:`P0_DANGEROUS_DEFAULTS`.
    Downstream consumers (notably ``short_backtest.run_short_backtest``)
    use the presence of any ``CRITICAL`` warning to *exclude* the
    strategy from candidate ranking — we never want a known-bad .set to
    win the daily backtest.
    """

    strategy_id: StrategyId
    set_path: Path
    fallback_level: int  # 0 = exact, 1 = symbol, 2 = universal
    parameters: dict[str, str]
    warnings: tuple[str, ...] = ()


def resolve_set_for_strategy(
    strategy_id: str,
    *,
    sets_dir: Path | str | None = None,
) -> ResolvedSet:
    """Resolve ``strategy_id`` to a parsed ``.set`` parameter dict.

    Walks the fallback chain (see module docstring) and returns a
    :class:`ResolvedSet` describing the chosen file plus its parameters.

    Args:
        strategy_id: e.g. ``"xauusd_h1_trend_up"``.
        sets_dir: Override directory holding the ``.set`` files. Defaults
            to :data:`DEFAULT_SETS_DIR`.

    Raises:
        ValueError: If ``strategy_id`` is malformed (delegated to
            :func:`parse_strategy_id`).
        StrategyResolutionError: If none of the three candidates exists.
    """
    strat = parse_strategy_id(strategy_id)
    base = Path(sets_dir) if sets_dir is not None else DEFAULT_SETS_DIR
    candidates: tuple[str, str, str] = _candidate_basenames(strat)
    for level, basename in enumerate(candidates):
        path = _resolve_path(base, basename)
        if path is not None:
            params = load_set_file(path)
            warnings = audit_set_parameters(params)
            return ResolvedSet(
                strategy_id=strat,
                set_path=path,
                fallback_level=level,
                parameters=params,
                warnings=warnings,
            )
    raise StrategyResolutionError(
        f"no .set file found for strategy_id={strategy_id!r} in {base}; "
        f"tried {[c + '.set' for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Convenience for tests
# ---------------------------------------------------------------------------


def list_available_strategy_files(
    sets_dir: Path | str | None = None,
) -> tuple[Path, ...]:
    """Return all ``.set`` files under ``sets_dir`` sorted by name.

    Used by smoke tests / CLI tooling to inspect the curated library.
    """
    base = Path(sets_dir) if sets_dir is not None else DEFAULT_SETS_DIR
    if not base.is_dir():
        return ()
    return tuple(sorted(p for p in base.glob("*.set")))


__all__ += ["ResolvedSet", "list_available_strategy_files"]
