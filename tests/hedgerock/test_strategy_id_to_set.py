"""Tests for ``smc.hedgerock.strategy_id_to_set``.

Covers:

- ``parse_strategy_id`` token splitting & validation.
- ``load_set_file`` reading real CRLF .set files, BOM handling, value
  whitespace preservation, comments.
- ``resolve_set_for_strategy`` fallback chain (exact / symbol / universal).
- ``StrategyResolutionError`` when chain exhausts.

The real example files in ``$HEDGEROCK_SETS_DIR`` (default
``$HOME/HedgeRock/HedeRockEXAMPLEsets``)
are used in one integration-style test; everything else is hermetic via
``tmp_path`` fixtures.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from smc.hedgerock.strategy_id_to_set import (
    DEFAULT_SETS_DIR,
    P0_DANGEROUS_DEFAULTS,
    ResolvedSet,
    StrategyId,
    StrategyResolutionError,
    UNIVERSAL_FALLBACK_BASENAME,
    audit_set_parameters,
    list_available_strategy_files,
    load_set_file,
    parse_strategy_id,
    resolve_set_for_strategy,
)


# ---------------------------------------------------------------------------
# parse_strategy_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_canonical_three_token_id() -> None:
    parsed = parse_strategy_id("xauusd_h1_transition")
    assert parsed == StrategyId(
        raw="xauusd_h1_transition",
        symbol="XAUUSD",
        timeframe="H1",
        regime="TRANSITION",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("slug", "expected_regime"),
    [
        ("xauusd_h1_trend_up", "TREND_UP"),
        ("xauusd_h1_trend_down", "TREND_DOWN"),
        ("xauusd_h4_ath_breakout", "ATH_BREAKOUT"),
    ],
)
def test_parse_handles_multi_token_regimes(slug: str, expected_regime: str) -> None:
    parsed = parse_strategy_id(slug)
    assert parsed.regime == expected_regime


@pytest.mark.unit
def test_parse_uppercases_symbol_and_timeframe() -> None:
    parsed = parse_strategy_id("EURUSD_M15_consolidation")
    assert parsed.symbol == "EURUSD"
    assert parsed.timeframe == "M15"


@pytest.mark.unit
def test_parse_rejects_empty() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_strategy_id("")


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["xauusd", "xauusd_h1", "_h1_trend_up"])
def test_parse_rejects_too_few_tokens_or_empty_token(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_strategy_id(bad)


@pytest.mark.unit
def test_parse_rejects_unknown_regime() -> None:
    with pytest.raises(ValueError, match="unknown regime"):
        parse_strategy_id("xauusd_h1_bogus_regime")


# ---------------------------------------------------------------------------
# load_set_file
# ---------------------------------------------------------------------------


def _write_set(path: Path, body: str, *, eol: str = "\r\n", bom: bool = False) -> None:
    blob = body.replace("\n", eol)
    raw = blob.encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


@pytest.mark.unit
def test_load_set_basic_key_value(tmp_path: Path) -> None:
    p = tmp_path / "basic.set"
    _write_set(p, "EvaluationDelay=0\nStartLotsi=0.05\n")
    parsed = load_set_file(p)
    assert parsed == {"EvaluationDelay": "0", "StartLotsi": "0.05"}


@pytest.mark.unit
def test_load_set_handles_utf8_bom(tmp_path: Path) -> None:
    """First key must not start with the BOM byte even if file has one."""
    p = tmp_path / "bom.set"
    _write_set(p, "EvaluationDelay=0\n", bom=True)
    parsed = load_set_file(p)
    assert "EvaluationDelay" in parsed  # no leading BOM character
    assert parsed["EvaluationDelay"] == "0"


@pytest.mark.unit
def test_load_set_handles_lf_line_endings(tmp_path: Path) -> None:
    p = tmp_path / "lf.set"
    _write_set(p, "A=1\nB=2\n", eol="\n")
    assert load_set_file(p) == {"A": "1", "B": "2"}


@pytest.mark.unit
def test_load_set_handles_crlf_line_endings(tmp_path: Path) -> None:
    p = tmp_path / "crlf.set"
    _write_set(p, "A=1\nB=2\n", eol="\r\n")
    assert load_set_file(p) == {"A": "1", "B": "2"}


@pytest.mark.unit
def test_load_set_preserves_value_whitespace(tmp_path: Path) -> None:
    """``footer=        HedgeRock EA`` keeps its leading spaces."""
    p = tmp_path / "footer.set"
    _write_set(p, "footer=        HedgeRock EA\nMAGICNUM=20222222\n")
    parsed = load_set_file(p)
    assert parsed["footer"] == "        HedgeRock EA"
    assert parsed["MAGICNUM"] == "20222222"


@pytest.mark.unit
def test_load_set_keeps_section_headers(tmp_path: Path) -> None:
    """Section markers like ``______LOTS______=______LOTS______`` are kept."""
    p = tmp_path / "sections.set"
    body = (
        "EvaluationDelay=0\n"
        "______LOTS______=______LOTS______\n"
        "StartLotsi=0.1\n"
    )
    _write_set(p, body)
    parsed = load_set_file(p)
    assert parsed["______LOTS______"] == "______LOTS______"


@pytest.mark.unit
def test_load_set_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "blank.set"
    _write_set(p, "A=1\n\n\nB=2\n")
    assert load_set_file(p) == {"A": "1", "B": "2"}


@pytest.mark.unit
@pytest.mark.parametrize("comment_char", [";", "#"])
def test_load_set_skips_comment_lines(tmp_path: Path, comment_char: str) -> None:
    p = tmp_path / "comment.set"
    _write_set(p, f"A=1\n{comment_char} this is a comment\nB=2\n")
    assert load_set_file(p) == {"A": "1", "B": "2"}


@pytest.mark.unit
def test_load_set_last_write_wins(tmp_path: Path) -> None:
    p = tmp_path / "dup.set"
    _write_set(p, "A=1\nA=2\nA=3\n")
    assert load_set_file(p) == {"A": "3"}


@pytest.mark.unit
def test_load_set_rejects_missing_separator(tmp_path: Path) -> None:
    p = tmp_path / "bad.set"
    _write_set(p, "A=1\nNotAPair\n")
    with pytest.raises(ValueError, match="expected 'key=value'"):
        load_set_file(p)


@pytest.mark.unit
def test_load_set_rejects_empty_key(tmp_path: Path) -> None:
    p = tmp_path / "emptykey.set"
    _write_set(p, "=value\n")
    with pytest.raises(ValueError, match="empty key"):
        load_set_file(p)


@pytest.mark.unit
def test_load_set_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_set_file(tmp_path / "nope.set")


# ---------------------------------------------------------------------------
# resolve_set_for_strategy — fallback chain
# ---------------------------------------------------------------------------


@pytest.fixture
def sets_dir(tmp_path: Path) -> Path:
    """Empty .set directory the tests populate as needed."""
    d = tmp_path / "sets"
    d.mkdir()
    # A universal fallback is required for the chain not to bottom out
    # in tests that don't set anything else up.
    _write_set(
        d / f"{UNIVERSAL_FALLBACK_BASENAME}.set",
        "Universal=true\nEvaluationDelay=0\n",
    )
    return d


@pytest.mark.unit
def test_resolve_picks_exact_match_when_present(sets_dir: Path) -> None:
    _write_set(
        sets_dir / "xauusd_h1_trend_up.set", "Match=exact\nEvaluationDelay=0\n"
    )
    _write_set(sets_dir / "real-XAUUSD.set", "Universal=true\n")
    result = resolve_set_for_strategy(
        "xauusd_h1_trend_up", sets_dir=sets_dir
    )
    assert isinstance(result, ResolvedSet)
    assert result.fallback_level == 0
    assert result.parameters["Match"] == "exact"
    assert result.set_path.name == "xauusd_h1_trend_up.set"
    assert result.strategy_id.regime == "TREND_UP"


@pytest.mark.unit
def test_resolve_falls_back_to_real_symbol_set(sets_dir: Path) -> None:
    """No exact match → ``real-{SYMBOL}.set`` is used."""
    _write_set(sets_dir / "real-EURUSD.set", "Symbol=eurusd\nEvaluationDelay=0\n")
    result = resolve_set_for_strategy(
        "eurusd_h4_consolidation", sets_dir=sets_dir
    )
    assert result.fallback_level == 1
    assert result.parameters["Symbol"] == "eurusd"
    assert result.set_path.name == "real-EURUSD.set"


@pytest.mark.unit
def test_resolve_uses_universal_fallback_when_symbol_missing(
    sets_dir: Path,
) -> None:
    """Unknown symbol → ``real-XAUUSD.set`` (the curated terminal default)."""
    result = resolve_set_for_strategy("btcusd_h1_trend_up", sets_dir=sets_dir)
    assert result.fallback_level == 2
    assert result.set_path.name == f"{UNIVERSAL_FALLBACK_BASENAME}.set"
    assert result.parameters["Universal"] == "true"


@pytest.mark.unit
def test_resolve_raises_when_chain_exhausts(tmp_path: Path) -> None:
    """If even the universal fallback is missing, raise loudly."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(StrategyResolutionError, match="no .set file found"):
        resolve_set_for_strategy("xauusd_m5_transition", sets_dir=empty)


@pytest.mark.unit
def test_resolve_propagates_parse_error(sets_dir: Path) -> None:
    with pytest.raises(ValueError, match="unknown regime"):
        resolve_set_for_strategy("xauusd_h1_BAD", sets_dir=sets_dir)


# ---------------------------------------------------------------------------
# Integration with the real curated library
# ---------------------------------------------------------------------------


def _hedgerock_sets_dir() -> Path:
    raw = os.environ.get("HEDGEROCK_SETS_DIR")
    if raw:
        return Path(raw).expanduser()
    home = os.environ.get("HEDGEROCK_HOME")
    base = Path(home).expanduser() if home else Path.home() / "HedgeRock"
    return base / "HedeRockEXAMPLEsets"


_HEDGEROCK_SETS = _hedgerock_sets_dir()


@pytest.mark.integration
@pytest.mark.skipif(
    not _HEDGEROCK_SETS.is_dir(),
    reason="HedgeRock example .set directory not available on this host",
)
def test_resolve_against_real_library_xauusd_falls_to_symbol_set() -> None:
    """No ``xauusd_h1_trend_up.set`` exists → ``real-XAUUSD.set`` is taken.

    For XAUUSD the symbol-level fallback IS the universal fallback (both
    point at ``real-XAUUSD.set``), but the chain finds the symbol entry
    first so ``fallback_level`` is ``1``, not ``2``. This is exactly the
    behaviour we want — the chain stops at the most specific match and
    never re-tries a file already seen.
    """
    result = resolve_set_for_strategy(
        "xauusd_h1_trend_up", sets_dir=_HEDGEROCK_SETS
    )
    assert result.fallback_level == 1
    assert result.set_path.name == "real-XAUUSD.set"
    # real-XAUUSD.set ships with at least these recognisable keys.
    assert "EvaluationDelay" in result.parameters
    assert "MAGICNUM" in result.parameters
    assert "footer" in result.parameters
    # Whitespace must survive through the loader unchanged.
    assert result.parameters["footer"] == "        HedgeRock EA"


@pytest.mark.integration
@pytest.mark.skipif(
    not _HEDGEROCK_SETS.is_dir(),
    reason="HedgeRock example .set directory not available on this host",
)
def test_resolve_against_real_library_unknown_symbol_uses_universal() -> None:
    """An instrument with no ``real-{SYMBOL}.set`` lands on ``real-XAUUSD.set``."""
    # ``BTCUSD`` has no curated file in the example library.
    result = resolve_set_for_strategy(
        "btcusd_h1_trend_up", sets_dir=_HEDGEROCK_SETS
    )
    assert result.fallback_level == 2
    assert result.set_path.name == f"{UNIVERSAL_FALLBACK_BASENAME}.set"


@pytest.mark.integration
@pytest.mark.skipif(
    not _HEDGEROCK_SETS.is_dir(),
    reason="HedgeRock example .set directory not available on this host",
)
def test_real_library_lists_expected_files() -> None:
    files = list_available_strategy_files(_HEDGEROCK_SETS)
    names = {p.name for p in files}
    # 11 example files documented in cross-system-lessons; check a couple.
    assert "real-XAUUSD.set" in names
    assert "real-EURUSD.set" in names
    assert len(files) >= 9  # tolerate library growth, just guard against zero


@pytest.mark.unit
def test_default_sets_dir_constant_is_documented() -> None:
    assert isinstance(DEFAULT_SETS_DIR, Path)
    assert DEFAULT_SETS_DIR.name == "HedeRockEXAMPLEsets"


# ---------------------------------------------------------------------------
# audit_set_parameters — Phase 2 lead-increment safety filter
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_audit_p0_table_contains_documented_inputs() -> None:
    """The P0 table mirrors the redflag-mapping P0 ticket list."""
    names = {entry[0] for entry in P0_DANGEROUS_DEFAULTS}
    assert names == {
        "MaxEquityDrawDown",
        "MaxOrderLoss",
        "maxSpread",
        "bailout",
        "GearRH",
        "EvaluationDelay",
    }


@pytest.mark.unit
def test_audit_clean_set_returns_empty_warnings() -> None:
    """A .set with all P0 inputs at safe values produces no warnings."""
    safe = {
        "MaxEquityDrawDown": "0.10",
        "MaxOrderLoss": "100.0",
        "maxSpread": "300",
        "bailout": "10",
        "GearRH": "0",
        "EvaluationDelay": "0",
    }
    assert audit_set_parameters(safe) == ()


@pytest.mark.unit
def test_audit_default_xauusd_set_flags_critical_inputs() -> None:
    """The shipped real-XAUUSD.set is the canonical "all-defaults" file.

    Verify the audit catches at least MaxEquityDrawDown and bailout —
    the two inputs whose defaults the redflag-mapping calls "structurally
    AGG tier".
    """
    danger = {
        "MaxEquityDrawDown": "0.8",  # KC A6-4 AGG tier territory
        "MaxOrderLoss": "10000000000",  # 1e10 = effectively disabled
        "maxSpread": "100000",  # disabled
        "bailout": "100000",  # disabled — PauseMultiple unreachable
        "GearRH": "1.2",  # KC A6-2 martingale
        "EvaluationDelay": "0",  # this one is fine
    }
    warnings = audit_set_parameters(danger)
    # 5 of 6 should fire; EvaluationDelay=0 stays silent.
    assert len(warnings) == 5
    joined = "\n".join(warnings)
    assert "MaxEquityDrawDown=0.8" in joined
    assert "GearRH=1.2" in joined
    assert "bailout=100000" in joined
    # Severity prefix is stable for downstream string-grep filtering.
    assert all(w.startswith(("CRITICAL", "HIGH")) for w in warnings)


@pytest.mark.unit
def test_audit_skips_inputs_absent_from_set() -> None:
    """When a key isn't in the .set we don't warn — the EA's own default
    is then in play and that's a separate concern (handled in EA-side code).
    """
    partial = {"GearRH": "1.2"}
    warnings = audit_set_parameters(partial)
    assert len(warnings) == 1
    assert "GearRH=1.2" in warnings[0]


@pytest.mark.unit
def test_audit_skips_non_numeric_values() -> None:
    """``EAOrderComment=foo`` should not crash the audit nor fire a warning."""
    parameters = {"GearRH": "not-a-number", "EAOrderComment": "HedgeRock EA"}
    assert audit_set_parameters(parameters) == ()


@pytest.mark.unit
def test_resolve_attaches_warnings_to_resolved_set(tmp_path: Path) -> None:
    """End-to-end: a dangerous .set returned via the fallback chain
    must surface its warnings in ``ResolvedSet.warnings``.
    """
    sets_dir = tmp_path / "sets"
    sets_dir.mkdir()
    (sets_dir / "real-XAUUSD.set").write_text(
        "MaxEquityDrawDown=0.8\nGearRH=1.2\nEvaluationDelay=0\n",
        encoding="utf-8",
    )
    result = resolve_set_for_strategy(
        "xauusd_h1_trend_up", sets_dir=sets_dir
    )
    assert len(result.warnings) == 2
    assert any("MaxEquityDrawDown" in w for w in result.warnings)
    assert any("GearRH" in w for w in result.warnings)


@pytest.mark.unit
def test_resolve_safe_set_has_no_warnings(tmp_path: Path) -> None:
    sets_dir = tmp_path / "sets"
    sets_dir.mkdir()
    (sets_dir / "real-XAUUSD.set").write_text(
        "MaxEquityDrawDown=0.10\nGearRH=0\nEvaluationDelay=0\n"
        "MaxOrderLoss=100\nmaxSpread=250\nbailout=10\n",
        encoding="utf-8",
    )
    result = resolve_set_for_strategy(
        "xauusd_h1_trend_up", sets_dir=sets_dir
    )
    assert result.warnings == ()
