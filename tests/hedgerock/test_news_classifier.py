"""Tests for ``smc.hedgerock.news_classifier``.

Covers (in order of locality):

1. ``compute_surprise_score`` — the only piece of arithmetic.
2. ``classify_for_xauusd`` exposure × surprise sign × currency matrix.
3. EUR threshold doubling (lead spec: "影响减半").
4. Polarity flip for negative-surprise events (Unemployment / Jobless).
5. Out-of-scope currencies (GBP, JPY) → neutral.
6. Threshold edge cases — 9.9% vs 10.1%.
7. ``surprise_score=None`` and ``direction="neutral"`` for upcoming events.
8. 12+ historical-fixture rows verifying NFP / FOMC / CPI / Retail / GDP /
   ECB matrix.

The tests are parameterised heavily so failure messages tell you which
of the (currency × surprise × exposure) cells broke without scrolling.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smc.hedgerock.news_classifier import (
    CLASSIFIER_VERSION,
    DEFAULT_SURPRISE_THRESHOLD,
    EUR_INFLUENCE_FACTOR,
    NEGATIVE_SURPRISE_KEYWORDS,
    NewsClassification,
    classify_for_xauusd,
    compute_surprise_score,
)
from smc.hedgerock.news_engine import NewsEvent


_TS = datetime(2024, 3, 6, 13, 30, tzinfo=timezone.utc)


def _evt(
    name: str,
    *,
    currency: str = "USD",
    intensity: str = "medium",
    actual: float | None = None,
    forecast: float | None = None,
) -> NewsEvent:
    """Compact NewsEvent builder for tests."""
    return NewsEvent(
        event_id=f"test-{abs(hash((name, currency))) % 10**8:08d}",
        name=name,
        currency=currency,
        intensity=intensity,  # type: ignore[arg-type]
        scheduled_at=_TS,
        actual=actual,
        forecast=forecast,
    )


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_classifier_version_is_documented() -> None:
    assert CLASSIFIER_VERSION.startswith("v")
    assert "." in CLASSIFIER_VERSION


@pytest.mark.unit
def test_default_threshold_is_ten_percent() -> None:
    assert DEFAULT_SURPRISE_THRESHOLD == pytest.approx(0.10)


@pytest.mark.unit
def test_eur_influence_factor_documented_value() -> None:
    """Lead spec: EUR is half-weighted on XAUUSD."""
    assert EUR_INFLUENCE_FACTOR == pytest.approx(0.5)


@pytest.mark.unit
def test_negative_surprise_keywords_lowercase() -> None:
    """Polarity-flip table is matched case-insensitively, so the table
    itself is canonicalised lowercase to make the matcher straightforward.
    """
    for kw in NEGATIVE_SURPRISE_KEYWORDS:
        assert kw == kw.lower()


# ---------------------------------------------------------------------------
# compute_surprise_score
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("actual", "forecast", "expected"),
    [
        (272_000.0, 200_000.0, 0.36),  # NFP beat
        (180_000.0, 200_000.0, -0.10),  # NFP miss exactly at threshold
        (3.4, 3.3, pytest.approx(0.0303, abs=1e-3)),  # CPI 3% beat
        (3.3, 3.3, 0.0),  # bang on forecast
        (-0.1, 0.2, pytest.approx(-1.5, abs=1e-3)),  # Retail Sales sign edge
        (5.0, 0.0, 5.0),  # zero forecast — denom guard at 1.0
    ],
)
def test_compute_surprise_score(
    actual: float, forecast: float, expected: float
) -> None:
    assert compute_surprise_score(actual=actual, forecast=forecast) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("actual", "forecast"),
    [(None, 200.0), (200.0, None), (None, None)],
)
def test_compute_surprise_score_returns_none_when_either_missing(
    actual: float | None, forecast: float | None
) -> None:
    assert compute_surprise_score(actual=actual, forecast=forecast) is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_classify_rejects_non_positive_threshold() -> None:
    evt = _evt("CPI", actual=3.5, forecast=3.3)
    with pytest.raises(ValueError, match="surprise_threshold must be positive"):
        classify_for_xauusd(
            evt, current_exposure_direction="long", surprise_threshold=0.0
        )
    with pytest.raises(ValueError, match="surprise_threshold must be positive"):
        classify_for_xauusd(
            evt, current_exposure_direction="long", surprise_threshold=-0.1
        )


@pytest.mark.unit
def test_classify_rejects_invalid_exposure_direction() -> None:
    evt = _evt("CPI", actual=3.5, forecast=3.3)
    with pytest.raises(ValueError, match="long' \\| 'short' \\| 'flat'"):
        classify_for_xauusd(
            evt, current_exposure_direction="bogus"  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Upcoming / unreleased events
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("actual", "forecast"),
    [(None, 200_000.0), (200_000.0, None), (None, None)],
)
def test_upcoming_event_returns_neutral_with_no_score(
    actual: float | None, forecast: float | None
) -> None:
    evt = _evt("Non-Farm Payrolls", actual=actual, forecast=forecast)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    assert cls.direction == "neutral"
    assert cls.surprise_score is None
    assert cls.impact_currency == "USD"
    assert cls.classifier_version == CLASSIFIER_VERSION


# ---------------------------------------------------------------------------
# Currency scope — out-of-scope returns neutral but still surfaces score
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("currency", ["GBP", "JPY", "AUD", "CHF"])
def test_out_of_scope_currency_yields_neutral_but_surface_score(
    currency: str,
) -> None:
    evt = _evt("Some Big Release", currency=currency, actual=5.0, forecast=4.0)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    assert cls.direction == "neutral"
    # Score is still computed for audit, even though direction is neutral.
    assert cls.surprise_score is not None
    assert cls.impact_currency == currency


# ---------------------------------------------------------------------------
# USD × surprise sign × exposure  — full 9-cell matrix
# ---------------------------------------------------------------------------


_USD_MATRIX: tuple[tuple[float, str, str], ...] = (
    # (actual_relative_to_forecast, exposure, expected_direction)
    # USD beat (actual > forecast, +0.36) → USD strong → XAU down
    (272_000.0, "long", "against"),
    (272_000.0, "short", "with"),
    (272_000.0, "flat", "neutral"),
    # USD miss (actual < forecast, -0.36) → USD weak → XAU up
    (128_000.0, "long", "with"),
    (128_000.0, "short", "against"),
    (128_000.0, "flat", "neutral"),
    # USD in-line (actual ≈ forecast) → neutral regardless of exposure
    (200_500.0, "long", "neutral"),
    (200_500.0, "short", "neutral"),
    (200_500.0, "flat", "neutral"),
)


@pytest.mark.unit
@pytest.mark.parametrize(("actual", "exposure", "expected"), _USD_MATRIX)
def test_usd_full_matrix(
    actual: float, exposure: str, expected: str
) -> None:
    evt = _evt(
        "Non-Farm Payrolls",
        currency="USD",
        actual=actual,
        forecast=200_000.0,
    )
    cls = classify_for_xauusd(evt, current_exposure_direction=exposure)  # type: ignore[arg-type]
    assert cls.direction == expected, (actual, exposure, cls)


# ---------------------------------------------------------------------------
# EUR is anti-correlated to USD on XAUUSD AND half-weighted
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_eur_below_doubled_threshold_yields_neutral() -> None:
    """EUR surprise of 15% (< 20% effective threshold) → neutral.

    A USD release of the same magnitude *would* trigger; the EUR side
    is half-weighted via the effective-threshold doubling.
    """
    # Forecast=4.0; actual=4.6 → relative surprise = +0.15 (15%)
    evt = _evt("ECB Rate Decision", currency="EUR", actual=4.6, forecast=4.0)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    assert cls.direction == "neutral"
    assert cls.surprise_score == pytest.approx(0.15)


@pytest.mark.unit
def test_eur_above_doubled_threshold_triggers_with_anti_correlation() -> None:
    """EUR strong (> 20%) → XAU UP (anti-correlated to USD) → long XAU = with."""
    # +25% surprise → above 20% effective threshold for EUR.
    evt = _evt("ECB Rate Decision", currency="EUR", actual=5.0, forecast=4.0)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    assert cls.direction == "with"
    assert cls.surprise_score == pytest.approx(0.25)


@pytest.mark.unit
def test_eur_weak_above_threshold_pushes_xau_down() -> None:
    """EUR weakens by 25% → XAU DOWN → long XAU = against."""
    evt = _evt("ECB Rate Decision", currency="EUR", actual=3.0, forecast=4.0)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    assert cls.direction == "against"


@pytest.mark.unit
def test_usd_at_same_magnitude_as_eur_neutral_still_triggers() -> None:
    """A 15% USD surprise *does* trigger because USD threshold is 10%."""
    evt = _evt("CPI YoY", currency="USD", actual=3.795, forecast=3.3)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    assert cls.surprise_score == pytest.approx(0.15)
    assert cls.direction == "against"  # USD strong → XAU down → long XAU = against


# ---------------------------------------------------------------------------
# Polarity-flip events
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unemployment_rate_higher_means_usd_weaker() -> None:
    """Unemployment 4.0 vs 3.5 = +14.3% raw, BUT polarity-flipped → USD weak."""
    evt = _evt("Unemployment Rate", currency="USD", actual=4.0, forecast=3.5)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    # Raw +14.3% → flipped to -14.3% → USD weak → XAU up → long = with
    assert cls.direction == "with"
    # Surface raw surprise (sign before flip) for audit transparency.
    assert cls.surprise_score == pytest.approx(0.143, abs=1e-3)


@pytest.mark.unit
def test_unemployment_rate_lower_means_usd_stronger() -> None:
    evt = _evt("Unemployment Rate", currency="USD", actual=3.0, forecast=3.5)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    assert cls.direction == "against"


@pytest.mark.unit
def test_jobless_claims_polarity_flip_with_short() -> None:
    """Jobless Claims fewer than expected → USD strong → XAU down → short = with."""
    evt = _evt(
        "Initial Jobless Claims",
        currency="USD",
        actual=210_000.0,
        forecast=240_000.0,
    )
    cls = classify_for_xauusd(evt, current_exposure_direction="short")
    # Raw -12.5% → flipped to +12.5% (USD strong) → XAU down → short = with.
    assert cls.direction == "with"


@pytest.mark.unit
def test_polarity_flip_keyword_is_case_insensitive() -> None:
    """Keyword scan should not care about title-case oddities."""
    evt = _evt("UNEMPLOYMENT RATE", currency="USD", actual=4.0, forecast=3.5)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    assert cls.direction == "with"  # flip applied


@pytest.mark.unit
def test_polarity_flip_does_not_apply_to_unrelated_events() -> None:
    """CPI is *not* a negative-surprise event; +14.3% directly = USD strong."""
    evt = _evt("CPI YoY", currency="USD", actual=4.0, forecast=3.5)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    # No flip → USD strong → XAU down → long = against.
    assert cls.direction == "against"


# ---------------------------------------------------------------------------
# Threshold boundary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_threshold_boundary_just_below_is_neutral() -> None:
    """+9.9% surprise sits below the 10% threshold → neutral."""
    # forecast=100 → actual=109.9 → +9.9%
    evt = _evt("CPI", currency="USD", actual=109.9, forecast=100.0)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    assert cls.direction == "neutral"
    assert cls.surprise_score == pytest.approx(0.099)


@pytest.mark.unit
def test_threshold_boundary_just_above_triggers() -> None:
    evt = _evt("CPI", currency="USD", actual=110.1, forecast=100.0)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    assert cls.direction == "against"
    assert cls.surprise_score == pytest.approx(0.101)


@pytest.mark.unit
def test_custom_threshold_overrides_default() -> None:
    """A 5% surprise is below default 10% but above a 2% custom threshold."""
    evt = _evt("CPI", currency="USD", actual=105.0, forecast=100.0)
    cls = classify_for_xauusd(
        evt,
        current_exposure_direction="long",
        surprise_threshold=0.02,
    )
    assert cls.direction == "against"  # USD beat → XAU down → long = against


# ---------------------------------------------------------------------------
# 12+ historical events fixture (lead spec)
# ---------------------------------------------------------------------------


_HISTORICAL_FIXTURE: tuple[tuple[NewsEvent, str, str], ...] = (
    # (event, exposure, expected_direction)
    # NFP beat (2024-06-07: 272K vs 200K = +36%)
    (
        _evt("Non-Farm Payrolls", currency="USD", actual=272_000.0, forecast=200_000.0),
        "long",
        "against",
    ),
    (
        _evt("Non-Farm Payrolls", currency="USD", actual=272_000.0, forecast=200_000.0),
        "short",
        "with",
    ),
    # NFP miss (2024-04-05: 175K vs 240K = -27%)
    (
        _evt("Non-Farm Payrolls", currency="USD", actual=175_000.0, forecast=240_000.0),
        "long",
        "with",
    ),
    # FOMC unchanged (in-line)
    (
        _evt("FOMC Rate Decision", currency="USD", actual=5.5, forecast=5.5),
        "long",
        "neutral",
    ),
    # CPI beat (2024-04-10: 3.5% vs 3.4%, surprise ≈ +2.9% < 10% threshold)
    (
        _evt("CPI YoY", currency="USD", actual=3.5, forecast=3.4),
        "long",
        "neutral",  # below threshold
    ),
    # CPI big beat (4.0% vs 3.4% = +17.6%)
    (
        _evt("CPI YoY", currency="USD", actual=4.0, forecast=3.4),
        "long",
        "against",
    ),
    # Retail Sales beat (USD)
    (
        _evt("Retail Sales", currency="USD", actual=0.5, forecast=0.3),
        "long",
        "against",  # +66% surprise, USD strong, XAU down, long = against
    ),
    # Retail Sales miss
    (
        _evt("Retail Sales", currency="USD", actual=-0.1, forecast=0.3),
        "long",
        "with",  # USD weak → XAU up
    ),
    # GDP advance beat (4.0 vs 3.0 = +33%)
    (
        _evt("GDP Growth Rate", currency="USD", actual=4.0, forecast=3.0),
        "long",
        "against",
    ),
    # Unemployment Rate up (polarity flipped)
    (
        _evt("Unemployment Rate", currency="USD", actual=4.2, forecast=3.7),
        "long",
        "with",  # higher unemployment → USD weak → XAU up
    ),
    # ECB Rate hike surprise (+25%)
    (
        _evt("ECB Rate Decision", currency="EUR", actual=5.0, forecast=4.0),
        "long",
        "with",  # EUR strong → XAU up → long = with
    ),
    # ECB unchanged → neutral
    (
        _evt("ECB Rate Decision", currency="EUR", actual=4.5, forecast=4.5),
        "long",
        "neutral",
    ),
    # Upcoming NFP — no direction
    (
        _evt("Non-Farm Payrolls", currency="USD", actual=None, forecast=200_000.0),
        "long",
        "neutral",
    ),
    # GBP PMI — out of scope
    (
        _evt("Manufacturing PMI", currency="GBP", actual=49.8, forecast=45.0),
        "long",
        "neutral",
    ),
)


@pytest.mark.unit
@pytest.mark.parametrize(("event", "exposure", "expected"), _HISTORICAL_FIXTURE)
def test_historical_fixture_matrix(
    event: NewsEvent, exposure: str, expected: str
) -> None:
    cls = classify_for_xauusd(event, current_exposure_direction=exposure)  # type: ignore[arg-type]
    assert cls.direction == expected, (event.name, event.actual, event.forecast, cls)


# ---------------------------------------------------------------------------
# Output dataclass invariants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_classification_is_frozen_dataclass() -> None:
    evt = _evt("CPI", actual=4.0, forecast=3.4)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    assert isinstance(cls, NewsClassification)
    with pytest.raises(Exception):  # frozen mutation must fail
        cls.direction = "with"  # type: ignore[misc]


@pytest.mark.unit
def test_classification_carries_event_passthrough() -> None:
    evt = _evt("CPI", actual=4.0, forecast=3.4)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    assert cls.event is evt  # same reference (frozen dataclass nesting)
    assert cls.impact_currency == "USD"
    assert cls.classifier_version == CLASSIFIER_VERSION


@pytest.mark.unit
def test_currency_uppercased_in_output() -> None:
    evt = _evt("CPI", currency="usd", actual=4.0, forecast=3.4)
    cls = classify_for_xauusd(evt, current_exposure_direction="long")
    assert cls.impact_currency == "USD"


# ---------------------------------------------------------------------------
# Performance — keep it cheap, decision_server iterates 10+ per cycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_classify_per_call_under_one_millisecond() -> None:
    import time

    evt = _evt("Non-Farm Payrolls", actual=272_000.0, forecast=200_000.0)
    iterations = 1_000
    t0 = time.perf_counter()
    for _ in range(iterations):
        classify_for_xauusd(evt, current_exposure_direction="long")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    per_call_us = elapsed_ms * 1000 / iterations
    # Pure arithmetic — well under 1 ms (lead budget). 100 µs is generous.
    assert per_call_us < 100.0, f"{per_call_us:.1f} µs/call exceeds 100 µs"
