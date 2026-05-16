"""News direction classifier for XAUUSD — surprise → with/against/neutral.

A pure-function adapter sitting between :mod:`smc.hedgerock.news_engine`
(which already classifies *intensity* via FF red dots) and the decision
server (which needs a `direction` axis to fold into the news context of
:class:`smc.hedgerock.schemas.SignalEnvelope`).

This module deliberately **does NOT** re-classify intensity — that is
news_engine's job and re-doing it here would split the source of truth
across two modules. We add one new dimension: *direction* relative to
the EA's current exposure.

Rule of thumb (XAUUSD-specific, see ``hedgerock-redflag-mapping.md §3``):

- USD strengthens (CPI/NFP/FOMC beat) → XAUUSD *falls*.
  - Long position → ``"against"`` (tighten TP, halt new entries).
  - Short position → ``"with"``.
  - Flat → ``"neutral"``.
- USD weakens → mirror of above.
- EUR data is anti-correlated to USD impact and **half-weighted**:
  the EUR threshold is doubled so a similarly-sized release on EUR
  moves the classification only half as often (lead spec: "影响减半").
- Currencies other than USD/EUR are out of XAUUSD-PoC scope: surprise
  is computed but direction is forced to ``"neutral"``.
- Missing actual or forecast → ``surprise_score=None`` and
  ``direction="neutral"``. We never guess a direction for an
  upcoming/uncalculated release.

Versioning:

The output carries ``classifier_version`` so the decision server can
audit which rule book produced a given decision. Bump the version
string when changing thresholds or polarity rules so a backtest
re-classifying historical events with a new ruleset is detectable in
the journal.

Performance: pure arithmetic; per-call latency well below 1 ms. The
decision server may classify 10+ events per OnTimer cycle — keep this
function allocation-free and dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from smc.hedgerock.news_engine import NewsEvent
from smc.hedgerock.schemas import NewsDirection


__all__ = [
    "CLASSIFIER_VERSION",
    "ClassificationSource",
    "DEFAULT_SURPRISE_THRESHOLD",
    "DEFAULT_SENTIMENT_THRESHOLD",
    "EUR_INFLUENCE_FACTOR",
    "ExposureDirection",
    "NEGATIVE_SURPRISE_KEYWORDS",
    "NewsClassification",
    "classify_for_xauusd",
    "classify_with_sentiment",
    "compute_surprise_score",
]


CLASSIFIER_VERSION: Final[str] = "v1.0.0"
"""Bump on rule/threshold changes so journals can be matched to a ruleset."""


DEFAULT_SENTIMENT_THRESHOLD: Final[float] = 0.20
"""LLM sentiment magnitude below which we treat the score as inconclusive.

|score| in [-0.20, +0.20] yields a ``"neutral"`` classification even
when sentiment data is available — small biases are noise, not edge.
"""


DEFAULT_SURPRISE_THRESHOLD: Final[float] = 0.10
"""Relative surprise (10%) below which the release is "in line".

Surprise is computed as ``(actual - forecast) / forecast``.  A 10%
threshold filters typical noise releases that come in within forecast.
"""


EUR_INFLUENCE_FACTOR: Final[float] = 0.5
"""How much an EUR release's surprise weighs vs a USD release.

Lead spec: EUR data on XAUUSD is "影响减半"
(half-weighted compared to USD). Implemented by doubling the effective
threshold for EUR events: a 20% EUR surprise has the same trigger
strength as a 10% USD surprise.
"""


# Series where a *higher* actual is *bad* for the home currency, and
# the polarity of the surprise must be flipped before mapping to
# currency strength.  Examples:
#   - Unemployment Rate up   → USD weakens
#   - Initial Jobless Claims up → USD weakens
#   - Trade Balance more negative → currency weakens (deficit widens)
#
# These are matched by case-insensitive substring on the event name to
# keep the table compact; FF event names are stable enough that this
# is not a maintenance burden.
NEGATIVE_SURPRISE_KEYWORDS: Final[tuple[str, ...]] = (
    "unemployment rate",
    "jobless claims",
    "trade balance",
    "trade deficit",
)


# ``Literal`` covers the three legal exposure states the EA reports.
ExposureDirection = Literal["long", "short", "flat"]


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


ClassificationSource = Literal["rule-based", "sentiment-driven"]
"""Audit tag identifying which path produced the classification.

- ``rule-based``: the deterministic ``classify_for_xauusd`` rules ran.
- ``sentiment-driven``: an LLM sentiment cache hit refined the
  direction decision (Phase 5 Stage D).

Default is ``"rule-based"`` so legacy callers (Phase 3 wiring) get the
same audit tag they always had.
"""


@dataclass(frozen=True)
class NewsClassification:
    """Classification result for a single ``NewsEvent``.

    Fields mirror lead's task #24 contract verbatim. The decision
    server folds ``direction`` into
    :class:`smc.hedgerock.schemas.SignalEnvelope.news_direction`;
    the rest of the fields support audit / journal logging.

    Phase 5 Stage D adds ``classification_source`` so audit logs can
    distinguish rule-based vs sentiment-driven decisions. The field
    defaults to ``"rule-based"`` so callers built before Stage D get
    the historically expected tag.
    """

    event: NewsEvent
    direction: NewsDirection
    surprise_score: float | None
    impact_currency: str
    classifier_version: str
    classification_source: ClassificationSource = "rule-based"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def compute_surprise_score(
    *, actual: float | None, forecast: float | None
) -> float | None:
    """Return ``(actual - forecast) / forecast`` (relative surprise).

    Returns ``None`` if either value is missing — we never invent a
    surprise on an unreleased event. When ``forecast`` is exactly zero
    the relative form is undefined; we substitute 1.0 in the denominator
    to surface raw surprise without producing ``inf`` (rare but possible
    on series like trade balance prints).
    """
    if actual is None or forecast is None:
        return None
    denom = abs(forecast) if abs(forecast) > 0 else 1.0
    return (actual - forecast) / denom


def _is_negative_surprise_event(event_name: str) -> bool:
    """Case-insensitive substring scan against ``NEGATIVE_SURPRISE_KEYWORDS``."""
    lower = event_name.lower()
    return any(kw in lower for kw in NEGATIVE_SURPRISE_KEYWORDS)


def _exposure_to_sign(exposure: ExposureDirection) -> int:
    """Map ``"long" | "short" | "flat"`` to ``+1 | -1 | 0``."""
    if exposure == "long":
        return 1
    if exposure == "short":
        return -1
    return 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def classify_with_sentiment(
    event: NewsEvent,
    *,
    current_exposure_direction: ExposureDirection,
    sentiment_score: float,
    sentiment_threshold: float = DEFAULT_SENTIMENT_THRESHOLD,
) -> NewsClassification:
    """Sentiment-driven classification path (Phase 5 Stage D).

    Used when the LLM sentiment cache has a fresh score for this
    event. Direction logic mirrors the rule-based path's polarity rules
    (USD strong → XAU down, EUR strong → XAU up) but relies on the
    LLM's hawkish/dovish bias rather than ``actual - forecast``.

    Args:
        event: The news event to classify.
        current_exposure_direction: ``"long" | "short" | "flat"``.
        sentiment_score: LLM-emitted score in [-1, +1]. Sign convention
            matches ``LLMSentimentClassifier``: positive = currency-
            supportive (hawkish for USD = XAU bearish).
        sentiment_threshold: |score| below this falls back to
            ``"neutral"``.

    Returns:
        :class:`NewsClassification` with ``classification_source =
        "sentiment-driven"``. The ``surprise_score`` field is reused
        to surface the sentiment value (so journals can plot both
        rule-based surprise and sentiment-driven score on the same
        axis).

    Notes:
        - Out-of-scope currencies (not USD/EUR) → ``"neutral"``.
        - Flat exposure → ``"neutral"``.
        - Threshold check identical structure to rule-based path so
          journals see consistent neutral-band treatment.
    """
    if sentiment_threshold <= 0.0:
        raise ValueError(
            f"sentiment_threshold must be positive, got {sentiment_threshold}"
        )
    if not -1.0 <= sentiment_score <= 1.0:
        raise ValueError(
            f"sentiment_score must lie in [-1, +1], got {sentiment_score}"
        )
    if current_exposure_direction not in ("long", "short", "flat"):
        raise ValueError(
            f"current_exposure_direction must be 'long' | 'short' | "
            f"'flat', got {current_exposure_direction!r}"
        )

    currency = event.currency.upper()

    if currency not in {"USD", "EUR"}:
        return NewsClassification(
            event=event,
            direction="neutral",
            surprise_score=sentiment_score,
            impact_currency=currency,
            classifier_version=CLASSIFIER_VERSION,
            classification_source="sentiment-driven",
        )

    if abs(sentiment_score) < sentiment_threshold:
        return NewsClassification(
            event=event,
            direction="neutral",
            surprise_score=sentiment_score,
            impact_currency=currency,
            classifier_version=CLASSIFIER_VERSION,
            classification_source="sentiment-driven",
        )

    currency_sign = 1 if sentiment_score > 0 else -1
    if currency == "USD":
        expected_xau_sign = -currency_sign
    else:  # EUR
        expected_xau_sign = currency_sign

    exposure_sign = _exposure_to_sign(current_exposure_direction)
    if exposure_sign == 0:
        direction: NewsDirection = "neutral"
    elif expected_xau_sign == exposure_sign:
        direction = "with"
    else:
        direction = "against"

    return NewsClassification(
        event=event,
        direction=direction,
        surprise_score=sentiment_score,
        impact_currency=currency,
        classifier_version=CLASSIFIER_VERSION,
        classification_source="sentiment-driven",
    )


def classify_for_xauusd(
    event: NewsEvent,
    *,
    current_exposure_direction: ExposureDirection,
    surprise_threshold: float = DEFAULT_SURPRISE_THRESHOLD,
    sentiment_classifier=None,
) -> NewsClassification:
    """Classify a single news event's directional impact on XAUUSD positions.

    Algorithm (deterministic, no LLM):

    1. Compute relative surprise; if either side is missing →
       ``direction="neutral"``, ``surprise_score=None``.
    2. Apply polarity flip for "higher = bad" series
       (:data:`NEGATIVE_SURPRISE_KEYWORDS`).
    3. Currencies outside ``{USD, EUR}`` are out of scope →
       ``direction="neutral"`` (surprise_score still surfaced for audit).
    4. Compare ``|adjusted_surprise|`` against the effective threshold
       (``threshold * 2`` for EUR per :data:`EUR_INFLUENCE_FACTOR`); if
       below, ``direction="neutral"``.
    5. Map currency-strength sign to XAUUSD price expectation:
       USD strong → XAU down, USD weak → XAU up. EUR is anti-correlated
       to USD on XAUUSD, so EUR strong → XAU up.
    6. Compare expected XAU move to exposure sign. Flat exposure
       always yields ``"neutral"``.

    Args:
        event: The :class:`NewsEvent` to classify (intensity already
            set by news_engine — we do not change it here).
        current_exposure_direction: Caller-supplied ``"long" | "short" |
            "flat"``. Computed by the decision server from the EA's
            position summary.
        surprise_threshold: Minimum ``|relative_surprise|`` to count as
            directional. Lead default is 10%; EUR events use ``2x``.

    Returns:
        :class:`NewsClassification` with frozen fields suitable for
        journal logging or :class:`SignalEnvelope` injection.

    Raises:
        ValueError: If ``surprise_threshold`` is non-positive or
            ``current_exposure_direction`` is invalid (Literal guards
            most of this at type-check time, but we double-check at
            runtime for safety).
    """
    if surprise_threshold <= 0.0:
        raise ValueError(
            f"surprise_threshold must be positive, got {surprise_threshold}"
        )
    if current_exposure_direction not in ("long", "short", "flat"):
        raise ValueError(
            f"current_exposure_direction must be 'long' | 'short' | "
            f"'flat', got {current_exposure_direction!r}"
        )

    # Phase 5 Stage D: prefer LLM sentiment when a fresh cache entry
    # exists for this event. None / cache miss / stale → fall through
    # to the rule-based path below (100 % rule-based fallback by spec).
    if sentiment_classifier is not None:
        sentiment = sentiment_classifier.get_sentiment(event.event_id)
        if sentiment is not None:
            return classify_with_sentiment(
                event,
                current_exposure_direction=current_exposure_direction,
                sentiment_score=sentiment.score,
            )

    surprise_score = compute_surprise_score(
        actual=event.actual, forecast=event.forecast
    )

    # 1. Missing actual/forecast → never assert a direction.
    if surprise_score is None:
        return NewsClassification(
            event=event,
            direction="neutral",
            surprise_score=None,
            impact_currency=event.currency.upper(),
            classifier_version=CLASSIFIER_VERSION,
        )

    currency = event.currency.upper()

    # 3. Out-of-scope currencies — surface the score for auditing but
    # refuse to translate it into a XAUUSD-specific direction.
    if currency not in {"USD", "EUR"}:
        return NewsClassification(
            event=event,
            direction="neutral",
            surprise_score=surprise_score,
            impact_currency=currency,
            classifier_version=CLASSIFIER_VERSION,
        )

    # 2. Polarity flip — "higher is bad" series weaken the home currency
    # on a positive raw surprise, so flip the score *before* the threshold
    # check so |score| still reflects "how far from forecast" the print
    # was. The sign drives currency strength below.
    adjusted_score = surprise_score
    if _is_negative_surprise_event(event.name):
        adjusted_score = -adjusted_score

    # 4. Threshold check — EUR events are half-weighted (2x threshold).
    effective_threshold = (
        surprise_threshold / EUR_INFLUENCE_FACTOR
        if currency == "EUR"
        else surprise_threshold
    )
    if abs(adjusted_score) < effective_threshold:
        return NewsClassification(
            event=event,
            direction="neutral",
            surprise_score=surprise_score,
            impact_currency=currency,
            classifier_version=CLASSIFIER_VERSION,
        )

    # 5. Map currency strength to expected XAUUSD price move.
    #    USD strong  (+) → XAU down (-1)
    #    USD weak    (-) → XAU up   (+1)
    #    EUR strong  (+) → XAU up   (+1)  (anti-correlated to USD)
    #    EUR weak    (-) → XAU down (-1)
    currency_sign = 1 if adjusted_score > 0 else -1
    if currency == "USD":
        expected_xau_sign = -currency_sign
    else:  # EUR
        expected_xau_sign = currency_sign

    # 6. Compare against exposure.
    exposure_sign = _exposure_to_sign(current_exposure_direction)
    if exposure_sign == 0:
        direction: NewsDirection = "neutral"
    elif expected_xau_sign == exposure_sign:
        direction = "with"
    else:
        direction = "against"

    return NewsClassification(
        event=event,
        direction=direction,
        surprise_score=surprise_score,
        impact_currency=currency,
        classifier_version=CLASSIFIER_VERSION,
    )
