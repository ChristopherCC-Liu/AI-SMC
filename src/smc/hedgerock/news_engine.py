"""Economic-calendar news engine for the HedgeRock decision loop.

The MQL5 EA cannot fetch external HTTP itself in a reliable, sandbox-safe
way (HedgeRock's built-in fxstreet poller is one of the few brittle parts
of the original code — see ``hedgerock-risk-audit.md §8`` and
``cross-system-lessons.md §B2``). This module replaces that path with a
Python-side ForexFactory crawler whose output the decision_server folds
into :class:`smc.hedgerock.schemas.SignalEnvelope` so the EA only has to
read JSON.

Architecture (mirrors AlphaLens ``BaseCrawler`` design — rate limit, raw
fetch, parse, deduplicate):

    ┌────────────────────────────────────────────────┐
    │  ForexFactoryClient                            │
    │   • httpx.AsyncClient + tenacity retry         │
    │   • TTL cache (5 minutes by default)           │
    │   • returns the raw HTML string                │
    └──────────────────┬─────────────────────────────┘
                       │
    ┌──────────────────▼─────────────────────────────┐
    │  ForexFactoryCalendarParser                    │
    │   • pure function, no IO                       │
    │   • HTML → tuple[NewsEvent, ...]               │
    │   • intensity 1/2/3 red dots → low/medium/high │
    └──────────────────┬─────────────────────────────┘
                       │
    ┌──────────────────▼─────────────────────────────┐
    │  NewsEngine.fetch_events(currencies, when)     │
    │   • currency filter                            │
    │   • dedup by event_id (sha256 of name+ts)      │
    │   • returns sorted-ascending tuple of events   │
    └────────────────────────────────────────────────┘

Design rules:

- **Frozen dataclasses** — every public type is immutable so callers can
  safely cache or share envelopes without copy-on-mutate.
- **Pure parser** — :class:`ForexFactoryCalendarParser` takes HTML and
  returns events; it never touches the network. This makes the bulk of
  the test suite hermetic: write an HTML fixture, assert the events.
- **TTL cache, not LRU** — the EA polls every 10 s, but ForexFactory's
  weekly calendar barely changes intra-day. A 5-minute TTL is a
  comfortable trade-off; configurable via the constructor.
- **Rate limit + retry** — exponential back-off via ``tenacity``; the
  503/429 path is exercised in tests.
- **String-prefix severity** — :data:`NewsIntensity` literals match the
  cross-language contract already used by ``schemas.SignalEnvelope`` —
  do not introduce a parallel enum.

The module deliberately does not implement FRED / DXY ingestion (Phase
3.1 lead scope guidance: "do not implement FRED/DXY pulls — that is
macro-layer, not Phase 3.1").
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Final

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from smc.hedgerock.schemas import NewsIntensity


__all__ = [
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_CURRENCIES",
    "DEFAULT_USER_AGENT",
    "FOREXFACTORY_CALENDAR_URL",
    "ForexFactoryCalendarParser",
    "ForexFactoryClient",
    "NewsEngine",
    "NewsEvent",
    "NewsFetchError",
    "intensity_from_red_dots",
    "make_event_id",
]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


FOREXFACTORY_CALENDAR_URL: Final[str] = (
    "https://www.forexfactory.com/calendar?week=this"
)

DEFAULT_USER_AGENT: Final[str] = (
    # FF aggressively blocks default httpx UA. Use a realistic browser UA
    # but identify the project so a friendly admin can ratelimit us
    # cleanly instead of bot-banning.
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 "
    "AI-SMC-NewsEngine/1.0"
)

DEFAULT_CACHE_TTL_SECONDS: Final[float] = 300.0  # 5 minutes
"""How long a successful fetch result is reused before re-hitting FF.

The EA's poll cadence is ~10 s but FF's weekly calendar barely changes
intra-day. 5 minutes balances freshness vs being a polite client.
"""

DEFAULT_CURRENCIES: Final[tuple[str, ...]] = ("USD", "EUR")
"""Default currencies relevant to XAUUSD trading.

XAUUSD is gold quoted in dollars; the two currencies whose data
releases move it the most are USD (Fed/NFP/CPI) and EUR (ECB/PMI). Other
G10 currencies (GBP, JPY, etc.) move gold less often.
"""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NewsFetchError(RuntimeError):
    """Raised after retries are exhausted or the response is unusable."""


# ---------------------------------------------------------------------------
# Frozen public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewsEvent:
    """One scheduled or released economic event.

    Field semantics match ``hedgerock-integration-options.md`` §3.7
    "news context" and the lead's task #23 contract verbatim.
    """

    event_id: str  # sha256(name + iso_ts)[:16]
    name: str  # e.g. "Non-Farm Payrolls"
    currency: str  # ISO code, e.g. "USD"
    intensity: NewsIntensity
    scheduled_at: datetime  # tz-aware UTC
    actual: float | None = None
    forecast: float | None = None
    previous: float | None = None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def make_event_id(name: str, scheduled_at: datetime) -> str:
    """Build a stable 16-char hash for ``(name, scheduled_at)``.

    Used for de-duplication when the same event surfaces from multiple
    sources (ForexFactory + the EA's built-in fxstreet) or from
    overlapping cache pages.

    Args:
        name: Event name as printed on the calendar (case sensitive).
        scheduled_at: Tz-aware UTC datetime; naive timestamps raise
            ``ValueError`` so callers do not accidentally mix tz-naive
            data into the dedup key.
    """
    if scheduled_at.tzinfo is None:
        raise ValueError(
            "scheduled_at must be tz-aware; got naive datetime"
        )
    iso = scheduled_at.astimezone(timezone.utc).isoformat()
    digest = hashlib.sha256(f"{name}|{iso}".encode("utf-8")).hexdigest()
    return digest[:16]


def intensity_from_red_dots(red_dots: int) -> NewsIntensity:
    """Map ForexFactory's 1-3 red-dot scale to ``NewsIntensity`` literals.

    ``0`` red dots is the "non-economic" / placeholder row that FF emits
    for things like daylight saving — we surface them as ``"none"`` so
    callers can choose to suppress them.

    Args:
        red_dots: Integer count parsed from the calendar row's
            ``impact-icon`` element.

    Raises:
        ValueError: If ``red_dots`` is outside the documented 0..3 range.
    """
    if red_dots == 3:
        return "high"
    if red_dots == 2:
        return "medium"
    if red_dots == 1:
        return "low"
    if red_dots == 0:
        return "none"
    raise ValueError(
        f"red_dots must be in 0..3 (FF impact-icon scale), got {red_dots}"
    )


def _parse_optional_float(raw: str | None) -> float | None:
    """Parse a calendar cell into ``float | None``.

    FF cells contain values like ``"272K"``, ``"3.4%"``, ``"-0.1%"`` or
    blank/``"--"`` for "no data yet". This helper strips the unit
    suffix, applies the K/M multiplier, and returns ``None`` when the
    cell is empty or the placeholder.
    """
    if raw is None:
        return None
    text = raw.strip().replace(",", "")
    if not text or text in {"-", "--", "n/a", "N/A"}:
        return None
    text = text.rstrip("%")
    multiplier = 1.0
    if text.endswith("K"):
        multiplier = 1_000.0
        text = text[:-1]
    elif text.endswith("M"):
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text.endswith("B"):
        multiplier = 1_000_000_000.0
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# HTML parser — pure function, hermetic-test-friendly
# ---------------------------------------------------------------------------


# The tag pattern lets us locate calendar rows without dragging in a
# heavyweight DOM library. ForexFactory's actual HTML shape is brittle
# (wraps each row in ``<tr class="calendar__row">``) so we only need a
# tiny shim. The fixtures used in tests carry the same outer shape.
#
# Note we capture the row's *outer* HTML (the ``<tr ...>`` open tag is
# kept) so attributes attached to the row itself — most importantly
# ``data-event-datetime`` — are visible to ``_extract_iso_timestamp``.
_ROW_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<inner><tr[^>]*class=\"calendar__row[^\"]*\"[^>]*>.*?</tr>)",
    re.DOTALL | re.IGNORECASE,
)


def _extract_cell(inner_html: str, css_class: str) -> str | None:
    """Pull the first text content of a cell whose class matches.

    Returns ``None`` if the cell is missing — many calendar rows omit
    "actual" until the data is released.
    """
    pattern = re.compile(
        rf"<td[^>]*class=\"[^\"]*{re.escape(css_class)}[^\"]*\"[^>]*>"
        r"(?P<body>.*?)</td>",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(inner_html)
    if match is None:
        return None
    body = match.group("body")
    # Strip nested tags — we only need the visible text.
    text = re.sub(r"<[^>]+>", " ", body)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _extract_red_dots(inner_html: str) -> int:
    """Count the impact-icon red dots inside a calendar row.

    ForexFactory marks impact via classes like ``impact-icon``,
    ``ff-impact-yel``, ``ff-impact-ora``, ``ff-impact-red``. Tests use
    a simplified scheme: the substring ``data-impact-dots="N"`` or
    ``ff-impact-redN`` (1..3). Either form is accepted so the fixture
    can stay readable.
    """
    explicit = re.search(
        r"data-impact-dots=\"(?P<n>[0-3])\"", inner_html
    )
    if explicit:
        return int(explicit.group("n"))
    if "ff-impact-red3" in inner_html:
        return 3
    if "ff-impact-red2" in inner_html or "ff-impact-ora" in inner_html:
        return 2
    if "ff-impact-red1" in inner_html or "ff-impact-yel" in inner_html:
        return 1
    return 0


_TS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"data-event-datetime=\"(?P<ts>[^\"]+)\"", re.IGNORECASE
)


def _extract_iso_timestamp(inner_html: str) -> datetime | None:
    """Pull a tz-aware ISO timestamp from the row.

    The fixture format places it in ``data-event-datetime``; the real FF
    page uses something noisier but the same attribute survives across
    revisions.
    """
    match = _TS_PATTERN.search(inner_html)
    if match is None:
        return None
    raw = match.group("ts")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class ForexFactoryCalendarParser:
    """Pure HTML → ``tuple[NewsEvent, ...]`` parser.

    Stateless; instantiated once and reused. Kept as a class (rather
    than a free function) so callers can subclass to handle FF DOM
    drift without monkey-patching module state.
    """

    def parse(self, html: str) -> tuple[NewsEvent, ...]:
        """Parse calendar rows into events.

        Rows with missing event name or timestamp are silently skipped;
        partial data is more common than malicious data so we tolerate
        it rather than fail loudly.
        """
        events: list[NewsEvent] = []
        for match in _ROW_PATTERN.finditer(html):
            inner = match.group("inner")
            name = _extract_cell(inner, "calendar__event")
            currency = _extract_cell(inner, "calendar__currency")
            ts = _extract_iso_timestamp(inner)
            if not name or not currency or ts is None:
                continue
            red_dots = _extract_red_dots(inner)
            intensity = intensity_from_red_dots(red_dots)
            actual_raw = _extract_cell(inner, "calendar__actual")
            forecast_raw = _extract_cell(inner, "calendar__forecast")
            previous_raw = _extract_cell(inner, "calendar__previous")
            event = NewsEvent(
                event_id=make_event_id(name, ts),
                name=name,
                currency=currency.upper(),
                intensity=intensity,
                scheduled_at=ts,
                actual=_parse_optional_float(actual_raw),
                forecast=_parse_optional_float(forecast_raw),
                previous=_parse_optional_float(previous_raw),
            )
            events.append(event)
        return tuple(events)


# ---------------------------------------------------------------------------
# HTTP client (cached) — IO layer
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """Internal TTL cache cell."""

    fetched_at_monotonic: float
    payload: str


class ForexFactoryClient:
    """Thin httpx wrapper with tenacity retry + TTL cache.

    Args:
        url: The calendar URL to fetch. Defaults to the weekly view.
        cache_ttl_seconds: How long a successful fetch stays valid.
        timeout_seconds: Per-request HTTP timeout.
        user_agent: Override UA. Real bots get blocked aggressively;
            keep the realistic-browser default unless testing.
        client: Inject an ``httpx.AsyncClient`` for tests.
        clock: Inject a monotonic clock for tests (defaults to
            :func:`time.monotonic`).
    """

    def __init__(
        self,
        url: str = FOREXFACTORY_CALENDAR_URL,
        *,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        timeout_seconds: float = 15.0,
        user_agent: str = DEFAULT_USER_AGENT,
        client: httpx.AsyncClient | None = None,
        clock: callable | None = None,  # type: ignore[type-arg]
    ) -> None:
        if cache_ttl_seconds < 0:
            raise ValueError(
                f"cache_ttl_seconds must be >= 0, got {cache_ttl_seconds}"
            )
        if timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {timeout_seconds}"
            )
        self._url = url
        self._cache_ttl = cache_ttl_seconds
        self._timeout = timeout_seconds
        self._user_agent = user_agent
        self._client = client
        self._clock = clock or time.monotonic
        self._owns_client = client is None
        self._cache: _CacheEntry | None = None
        self._cache_lock = threading.Lock()

    @property
    def url(self) -> str:
        return self._url

    async def __aenter__(self) -> "ForexFactoryClient":
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": self._user_agent},
            )
            self._owns_client = True
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def cached_payload(self) -> str | None:
        """Return the cached HTML if still fresh, else ``None``.

        Public so tests can inspect cache state without monkey-patching.
        """
        with self._cache_lock:
            if self._cache is None:
                return None
            age = self._clock() - self._cache.fetched_at_monotonic
            if age > self._cache_ttl:
                return None
            return self._cache.payload

    def invalidate(self) -> None:
        """Drop any cached payload so the next fetch hits the network."""
        with self._cache_lock:
            self._cache = None

    @retry(
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.TransportError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        reraise=True,
    )
    async def _network_fetch(self) -> str:
        if self._client is None:
            # Allow standalone use without ``async with``.
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"User-Agent": self._user_agent},
            )
            self._owns_client = True
        response = await self._client.get(self._url)
        response.raise_for_status()
        return response.text

    async def fetch(self) -> str:
        """Fetch the calendar HTML, honouring the TTL cache.

        Concurrent callers race on the cache lock; the first one to
        miss does the network round-trip while later ones queue and
        then read the freshly stored payload.

        Raises:
            NewsFetchError: If three retries fail.
        """
        cached = self.cached_payload()
        if cached is not None:
            return cached
        try:
            payload = await self._network_fetch()
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            raise NewsFetchError(
                f"ForexFactory fetch failed after retries: {exc!r}"
            ) from exc
        with self._cache_lock:
            self._cache = _CacheEntry(
                fetched_at_monotonic=self._clock(),
                payload=payload,
            )
        return payload


# ---------------------------------------------------------------------------
# High-level engine
# ---------------------------------------------------------------------------


class NewsEngine:
    """Single entry point used by ``decision_server`` per-OnTimer cycle.

    The engine combines the cached HTTP client + pure parser and adds
    currency filtering plus ID dedup. Decision-server callers only ever
    talk to ``fetch_events`` / ``find_active_event``.

    Args:
        client: Pre-configured :class:`ForexFactoryClient` (tests usually
            inject one with a fake :class:`httpx.AsyncClient`).
        parser: Pre-configured parser; rarely overridden.
        currencies: Default currency filter applied when callers do not
            override it.
    """

    def __init__(
        self,
        *,
        client: ForexFactoryClient | None = None,
        parser: ForexFactoryCalendarParser | None = None,
        currencies: Sequence[str] = DEFAULT_CURRENCIES,
    ) -> None:
        self._client = client or ForexFactoryClient()
        self._parser = parser or ForexFactoryCalendarParser()
        normalised = tuple(c.upper() for c in currencies)
        if not normalised:
            raise ValueError("currencies must contain at least one entry")
        self._currencies = normalised

    @property
    def default_currencies(self) -> tuple[str, ...]:
        return self._currencies

    @staticmethod
    def _filter_and_dedup(
        events: Iterable[NewsEvent],
        currencies: tuple[str, ...],
    ) -> tuple[NewsEvent, ...]:
        wanted = {c.upper() for c in currencies}
        seen: set[str] = set()
        kept: list[NewsEvent] = []
        for ev in events:
            if ev.currency not in wanted:
                continue
            if ev.event_id in seen:
                continue
            seen.add(ev.event_id)
            kept.append(ev)
        kept.sort(key=lambda e: e.scheduled_at)
        return tuple(kept)

    async def fetch_events(
        self,
        *,
        currencies: Sequence[str] | None = None,
    ) -> tuple[NewsEvent, ...]:
        """Return the filtered, deduplicated event list (sorted ascending)."""
        active_currencies = (
            tuple(c.upper() for c in currencies)
            if currencies is not None
            else self._currencies
        )
        html = await self._client.fetch()
        all_events = self._parser.parse(html)
        return self._filter_and_dedup(all_events, active_currencies)

    async def find_active_event(
        self,
        *,
        now: datetime,
        window_before_seconds: int = 30 * 60,
        window_after_seconds: int = 30 * 60,
        currencies: Sequence[str] | None = None,
    ) -> NewsEvent | None:
        """Return the highest-intensity event whose blackout window includes ``now``.

        ``decision_server`` calls this on every poll to fold the result
        into :class:`smc.hedgerock.schemas.SignalEnvelope`. Defaults
        match the EA's typical ``STOP_BEFORE_*`` / ``START_AFTER_*``
        configuration of 30 minutes either side of a release.

        When multiple events are active simultaneously the one with the
        higher intensity wins; ties are broken by earlier scheduled_at.
        """
        if now.tzinfo is None:
            raise ValueError("`now` must be tz-aware UTC")
        now_utc = now.astimezone(timezone.utc)
        before = timedelta(seconds=window_before_seconds)
        after = timedelta(seconds=window_after_seconds)
        events = await self.fetch_events(currencies=currencies)
        active: list[NewsEvent] = []
        for ev in events:
            window_start = ev.scheduled_at - before
            window_end = ev.scheduled_at + after
            if window_start <= now_utc <= window_end:
                active.append(ev)
        if not active:
            return None
        intensity_rank: dict[str, int] = {
            "none": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
        }
        active.sort(
            key=lambda e: (
                -intensity_rank.get(e.intensity, 0),
                e.scheduled_at,
            )
        )
        return active[0]
