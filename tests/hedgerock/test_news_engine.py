"""Tests for ``smc.hedgerock.news_engine``.

Three layers, in order of strictness:

1. **Pure helpers** (``intensity_from_red_dots``, ``make_event_id``,
   ``_parse_optional_float``) — unit, no IO.
2. **Parser** — feeds the synthetic fixture
   ``tests/fixtures/forexfactory_2024_03_06.html`` and asserts the seven
   expected events come out with correct fields.
3. **Client + Engine** — uses an injected ``httpx.AsyncClient`` against
   an ``httpx.MockTransport`` so we can drive cache TTL, retry on
   transient failures, and currency filtering deterministically.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from smc.hedgerock.news_engine import (
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_CURRENCIES,
    FOREXFACTORY_CALENDAR_URL,
    ForexFactoryCalendarParser,
    ForexFactoryClient,
    NewsEngine,
    NewsEvent,
    NewsFetchError,
    intensity_from_red_dots,
    make_event_id,
)


_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "forexfactory_2024_03_06.html"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dots", "expected"),
    [(3, "high"), (2, "medium"), (1, "low"), (0, "none")],
)
def test_intensity_mapping_matches_forexfactory_scale(
    dots: int, expected: str
) -> None:
    assert intensity_from_red_dots(dots) == expected


@pytest.mark.unit
@pytest.mark.parametrize("bad", [-1, 4, 99])
def test_intensity_rejects_out_of_range(bad: int) -> None:
    with pytest.raises(ValueError, match="0..3"):
        intensity_from_red_dots(bad)


@pytest.mark.unit
def test_make_event_id_is_stable_for_same_inputs() -> None:
    ts = datetime(2024, 3, 6, 13, 30, tzinfo=timezone.utc)
    a = make_event_id("Non-Farm Payrolls", ts)
    b = make_event_id("Non-Farm Payrolls", ts)
    assert a == b
    assert len(a) == 16


@pytest.mark.unit
def test_make_event_id_changes_with_either_input() -> None:
    ts = datetime(2024, 3, 6, 13, 30, tzinfo=timezone.utc)
    base = make_event_id("Non-Farm Payrolls", ts)
    different_name = make_event_id("ADP Employment Change", ts)
    different_time = make_event_id(
        "Non-Farm Payrolls", ts + timedelta(minutes=1)
    )
    assert base != different_name
    assert base != different_time


@pytest.mark.unit
def test_make_event_id_normalises_timezone_offsets() -> None:
    """Same UTC instant in two zones must yield the same id."""
    utc = datetime(2024, 3, 6, 13, 30, tzinfo=timezone.utc)
    pst = utc.astimezone(timezone(timedelta(hours=-8)))
    assert make_event_id("CPI", utc) == make_event_id("CPI", pst)


@pytest.mark.unit
def test_make_event_id_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        make_event_id("CPI", datetime(2024, 3, 6, 13, 30))


# ---------------------------------------------------------------------------
# Parser — fixture-driven
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_html() -> str:
    return _FIXTURE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def parser() -> ForexFactoryCalendarParser:
    return ForexFactoryCalendarParser()


@pytest.mark.unit
def test_parser_extracts_seven_documented_events(
    parser: ForexFactoryCalendarParser, fixture_html: str
) -> None:
    events = parser.parse(fixture_html)
    assert len(events) == 7
    names = [e.name for e in events]
    assert "Non-Farm Payrolls" in names
    assert "ADP Employment Change" in names
    assert "Retail Sales" in names
    assert "CPI YoY" in names
    assert "ECB Rate Decision" in names
    assert "Manufacturing PMI" in names
    assert "Daylight Saving Time" in names


@pytest.mark.unit
def test_parser_assigns_correct_intensity_per_red_dot_count(
    parser: ForexFactoryCalendarParser, fixture_html: str
) -> None:
    by_name = {e.name: e for e in parser.parse(fixture_html)}
    assert by_name["Non-Farm Payrolls"].intensity == "high"
    assert by_name["ADP Employment Change"].intensity == "medium"
    assert by_name["Retail Sales"].intensity == "medium"
    assert by_name["CPI YoY"].intensity == "high"
    assert by_name["ECB Rate Decision"].intensity == "high"
    assert by_name["Manufacturing PMI"].intensity == "low"
    assert by_name["Daylight Saving Time"].intensity == "none"


@pytest.mark.unit
def test_parser_normalises_numeric_cells(
    parser: ForexFactoryCalendarParser, fixture_html: str
) -> None:
    by_name = {e.name: e for e in parser.parse(fixture_html)}
    # "272K" → 272_000.0
    assert by_name["Non-Farm Payrolls"].actual == pytest.approx(272_000.0)
    # "3.4%" → 3.4
    assert by_name["CPI YoY"].actual == pytest.approx(3.4)
    # "-0.1%" → -0.1
    assert by_name["Retail Sales"].actual == pytest.approx(-0.1)
    # "" / "--" → None
    assert by_name["ECB Rate Decision"].actual is None
    assert by_name["Daylight Saving Time"].actual is None


@pytest.mark.unit
def test_parser_returns_tz_aware_utc_timestamps(
    parser: ForexFactoryCalendarParser, fixture_html: str
) -> None:
    for e in parser.parse(fixture_html):
        assert e.scheduled_at.tzinfo is not None
        assert e.scheduled_at.utcoffset() == timedelta(0)


@pytest.mark.unit
def test_parser_skips_rows_missing_name_or_timestamp(
    parser: ForexFactoryCalendarParser,
) -> None:
    html = (
        '<table><tr class="calendar__row" data-event-datetime="2024-03-06T13:30:00+00:00">'
        '<td class="calendar__currency">USD</td>'
        # name cell missing entirely
        "</tr></table>"
    )
    assert parser.parse(html) == ()


@pytest.mark.unit
def test_parser_handles_empty_html(parser: ForexFactoryCalendarParser) -> None:
    assert parser.parse("") == ()
    assert parser.parse("<html><body><p>nothing</p></body></html>") == ()


# ---------------------------------------------------------------------------
# ForexFactoryClient — cache + retry
# ---------------------------------------------------------------------------


def _build_mock_client(
    *,
    payloads: list[str | int],  # str = body; int = HTTP status code (5xx)
    capture: list[int] | None = None,
) -> httpx.AsyncClient:
    """Sequence of either HTML strings or status codes for transient failures."""

    state = {"i": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        idx = state["i"]
        if capture is not None:
            capture.append(idx)
        state["i"] += 1
        if idx >= len(payloads):
            return httpx.Response(200, text=payloads[-1] if payloads else "")  # type: ignore[arg-type]
        item = payloads[idx]
        if isinstance(item, int):
            return httpx.Response(item, text="")
        return httpx.Response(200, text=item)

    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


@pytest.mark.unit
def test_client_constructor_validates_args() -> None:
    with pytest.raises(ValueError, match="cache_ttl_seconds"):
        ForexFactoryClient(cache_ttl_seconds=-1.0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        ForexFactoryClient(timeout_seconds=0.0)


@pytest.mark.integration
def test_client_fetch_caches_payload_within_ttl() -> None:
    """Two calls inside the TTL window hit the network only once."""
    fixture = _FIXTURE_PATH.read_text(encoding="utf-8")
    counter: list[int] = []
    fake_now = {"t": 0.0}

    async def runner() -> None:
        client = ForexFactoryClient(
            cache_ttl_seconds=300.0,
            client=_build_mock_client(payloads=[fixture, fixture], capture=counter),
            clock=lambda: fake_now["t"],
        )
        first = await client.fetch()
        # Advance time *less* than TTL — second call should be a cache hit.
        fake_now["t"] += 60.0
        second = await client.fetch()
        assert first == second == fixture

    asyncio.run(runner())
    assert counter == [0]  # only one network call


@pytest.mark.integration
def test_client_fetch_refreshes_after_ttl_expires() -> None:
    fixture = _FIXTURE_PATH.read_text(encoding="utf-8")
    counter: list[int] = []
    fake_now = {"t": 0.0}

    async def runner() -> None:
        client = ForexFactoryClient(
            cache_ttl_seconds=300.0,
            client=_build_mock_client(
                payloads=[fixture, fixture], capture=counter
            ),
            clock=lambda: fake_now["t"],
        )
        await client.fetch()
        # Cross the TTL boundary.
        fake_now["t"] += 301.0
        await client.fetch()

    asyncio.run(runner())
    assert counter == [0, 1]


@pytest.mark.integration
def test_client_invalidate_drops_cache() -> None:
    fixture = _FIXTURE_PATH.read_text(encoding="utf-8")
    counter: list[int] = []

    async def runner() -> None:
        client = ForexFactoryClient(
            cache_ttl_seconds=300.0,
            client=_build_mock_client(
                payloads=[fixture, fixture], capture=counter
            ),
        )
        await client.fetch()
        client.invalidate()
        await client.fetch()

    asyncio.run(runner())
    assert counter == [0, 1]


@pytest.mark.integration
def test_client_retries_on_503_then_succeeds() -> None:
    fixture = _FIXTURE_PATH.read_text(encoding="utf-8")
    counter: list[int] = []

    async def runner() -> str:
        client = ForexFactoryClient(
            cache_ttl_seconds=300.0,
            client=_build_mock_client(payloads=[503, 503, fixture], capture=counter),
        )
        return await client.fetch()

    payload = asyncio.run(runner())
    assert payload == fixture
    # Three attempts: two 503 + one OK
    assert counter == [0, 1, 2]


@pytest.mark.integration
def test_client_raises_news_fetch_error_after_retries_exhausted() -> None:
    counter: list[int] = []

    async def runner() -> None:
        client = ForexFactoryClient(
            cache_ttl_seconds=300.0,
            client=_build_mock_client(payloads=[503, 503, 503], capture=counter),
        )
        await client.fetch()

    with pytest.raises(NewsFetchError, match="ForexFactory fetch failed"):
        asyncio.run(runner())
    assert counter == [0, 1, 2]


# ---------------------------------------------------------------------------
# NewsEngine — currency filter + dedup + active-event window
# ---------------------------------------------------------------------------


def _engine_with_fixture(currencies: tuple[str, ...] = DEFAULT_CURRENCIES) -> NewsEngine:
    fixture = _FIXTURE_PATH.read_text(encoding="utf-8")
    client = ForexFactoryClient(
        cache_ttl_seconds=300.0,
        client=_build_mock_client(payloads=[fixture] * 5),
    )
    return NewsEngine(client=client, currencies=currencies)


@pytest.mark.unit
def test_engine_constructor_rejects_empty_currencies() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        NewsEngine(currencies=())


@pytest.mark.integration
def test_engine_default_filter_keeps_only_usd_and_eur() -> None:
    """GBP / unknown currencies are dropped under the default ``("USD","EUR")``."""

    async def runner() -> tuple[NewsEvent, ...]:
        engine = _engine_with_fixture()
        return await engine.fetch_events()

    events = asyncio.run(runner())
    currencies = {e.currency for e in events}
    assert currencies == {"USD", "EUR"}
    # 4 USD (NFP + ADP + CPI + Daylight Saving) + 2 EUR (Retail + ECB)
    assert len(events) == 6


@pytest.mark.integration
def test_engine_currency_override_picks_only_one_currency() -> None:
    async def runner() -> tuple[NewsEvent, ...]:
        engine = _engine_with_fixture()
        return await engine.fetch_events(currencies=("USD",))

    events = asyncio.run(runner())
    assert all(e.currency == "USD" for e in events)
    # NFP + ADP + CPI + Daylight Saving Time = 4 USD rows.
    assert len(events) == 4
    # Note: fixture's Daylight Saving row is USD with intensity=none — it
    # is *not* filtered out by currency, only by upstream callers if they
    # care. We surface it so callers can choose.
    assert "Daylight Saving Time" in {e.name for e in events}


@pytest.mark.integration
def test_engine_currency_override_supports_gbp() -> None:
    async def runner() -> tuple[NewsEvent, ...]:
        engine = _engine_with_fixture()
        return await engine.fetch_events(currencies=("GBP",))

    events = asyncio.run(runner())
    assert {e.currency for e in events} == {"GBP"}
    assert len(events) == 1
    assert events[0].name == "Manufacturing PMI"


@pytest.mark.integration
def test_engine_results_are_sorted_ascending_by_time() -> None:
    async def runner() -> tuple[NewsEvent, ...]:
        engine = _engine_with_fixture(currencies=("USD", "EUR", "GBP"))
        return await engine.fetch_events()

    events = asyncio.run(runner())
    assert list(events) == sorted(events, key=lambda e: e.scheduled_at)


@pytest.mark.integration
def test_engine_dedupes_repeated_event_ids() -> None:
    """Two parses of the same fixture must not produce duplicates after merging."""
    fixture = _FIXTURE_PATH.read_text(encoding="utf-8")
    parser = ForexFactoryCalendarParser()
    events = parser.parse(fixture) + parser.parse(fixture)  # double up
    deduped = NewsEngine._filter_and_dedup(events, ("USD", "EUR"))
    # Same six filtered events as a single parse pass — duplicates removed.
    assert len(deduped) == 6
    assert len({e.event_id for e in deduped}) == 6


# ---------------------------------------------------------------------------
# find_active_event — windowing + intensity tie-break
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_find_active_event_picks_high_intensity_when_overlapping() -> None:
    """At 13:30 UTC on 2024-03-06 three USD/EUR releases hit simultaneously."""

    async def runner() -> NewsEvent | None:
        engine = _engine_with_fixture()
        return await engine.find_active_event(
            now=datetime(2024, 3, 6, 13, 30, tzinfo=timezone.utc),
        )

    active = asyncio.run(runner())
    assert active is not None
    # NFP (high) wins over ADP (medium) and Retail (medium).
    assert active.name == "Non-Farm Payrolls"
    assert active.intensity == "high"


@pytest.mark.integration
def test_find_active_event_returns_none_outside_window() -> None:
    async def runner() -> NewsEvent | None:
        engine = _engine_with_fixture()
        # 4 hours before the earliest release — nothing should be active.
        return await engine.find_active_event(
            now=datetime(2024, 3, 6, 9, 0, tzinfo=timezone.utc),
        )

    assert asyncio.run(runner()) is None


@pytest.mark.integration
def test_find_active_event_respects_window_after_release() -> None:
    """Default 30-minute trailing window should still catch a release 20 min later."""

    async def runner() -> NewsEvent | None:
        engine = _engine_with_fixture()
        return await engine.find_active_event(
            now=datetime(2024, 3, 6, 13, 50, tzinfo=timezone.utc),
        )

    active = asyncio.run(runner())
    assert active is not None
    assert active.name == "Non-Farm Payrolls"


@pytest.mark.integration
def test_find_active_event_drops_after_trailing_window() -> None:
    async def runner() -> NewsEvent | None:
        engine = _engine_with_fixture()
        return await engine.find_active_event(
            now=datetime(2024, 3, 6, 14, 1, tzinfo=timezone.utc),
            window_after_seconds=30 * 60,
        )

    assert asyncio.run(runner()) is None


@pytest.mark.unit
def test_find_active_event_rejects_naive_now() -> None:
    engine = _engine_with_fixture()

    async def runner() -> None:
        await engine.find_active_event(now=datetime(2024, 3, 6, 13, 30))

    with pytest.raises(ValueError, match="tz-aware"):
        asyncio.run(runner())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_constants_are_documented() -> None:
    assert FOREXFACTORY_CALENDAR_URL.startswith("https://")
    assert "USD" in DEFAULT_CURRENCIES
    assert "EUR" in DEFAULT_CURRENCIES
    assert DEFAULT_CACHE_TTL_SECONDS > 0


# ---------------------------------------------------------------------------
# Context-manager + ergonomics
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_client_is_async_context_manager() -> None:
    """``async with`` enters/exits cleanly even when no fetch is performed."""
    fixture = _FIXTURE_PATH.read_text(encoding="utf-8")

    async def runner() -> str:
        client = ForexFactoryClient(
            cache_ttl_seconds=300.0,
            client=_build_mock_client(payloads=[fixture]),
        )
        async with client as ctx:
            return await ctx.fetch()

    assert asyncio.run(runner()) == fixture


@pytest.mark.unit
def test_client_url_is_exposed() -> None:
    client = ForexFactoryClient(url="https://example.test/cal", cache_ttl_seconds=0.0)
    assert client.url == "https://example.test/cal"


@pytest.mark.unit
def test_engine_default_currencies_property_normalises_case() -> None:
    engine = NewsEngine(
        client=ForexFactoryClient(client=_build_mock_client(payloads=[""])),
        currencies=("usd", "Eur", "GBP"),
    )
    assert engine.default_currencies == ("USD", "EUR", "GBP")
