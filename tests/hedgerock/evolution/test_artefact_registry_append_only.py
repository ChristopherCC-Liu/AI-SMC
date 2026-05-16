"""Ticket 4 v2 Step 9 closeout — append-only registry guard tests.

The shadow-artefact registry under
``$HEDGEROCK_HOME/policy_registry/shadow_artefacts/`` is
append-only by contract. Once a `*.json` artefact lands there it
MUST be preserved byte-for-byte, regardless of subsequent gate
changes, runner-version bumps, or correctness concerns.

These tests enforce the contract at the source-code level:

  * No production runner / report module imports or invokes a
    delete primitive (``os.remove``, ``os.unlink``, ``shutil.rmtree``,
    ``pathlib.Path.unlink``, ``pathlib.Path.rmdir``).
  * No production module contains a ``rm``-style shell command
    string targeting the registry.
  * Behavioural smoke: running the v0.3.0 runner twice in a row
    with the SAME ``out_dir`` does not modify or delete the first
    artefact (already pinned in
    ``test_shadow_runner_v030_multi_window``; this file adds a
    stronger second-pass check by writing a synthetic artefact
    first and verifying it survives a runner invocation).

A separate audit log
(``policy_registry/shadow_artefacts/_audit.md``) records the
2026-05-02 red-line incident where four ``shadow_runner-0.3.0``
artefacts were deleted from the real registry mid-iteration.
Those artefacts are unrecoverable and the report renderer's
hard-boundary footer surfaces the loss explicitly so it cannot be
silently glossed over.
"""

from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from tests.hedgerock.evolution._paths import (
    real_audit_log as _real_audit_log_p,
)


_REPO = Path(__file__).resolve().parents[3]


# Production modules that touch the shadow-artefact registry.
# Every entry here is scanned for forbidden delete primitives.
_GUARDED_PRODUCTION_PATHS: tuple[Path, ...] = (
    _REPO / "src" / "smc" / "hedgerock" / "evolution" / "shadow_runner.py",
    _REPO / "src" / "smc" / "hedgerock" / "evolution" / "shadow_artefact.py",
    _REPO / "src" / "smc" / "hedgerock" / "evolution" / "multi_window_report.py",
    _REPO / "scripts" / "hedgerock_shadow_run.py",
)


_FORBIDDEN_NAMES: tuple[str, ...] = (
    "unlink",
    "rmtree",
    "remove",   # os.remove / shutil.rmtree are both flagged
    "rmdir",
)


# ---------------------------------------------------------------------------
# 1. AST-level — no production module calls a delete primitive
# ---------------------------------------------------------------------------


def _collect_called_names(tree: ast.AST) -> set[str]:
    """Return the set of attribute / function names invoked in the
    parsed source tree. Captures both bare ``f(...)`` and
    ``obj.f(...)`` call shapes."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute):
                out.add(fn.attr)
            elif isinstance(fn, ast.Name):
                out.add(fn.id)
    return out


@pytest.mark.unit
@pytest.mark.parametrize("path", _GUARDED_PRODUCTION_PATHS)
def test_no_delete_primitive_in_production_module(path: Path) -> None:
    """Each guarded module's source MUST NOT call any of the
    forbidden delete primitives. AST-level — substring search would
    flag false-positives (e.g. comments). We parse and walk."""
    assert path.exists(), f"guarded path missing: {path}"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    called = _collect_called_names(tree)
    leaks = sorted(name for name in _FORBIDDEN_NAMES if name in called)
    assert not leaks, (
        f"{path}: production module invokes forbidden delete "
        f"primitive(s) {leaks}; the registry under "
        "policy_registry/shadow_artefacts is append-only."
    )


@pytest.mark.unit
@pytest.mark.parametrize("path", _GUARDED_PRODUCTION_PATHS)
def test_no_rm_string_in_production_module(path: Path) -> None:
    """A defence-in-depth substring check: no ``rm -rf`` /
    ``shutil.rmtree`` / ``os.unlink`` literal strings in production
    source. AST-only doesn't catch shell-command strings."""
    src = path.read_text(encoding="utf-8")
    forbidden_substrings = (
        "os.remove(",
        "os.unlink(",
        "shutil.rmtree(",
        ".unlink(",
        ".rmdir(",
        "rm -rf",
        "rm -r ",
    )
    leaks = [s for s in forbidden_substrings if s in src]
    assert not leaks, (
        f"{path}: production source contains forbidden delete "
        f"substring(s) {leaks}."
    )


# ---------------------------------------------------------------------------
# 2. Behavioural — runner does not rewrite a pre-existing artefact
# ---------------------------------------------------------------------------


def _bars(start: datetime, n: int, hours_step: float = 1.0,
          *, base_price: float = 100.0):
    rows = []
    for i in range(n):
        ts = start + timedelta(hours=hours_step * i)
        rows.append({
            "ts": ts, "open": base_price, "high": base_price + 0.5,
            "low": base_price - 0.5, "close": base_price, "volume": 100.0,
        })
    return pl.DataFrame(rows).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC")
    )


class _StubLake:
    def __init__(self, data):
        self._data = data
        self._root = Path("/tmp/stub_t4_append_only")

    def list_instruments(self):
        return sorted({k[0] for k in self._data})

    def query(self, instrument, timeframe, start, end):
        df = self._data.get((instrument, str(timeframe)))
        if df is None or df.is_empty():
            return pl.DataFrame()
        return df.filter((pl.col("ts") >= start) & (pl.col("ts") < end))


@pytest.fixture
def long_lake():
    base = datetime(2023, 12, 1, tzinfo=timezone.utc)
    return _StubLake({
        ("XAUUSD", "H1"): _bars(base, n=24 * 120),
        ("XAUUSD", "H4"): _bars(base, n=6 * 120, hours_step=4.0),
        ("XAUUSD", "D1"): _bars(base, n=120, hours_step=24.0),
    })


@pytest.mark.unit
def test_runner_preserves_pre_existing_v030_artefact(
    long_lake, tmp_path,
) -> None:
    """Plant a fake "earlier" v0.3.0 artefact under the candidate's
    subdirectory. After the runner writes a NEW artefact alongside,
    the planted file must remain byte-identical."""
    from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
    from smc.hedgerock.evolution.shadow_runner import (
        run_shadow_for_candidate_multi_window,
    )
    from smc.hedgerock.evolution.window_coverage import WindowSpec

    cand = next(c for c in CANDIDATE_MENU_V0
                if c.candidate_id == "c1-lower-observe-floor-0.50")
    sub = tmp_path / cand.candidate_id
    sub.mkdir(parents=True, exist_ok=True)
    earlier = sub / "20260101T000000-000001.json"
    earlier.write_bytes(b'{"sentinel": "must_not_be_overwritten"}\n')
    earlier_blob = earlier.read_bytes()
    earlier.chmod(0o444)

    windows = [
        WindowSpec(
            window_id="y2024_a",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 31, tzinfo=timezone.utc),
            declared_regime_bucket="range_low_vol",
        ),
        WindowSpec(
            window_id="y2024_b",
            start=datetime(2024, 2, 1, tzinfo=timezone.utc),
            end=datetime(2024, 3, 1, tzinfo=timezone.utc),
            declared_regime_bucket="range_low_vol",
        ),
    ]
    new_path = run_shadow_for_candidate_multi_window(
        candidate=cand, lake=long_lake, symbol="XAUUSD",
        windows=windows, out_dir=tmp_path,
    )
    assert new_path != earlier
    assert earlier.exists()
    assert earlier.read_bytes() == earlier_blob, (
        "runner mutated a pre-existing artefact; the registry must "
        "be append-only"
    )


# ---------------------------------------------------------------------------
# 3. Hard-boundary surface — report renderer's footer states the loss
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_report_footer_carries_stale_v030_deletion_flag() -> None:
    """The hard-boundary footer must include an explicit
    "stale v0.3.0 artefacts deleted during this session" line.
    Operators reading the report cannot miss the loss."""
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    out = render_multi_window_report([])
    assert "stale v0.3.0 artefacts deleted during this session" in out


@pytest.mark.unit
def test_report_footer_links_to_audit_log() -> None:
    """The hard-boundary footer must reference the audit log path
    so operators know where the lost-SHAs evidence lives."""
    from smc.hedgerock.evolution.multi_window_report import (
        render_multi_window_report,
    )
    out = render_multi_window_report([])
    assert "_audit.md" in out


# ---------------------------------------------------------------------------
# 4. Audit log itself — exists, append-only-style header
# ---------------------------------------------------------------------------


_AUDIT_LOG = _real_audit_log_p()


@pytest.mark.unit
def test_audit_log_exists_and_records_2026_05_02_incident() -> None:
    """The audit log MUST exist and document the deletion explicitly.

    Skipped on machines without the real HedgeRock checkout (CI /
    fresh clones) — only fires on the operator's box where
    ``$HEDGEROCK_HOME/policy_registry/shadow_artefacts/_audit.md``
    was preserved across the 2026-05-02 incident.
    """
    if not _AUDIT_LOG.exists():
        pytest.skip(
            f"audit log missing at {_AUDIT_LOG}; this test requires "
            "the real operator-team registry to be present "
            "($HEDGEROCK_HOME/policy_registry/shadow_artefacts/_audit.md)."
        )
    body = _AUDIT_LOG.read_text(encoding="utf-8")
    assert "append-only" in body.lower()
    assert "Stale v0.3.0 artefacts deleted" in body
    # SHAs of the four lost artefacts must be present so a future
    # audit can detect re-creation attempts.
    for sha in (
        "f923fc24c3f1ecd7c2bae21a30b745791baeff6b55c2a4b530df4203c60bd186",
        "c50a2f9b28cdd841a9d62cc1816bba24ad8d65f5dfc2b050fdf5dff5df544d65",
        "913269e479ae57c96d579ba730b0601d8ac1f13fc5dfe1e28ead635716e5b933",
        "6916b7902fa93c5ed8bc755d6b7c973a92e14aa820d57da2bf10a06ceff5de86",
    ):
        assert sha in body, f"audit log missing lost-artefact sha {sha}"
