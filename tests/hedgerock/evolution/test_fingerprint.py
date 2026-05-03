"""Tests for the deterministic fingerprint chain."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from smc.hedgerock.evolution.fingerprint import (
    ALGORITHM_VERSION,
    FingerprintChain,
    FingerprintEntry,
    GENESIS_PREV_HASH,
    compute_fingerprint,
    current_git_commit,
    hash_payload,
    verify_chain,
)


_REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Canonical hash + determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hash_payload_is_deterministic() -> None:
    a = hash_payload({"x": 1, "y": [1, 2, 3]})
    b = hash_payload({"y": [1, 2, 3], "x": 1})
    assert a == b


@pytest.mark.unit
def test_hash_payload_distinguishes_different_payloads() -> None:
    assert hash_payload({"x": 1}) != hash_payload({"x": 2})


@pytest.mark.unit
def test_compute_fingerprint_recomputes_entry_hash() -> None:
    entry = compute_fingerprint(
        operation_type="test",
        inputs={"a": 1}, outputs={"b": 2}, params={"c": 3},
        timestamp="2026-05-03T12:00:00+00:00",
        git_commit="deadbeef0000",
    )
    assert entry.entry_hash == entry.recompute_entry_hash()


@pytest.mark.unit
def test_compute_fingerprint_default_algorithm_version() -> None:
    entry = compute_fingerprint(
        operation_type="test", inputs={}, outputs={},
        timestamp="2026-05-03T00:00:00+00:00", git_commit="x",
    )
    assert entry.algorithm_version == ALGORITHM_VERSION


@pytest.mark.unit
def test_compute_fingerprint_genesis_prev_hash() -> None:
    entry = compute_fingerprint(
        operation_type="test", inputs={}, outputs={},
        timestamp="2026-05-03T00:00:00+00:00", git_commit="x",
    )
    assert entry.prev_hash == GENESIS_PREV_HASH


@pytest.mark.unit
def test_current_git_commit_returns_string() -> None:
    out = current_git_commit()
    assert isinstance(out, str)
    assert out  # non-empty (either a sha or "unknown")


# ---------------------------------------------------------------------------
# Chain append + verify
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chain_append_then_verify(tmp_path: Path) -> None:
    chain = FingerprintChain(tmp_path / "chain.jsonl")
    e1 = chain.append(
        operation_type="op1",
        inputs={"i": 1}, outputs={"o": 1},
    )
    e2 = chain.append(
        operation_type="op2",
        inputs={"i": 2}, outputs={"o": 2},
    )
    # prev linkage
    assert e1.prev_hash == GENESIS_PREV_HASH
    assert e2.prev_hash == e1.entry_hash
    # verifier reports OK
    result = verify_chain(chain.path)
    assert result.ok is True
    assert result.n_entries == 2
    assert result.first_break_index is None


@pytest.mark.unit
def test_verify_chain_detects_tampered_entry(tmp_path: Path) -> None:
    chain = FingerprintChain(tmp_path / "chain.jsonl")
    chain.append(operation_type="op1", inputs={"i": 1}, outputs={})
    chain.append(operation_type="op2", inputs={"i": 2}, outputs={})

    # Tamper with the first line: change inputs but leave entry_hash.
    raw = chain.path.read_text(encoding="utf-8").splitlines()
    obj = json.loads(raw[0])
    obj["input_hash"] = "00" * 32
    raw[0] = json.dumps(obj, sort_keys=True)
    chain.path.write_text("\n".join(raw) + "\n", encoding="utf-8")

    result = verify_chain(chain.path)
    assert result.ok is False
    assert result.first_break_index == 0
    assert "entry_hash mismatch" in (result.first_break_reason or "")


@pytest.mark.unit
def test_verify_chain_detects_broken_prev_hash(tmp_path: Path) -> None:
    chain = FingerprintChain(tmp_path / "chain.jsonl")
    chain.append(operation_type="op1", inputs={"i": 1}, outputs={})
    chain.append(operation_type="op2", inputs={"i": 2}, outputs={})
    raw = chain.path.read_text(encoding="utf-8").splitlines()
    # Drop the first entry — second's prev_hash now points at nothing.
    chain.path.write_text(raw[1] + "\n", encoding="utf-8")
    result = verify_chain(chain.path)
    assert result.ok is False
    assert "prev_hash mismatch" in (result.first_break_reason or "")


@pytest.mark.unit
def test_verify_chain_handles_missing_file(tmp_path: Path) -> None:
    result = verify_chain(tmp_path / "missing.jsonl")
    assert result.ok is False
    assert "not found" in (result.first_break_reason or "")


@pytest.mark.unit
def test_chain_refuses_forbidden_paths(tmp_path: Path) -> None:
    bad = tmp_path / "policy_registry" / "approved" / "chain.jsonl"
    with pytest.raises(ValueError):
        FingerprintChain(bad)


@pytest.mark.unit
def test_entry_round_trips_through_json(tmp_path: Path) -> None:
    chain = FingerprintChain(tmp_path / "chain.jsonl")
    chain.append(operation_type="op1", inputs={"i": 1}, outputs={"o": 1})
    raw = chain.path.read_text(encoding="utf-8").splitlines()
    parsed = json.loads(raw[0])
    restored = FingerprintEntry(**parsed)
    assert restored.recompute_entry_hash() == restored.entry_hash


# ---------------------------------------------------------------------------
# CLI smoke (verify + diff)
# ---------------------------------------------------------------------------


def _import_verify_cli():
    import sys
    scripts_dir = _REPO / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import hedgerock_evolution_verify as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return cli


@pytest.mark.unit
def test_verify_cli_returns_zero_on_clean_chain(tmp_path: Path) -> None:
    chain = FingerprintChain(tmp_path / "chain.jsonl")
    chain.append(operation_type="x", inputs={}, outputs={})
    cli = _import_verify_cli()
    rc = cli.main(["verify", str(chain.path)])
    assert rc == 0


@pytest.mark.unit
def test_verify_cli_returns_nonzero_on_tampered_chain(tmp_path: Path) -> None:
    chain = FingerprintChain(tmp_path / "chain.jsonl")
    chain.append(operation_type="x", inputs={}, outputs={})
    raw = chain.path.read_text(encoding="utf-8").splitlines()
    obj = json.loads(raw[0])
    obj["entry_hash"] = "ff" * 32
    raw[0] = json.dumps(obj, sort_keys=True)
    chain.path.write_text("\n".join(raw) + "\n", encoding="utf-8")
    cli = _import_verify_cli()
    rc = cli.main(["verify", str(chain.path)])
    assert rc != 0


@pytest.mark.unit
def test_verify_cli_diff_match(tmp_path: Path) -> None:
    a = FingerprintChain(tmp_path / "a.jsonl")
    b = FingerprintChain(tmp_path / "b.jsonl")
    fixed_ts = "2026-05-03T12:00:00+00:00"
    a_entry = compute_fingerprint(
        operation_type="x", inputs={"k": 1}, outputs={"v": 1},
        timestamp=fixed_ts, git_commit="cafe1234",
    )
    b_entry = compute_fingerprint(
        operation_type="x", inputs={"k": 1}, outputs={"v": 1},
        timestamp=fixed_ts, git_commit="cafe1234",
    )
    for c, e in [(a, a_entry), (b, b_entry)]:
        with c.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(asdict(e), sort_keys=True, ensure_ascii=False)
                + "\n"
            )
    cli = _import_verify_cli()
    rc = cli.main(["diff", str(a.path), str(b.path)])
    assert rc == 0


@pytest.mark.unit
def test_verify_cli_diff_diverges(tmp_path: Path) -> None:
    a = FingerprintChain(tmp_path / "a.jsonl")
    b = FingerprintChain(tmp_path / "b.jsonl")
    a.append(operation_type="op1", inputs={"k": 1}, outputs={})
    b.append(operation_type="op2", inputs={"k": 1}, outputs={})
    cli = _import_verify_cli()
    rc = cli.main(["diff", str(a.path), str(b.path)])
    assert rc != 0


# ---------------------------------------------------------------------------
# Integration with candidate_generator (optional sink)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_candidate_generator_populates_fingerprint_sink() -> None:
    from smc.hedgerock.evolution.candidate_generator import (
        generate_candidate_proposals,
    )
    from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
    from smc.hedgerock.evolution.policy_manifest import EvidenceBundle

    bundle = EvidenceBundle(
        bundle_id="<test>",
        bundle_hash_sha256="0" * 64,
        atlas_report_path="<test>",
        atlas_report_hash_sha256="0" * 64,
        data_availability_report_path="<test>",
        data_availability_report_hash_sha256="0" * 64,
        walk_forward_run_paths=("<test>",),
        year_replication={
            "XAUUSD": {"years_total": 5, "years_passing": 5,
                       "negative_sign_years": ()},
        },
        cross_symbol_count=1,
        halt_event_count=42,
        no_strategy_change=True,
    )
    sink: dict = {}
    generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate={},
        blocking_reasons_per_candidate={},
        fingerprint_sink=sink,
    )
    assert "entry" in sink
    assert isinstance(sink["entry"], FingerprintEntry)
    assert sink["entry"].operation_type == "generate_candidate_proposals"


@pytest.mark.unit
def test_candidate_generator_default_does_not_touch_sink() -> None:
    from smc.hedgerock.evolution.candidate_generator import (
        generate_candidate_proposals,
    )
    from smc.hedgerock.evolution.candidate_menu import CANDIDATE_MENU_V0
    from smc.hedgerock.evolution.policy_manifest import EvidenceBundle

    bundle = EvidenceBundle(
        bundle_id="<test>",
        bundle_hash_sha256="0" * 64,
        atlas_report_path="<test>",
        atlas_report_hash_sha256="0" * 64,
        data_availability_report_path="<test>",
        data_availability_report_hash_sha256="0" * 64,
        walk_forward_run_paths=("<test>",),
        year_replication={
            "XAUUSD": {"years_total": 5, "years_passing": 5,
                       "negative_sign_years": ()},
        },
        cross_symbol_count=1,
        halt_event_count=42,
        no_strategy_change=True,
    )
    sink: dict = {}
    generate_candidate_proposals(
        candidate_menu=CANDIDATE_MENU_V0,
        bundle=bundle,
        gate_results_per_candidate={},
        blocking_reasons_per_candidate={},
    )
    assert sink == {}
