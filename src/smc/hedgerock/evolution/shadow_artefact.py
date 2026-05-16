"""Ticket 2 Step 1 — Shadow comparison artefact schema.

**Pure dataclasses + JSON IO. No imports from production runtime.**

ShadowArtefact is the hash-pinned evidence object produced by the
shadow runner (Step 5) and consumed by ``g8_shadow_comparison``
(Step 6). Schema follows R2 of the Ticket 2 plan.

On-disk layout mirrors Ticket 1's wrapped envelope::

    {
      "content_sha256": "<sha256 of inner artefact payload>",
      "artefact":       { ...all R2 v1 fields... }
    }

Strict load semantics:
  - Top-level must be the wrapped envelope; bare layouts are refused
    so registry-managed artefacts cannot be silently swapped for
    hand-authored ones.
  - SHA-256 over the canonical inner serialisation must match the
    recorded envelope hash; tamper at any field → fail.
  - Every R2 v1 top-level field is mandatory; absence → fail.
  - Schema major-version mismatch → fail (v1 loader refuses v2+).

Non-goals for Step 1:
  - No metric computation (that's Step 5).
  - No runner / mirror / overlay logic (Steps 3–5).
  - No registry write helpers (Step 5 will add a thin write layer
    that delegates to ``dump_shadow_artefact``).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


__all__ = [
    "SHADOW_ARTEFACT_SCHEMA_VERSION",
    "SHADOW_ARTEFACT_SCHEMA_MAJOR",
    "CandidateDiffSnapshot",
    "DataSliceIdentity",
    "NoLiveEvidence",
    "NoLookaheadAudit",
    "ReplayInvariants",
    "ShadowArtefact",
    "ShadowArtefactIntegrityError",
    "ShadowMetrics",
    "ShadowVerdict",
    "SidecarModuleHashes",
    "artefact_from_dict",
    "artefact_to_dict",
    "dump_shadow_artefact",
    "load_shadow_artefact",
    "verify_shadow_artefact_unchanged",
]


SHADOW_ARTEFACT_SCHEMA_VERSION: str = "1.0.0"
SHADOW_ARTEFACT_SCHEMA_MAJOR: int = 1


class ShadowArtefactIntegrityError(ValueError):
    """Raised when a shadow artefact file fails its content_sha256
    integrity check, lacks the wrapped envelope, is missing a
    required v1 field, or carries an unsupported schema-major
    version. Subclassed from ``ValueError`` so callers may catch the
    looser type when the exact failure mode does not matter.
    """


class ShadowVerdict(StrEnum):
    """Per R1 verdict table. ``NOT_RUN`` is the default when no
    artefact / runner output exists; gates and report code may
    surface this label without ever loading an artefact."""

    PASS = "PASS"
    FAIL = "FAIL"
    ABSTAIN = "ABSTAIN"
    NOT_RUN = "NOT_RUN"


# ---------------------------------------------------------------------------
# Sub-structures (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateDiffSnapshot:
    """Pinned snapshot of the candidate's diff at the moment the
    artefact was produced. Compared against the live candidate
    manifest at G8 evaluation time so menu drift is detectable
    before any verdict is returned."""

    target: str
    proposed_value: float | int | str | None
    baseline_value: float | int | str | None


@dataclass(frozen=True)
class DataSliceIdentity:
    """Identity of the historical data slice used for the replay.

    All four hash-pinned fields (lake_snapshot_hash + row_counts +
    closed_bar_rule_version + symbols) are checked at G8 time
    against the report CLI's current view of the lake. Any
    divergence → G8 FAIL: shadow_artefact_data_slice_mismatch.
    """

    symbols: tuple[str, ...]
    time_range_start: str  # ISO date
    time_range_end: str    # ISO date
    timeframes: tuple[str, ...]
    closed_bar_rule_version: str
    lake_snapshot_hash: str
    lake_snapshot_row_counts: dict[str, int]


@dataclass(frozen=True)
class SidecarModuleHashes:
    """SHA-256 of each sidecar module's source bytes at runtime.
    Allows G8 to detect "artefact produced with sidecar X, evaluated
    by sidecar Y"."""

    policy_overlay: str
    rule_engine_mirror: str
    replay_constant_mirror: str
    shadow_runner: str
    shadow_metrics: str


@dataclass(frozen=True)
class ShadowMetrics:
    """Per-replay outcome metrics (placeholder schema for Step 1;
    Step 5 will populate full set of fields used by
    g8 thresholds)."""

    final_equity: float
    total_return_pct: float
    max_dd_pct: float
    near_stopout_count: int
    n_trades: int
    max_open_lots: float
    max_grid_density: int
    halt_event_count: int
    n_bars_envelope_decided: int


@dataclass(frozen=True)
class ReplayInvariants:
    """Self-reported invariants about the replay runtime. Each must
    be True (or False, where indicated) for a valid artefact.
    G8 will refuse PASS on any False here that should be True."""

    same_bar_set_used: bool                             # must be True
    same_transition_lock_state_machine: bool            # must be True
    same_cooldown_carryover: bool                       # must be True
    decision_only_uses_strictly_prior_data: bool        # must be True
    h4_partial_bar_in_window: bool                      # must be False
    d1_partial_bar_in_window: bool                      # must be False
    decision_uses_data_with_ts_lt_trade_bar_ts: bool    # must be True


@dataclass(frozen=True)
class NoLiveEvidence:
    """Self-reported "I did not touch live" evidence."""

    decision_server_routes_unchanged_hash: str
    rule_engine_constants_unchanged_hash: str
    http_calls_made_count: int                          # must be 0
    broker_api_calls_made_count: int                    # must be 0
    files_written_under_src_or_config_or_mq5_count: int # must be 0
    files_written_under_approved_or_pointer_count: int  # must be 0


@dataclass(frozen=True)
class NoLookaheadAudit:
    decision_uses_only_prior_closed_bars: bool          # must be True
    partial_bar_violation_count: int                    # must be 0


# ---------------------------------------------------------------------------
# Top-level artefact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowArtefact:
    """v1 schema. Ticket 4 v2 ADDS three OPTIONAL dict fields
    (``per_window`` / ``window_coverage`` / ``gold_profile``) at
    the tail; defaults to empty dicts so:

      - new ``shadow_runner-0.3.0`` artefacts populate them with
        XAUUSD multi-window evidence;
      - old ``shadow_runner-0.1.0`` / ``-0.2.0`` artefacts that
        omit these fields still load cleanly.

    Schema major version unchanged at v1; runner_version embedded
    in the artefact distinguishes Ticket 2 / 3 / 4 vintages, and
    G8's MIN_RUNNER_VERSION_FOR_ACTIVE_PASS_EVALUATION gate uses
    that string to decide whether the active-PASS path is
    reachable for this artefact.
    """

    artefact_schema_version: str
    artefact_id: str
    generated_at: str

    candidate_id: str
    candidate_manifest_content_hash: str
    candidate_diff: CandidateDiffSnapshot
    candidate_diff_hash: str

    baseline_policy_id: str
    baseline_policy_hash: str
    candidate_overlay_id: str

    data_slice: DataSliceIdentity

    runner_version: str
    mirror_version: str
    metric_schema_version: str
    sidecar_module_hashes: SidecarModuleHashes

    baseline_metrics: ShadowMetrics
    candidate_metrics: ShadowMetrics
    delta_metrics: ShadowMetrics

    replay_invariants: ReplayInvariants
    no_live_evidence: NoLiveEvidence

    mirror_consistency_check: str          # "PASS" | "FAIL"
    exposure_class_violation: bool
    no_lookahead_audit: NoLookaheadAudit

    verdict: ShadowVerdict
    verdict_reason: str

    # Ticket 4 v2 — XAUUSD multi-window optional fields. Defaults
    # to {} so any artefact without these (Ticket 2 / 3) still
    # loads cleanly under the strict envelope loader.
    per_window: dict[str, Any] = field(default_factory=dict)
    window_coverage: dict[str, Any] = field(default_factory=dict)
    gold_profile: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses / enums / tuples to JSON-safe
    primitives. Sorted dicts give byte-deterministic output."""
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def artefact_to_dict(artefact: ShadowArtefact) -> dict[str, Any]:
    return _to_jsonable(artefact)  # type: ignore[return-value]


# Optional-on-load top-level fields (Ticket 4 v2). Missing → default {};
# present but malformed type → ShadowArtefactIntegrityError.
_OPTIONAL_TOP_LEVEL_FIELDS = ("per_window", "window_coverage", "gold_profile")
_REQUIRED_TOP_LEVEL_FIELDS = tuple(
    f.name for f in fields(ShadowArtefact)
    if f.name not in _OPTIONAL_TOP_LEVEL_FIELDS
)


def artefact_from_dict(data: dict[str, Any]) -> ShadowArtefact:
    """Bare-payload loader. Used by ``load_shadow_artefact`` after
    envelope verification, and by test fixtures that need to bypass
    the envelope. Validates every R2 v1 field is present."""
    if not isinstance(data, dict):
        raise ShadowArtefactIntegrityError(
            "artefact payload must be a JSON object"
        )
    missing = [f for f in _REQUIRED_TOP_LEVEL_FIELDS if f not in data]
    if missing:
        raise ShadowArtefactIntegrityError(
            f"artefact missing required v1 field(s): {missing}"
        )
    version = data["artefact_schema_version"]
    if not isinstance(version, str):
        raise ShadowArtefactIntegrityError(
            f"artefact_schema_version must be string, got {type(version)}"
        )
    major = version.split(".", 1)[0]
    if major != str(SHADOW_ARTEFACT_SCHEMA_MAJOR):
        raise ShadowArtefactIntegrityError(
            f"unsupported artefact_schema_version major: {version} "
            f"(this loader supports v{SHADOW_ARTEFACT_SCHEMA_MAJOR}.x)"
        )

    diff_raw = data["candidate_diff"]
    candidate_diff = CandidateDiffSnapshot(
        target=diff_raw["target"],
        proposed_value=diff_raw.get("proposed_value"),
        baseline_value=diff_raw.get("baseline_value"),
    )

    ds_raw = data["data_slice"]
    data_slice = DataSliceIdentity(
        symbols=tuple(ds_raw["symbols"]),
        time_range_start=ds_raw["time_range_start"],
        time_range_end=ds_raw["time_range_end"],
        timeframes=tuple(ds_raw["timeframes"]),
        closed_bar_rule_version=ds_raw["closed_bar_rule_version"],
        lake_snapshot_hash=ds_raw["lake_snapshot_hash"],
        lake_snapshot_row_counts={
            str(k): int(v)
            for k, v in dict(ds_raw["lake_snapshot_row_counts"]).items()
        },
    )

    sm_raw = data["sidecar_module_hashes"]
    sidecar_hashes = SidecarModuleHashes(
        policy_overlay=sm_raw["policy_overlay"],
        rule_engine_mirror=sm_raw["rule_engine_mirror"],
        replay_constant_mirror=sm_raw["replay_constant_mirror"],
        shadow_runner=sm_raw["shadow_runner"],
        shadow_metrics=sm_raw["shadow_metrics"],
    )

    def _metrics(raw: dict[str, Any]) -> ShadowMetrics:
        return ShadowMetrics(
            final_equity=float(raw["final_equity"]),
            total_return_pct=float(raw["total_return_pct"]),
            max_dd_pct=float(raw["max_dd_pct"]),
            near_stopout_count=int(raw["near_stopout_count"]),
            n_trades=int(raw["n_trades"]),
            max_open_lots=float(raw["max_open_lots"]),
            max_grid_density=int(raw["max_grid_density"]),
            halt_event_count=int(raw["halt_event_count"]),
            n_bars_envelope_decided=int(raw["n_bars_envelope_decided"]),
        )

    baseline_metrics = _metrics(data["baseline_metrics"])
    candidate_metrics = _metrics(data["candidate_metrics"])
    delta_metrics = _metrics(data["delta_metrics"])

    inv_raw = data["replay_invariants"]
    invariants = ReplayInvariants(
        same_bar_set_used=bool(inv_raw["same_bar_set_used"]),
        same_transition_lock_state_machine=bool(
            inv_raw["same_transition_lock_state_machine"]
        ),
        same_cooldown_carryover=bool(inv_raw["same_cooldown_carryover"]),
        decision_only_uses_strictly_prior_data=bool(
            inv_raw["decision_only_uses_strictly_prior_data"]
        ),
        h4_partial_bar_in_window=bool(inv_raw["h4_partial_bar_in_window"]),
        d1_partial_bar_in_window=bool(inv_raw["d1_partial_bar_in_window"]),
        decision_uses_data_with_ts_lt_trade_bar_ts=bool(
            inv_raw["decision_uses_data_with_ts_lt_trade_bar_ts"]
        ),
    )

    nle_raw = data["no_live_evidence"]
    no_live = NoLiveEvidence(
        decision_server_routes_unchanged_hash=nle_raw[
            "decision_server_routes_unchanged_hash"
        ],
        rule_engine_constants_unchanged_hash=nle_raw[
            "rule_engine_constants_unchanged_hash"
        ],
        http_calls_made_count=int(nle_raw["http_calls_made_count"]),
        broker_api_calls_made_count=int(nle_raw["broker_api_calls_made_count"]),
        files_written_under_src_or_config_or_mq5_count=int(
            nle_raw["files_written_under_src_or_config_or_mq5_count"]
        ),
        files_written_under_approved_or_pointer_count=int(
            nle_raw["files_written_under_approved_or_pointer_count"]
        ),
    )

    nla_raw = data["no_lookahead_audit"]
    no_lookahead = NoLookaheadAudit(
        decision_uses_only_prior_closed_bars=bool(
            nla_raw["decision_uses_only_prior_closed_bars"]
        ),
        partial_bar_violation_count=int(nla_raw["partial_bar_violation_count"]),
    )

    return ShadowArtefact(
        artefact_schema_version=version,
        artefact_id=data["artefact_id"],
        generated_at=data["generated_at"],
        candidate_id=data["candidate_id"],
        candidate_manifest_content_hash=data["candidate_manifest_content_hash"],
        candidate_diff=candidate_diff,
        candidate_diff_hash=data["candidate_diff_hash"],
        baseline_policy_id=data["baseline_policy_id"],
        baseline_policy_hash=data["baseline_policy_hash"],
        candidate_overlay_id=data["candidate_overlay_id"],
        data_slice=data_slice,
        runner_version=data["runner_version"],
        mirror_version=data["mirror_version"],
        metric_schema_version=data["metric_schema_version"],
        sidecar_module_hashes=sidecar_hashes,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        delta_metrics=delta_metrics,
        replay_invariants=invariants,
        no_live_evidence=no_live,
        mirror_consistency_check=str(data["mirror_consistency_check"]),
        exposure_class_violation=bool(data["exposure_class_violation"]),
        no_lookahead_audit=no_lookahead,
        verdict=ShadowVerdict(data["verdict"]),
        verdict_reason=str(data["verdict_reason"]),
        per_window=dict(data.get("per_window", {})),
        window_coverage=dict(data.get("window_coverage", {})),
        gold_profile=dict(data.get("gold_profile", {})),
    )


# ---------------------------------------------------------------------------
# Wrapped envelope IO (mirrors Ticket 1 policy_manifest)
# ---------------------------------------------------------------------------


def _canonical_payload(artefact: ShadowArtefact) -> tuple[str, dict[str, Any]]:
    payload = artefact_to_dict(artefact)
    canonical = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False,
    )
    return canonical, payload


def _wrap_with_hash(canonical: str, payload: dict[str, Any]) -> str:
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    wrapper = {"content_sha256": h, "artefact": payload}
    return json.dumps(wrapper, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def dump_shadow_artefact(artefact: ShadowArtefact, path: Path) -> None:
    """Write artefact as wrapped envelope, mode 0444, no overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"shadow artefact already exists at {path}; artefacts are "
            "immutable once written"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical, payload = _canonical_payload(artefact)
    text = _wrap_with_hash(canonical, payload)
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o444)


def _verify_envelope_and_unwrap(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ShadowArtefactIntegrityError(
            "shadow artefact top-level must be a JSON object"
        )
    if "content_sha256" not in raw or "artefact" not in raw:
        raise ShadowArtefactIntegrityError(
            "shadow artefact is bare (no `content_sha256` envelope); "
            "registry-managed artefacts must be wrapped"
        )
    expected = raw["content_sha256"]
    if not isinstance(expected, str) or not expected:
        raise ShadowArtefactIntegrityError(
            f"content_sha256 must be a non-empty string, got {type(expected)}"
        )
    inner = raw["artefact"]
    if not isinstance(inner, dict):
        raise ShadowArtefactIntegrityError(
            "envelope `artefact` field must be a JSON object"
        )
    canonical = json.dumps(
        inner, indent=2, sort_keys=True, ensure_ascii=False,
    )
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual != expected:
        raise ShadowArtefactIntegrityError(
            f"shadow artefact content_sha256 mismatch: stored "
            f"{expected[:16]}…, recomputed {actual[:16]}…"
        )
    return inner


def load_shadow_artefact(path: Path) -> ShadowArtefact:
    """Strict shadow-artefact loader.

    On-disk shape MUST be the wrapped envelope. The hash is verified
    BEFORE unwrapping; mismatch raises
    ``ShadowArtefactIntegrityError``. After the envelope is verified,
    the inner payload is parsed by :func:`artefact_from_dict`."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ShadowArtefactIntegrityError(
            f"shadow artefact is not valid JSON: {e}"
        ) from e
    payload = _verify_envelope_and_unwrap(raw)
    return artefact_from_dict(payload)


def verify_shadow_artefact_unchanged(path: Path) -> bool:
    """Returns True when the artefact at ``path`` loads cleanly under
    the strict loader. False on any failure mode (tamper, bare,
    missing file, malformed JSON, missing field, schema mismatch)."""
    try:
        load_shadow_artefact(Path(path))
    except (ShadowArtefactIntegrityError, FileNotFoundError, KeyError, ValueError):
        return False
    except OSError:
        return False
    return True
