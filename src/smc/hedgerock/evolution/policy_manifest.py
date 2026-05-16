"""Phase D-cont3 / Ticket 1 — policy manifest schema + JSON IO.

**Pure dataclasses + JSON IO. No imports from production runtime.**

The schema mirrors the JSON example in
``docs/ticket-1-report-only-policy-registry-plan.md`` §6.3. Every
field is frozen; manifests are immutable once written. Files are
written ``chmod 0444`` so a second write (overwrite) fails. Loaders
verify the schema major version and refuse v2+.

This module never imports ``smc.hedgerock.rule_engine`` or
``smc.hedgerock.decision_server`` — the registry is a sidecar layer.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_MAJOR",
    "CandidateDiff",
    "CandidateDiffScope",
    "CandidateManifest",
    "CandidateState",
    "EvidenceBundle",
    "GateStatus",
    "ManifestIntegrityError",
    "OverallResult",
    "PromotionGateResult",
    "compute_canonical_candidate_hash",
    "dump_manifest",
    "load_manifest",
    "verify_manifest_unchanged",
    "manifest_to_dict",
    "manifest_from_dict",
]


class ManifestIntegrityError(ValueError):
    """Raised when a manifest file fails its ``content_sha256``
    integrity check, when the envelope is missing (bare layout in a
    registry-managed file), or when the file is not valid JSON.

    Subclassed from :class:`ValueError` so callers may catch the
    looser type when they don't need to distinguish integrity issues
    from other validation failures.
    """


# Schema version is human-meaningful semver. Major bumps require RFC
# amendment; minor bumps are additive (backwards-compatible) only.
MANIFEST_SCHEMA_VERSION: str = "1.0.0"
MANIFEST_SCHEMA_MAJOR: int = 1


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CandidateState(StrEnum):
    """Per RFC §3 + Plan §6.2 state machine. Ticket 1 only writes
    candidates in DRAFT (and reads back the same)."""

    DRAFT = "draft"
    TESTED = "tested"
    SHADOW_VALIDATED = "shadow_validated"
    CANARY = "canary"
    APPROVED = "approved"
    QUARANTINED = "quarantined"


class GateStatus(StrEnum):
    """Per Plan §3."""

    PASS = "PASS"
    FAIL = "FAIL"
    ABSTAIN = "ABSTAIN"
    NOT_RUN = "NOT_RUN"


class OverallResult(StrEnum):
    """Per Plan §3 ``compute_overall_result``.

    ``PROMOTION_BLOCKED_MANIFEST_INVALID`` is the explicit composite
    label from Plan §3 G6 (b): an undefined safety-bound makes the
    candidate's evidence structurally invalid for promotion, distinct
    from a candidate that simply fails a semantic gate.
    """

    READY_FOR_TESTED = "READY_FOR_TESTED"
    PROMOTION_BLOCKED = "PROMOTION_BLOCKED"
    PROMOTION_BLOCKED_MANIFEST_INVALID = "PROMOTION_BLOCKED / manifest_invalid"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateDiffScope:
    """Per-knob effect description. Booleans drive G6 (safety bounds),
    G5 (halt-event corpus), and the exposure-class veto in
    ``compute_overall_result``."""

    regimes_affected: tuple[str, ...] = ()
    affects_halt_mode: bool = False
    affects_classifier_or_rule_engine: bool = True
    raises_gross_exposure: bool = False
    raises_leverage: bool = False
    raises_max_open_positions: bool = False
    raises_max_recovery_multiplier: bool = False
    raises_max_grid_density: bool = False
    interfaces_touched: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateDiff:
    """A single proposed change. ``target`` is a dotted module path so
    G6 can look up the corresponding band in ``safety_bounds.yaml``."""

    kind: str
    target: str
    baseline_value: float | int | str | None
    proposed_value: float | int | str | None
    scope: CandidateDiffScope


@dataclass(frozen=True)
class EvidenceBundle:
    """Hash-pinned reference to the artefacts the gates will read.

    Hashes guarantee that a candidate written today against a specific
    Phase D bundle cannot be silently re-evaluated against a different
    bundle later — the registry will refuse the load.

    Ticket 2 fields (``shadow_artefact_*``) are optional. When absent
    the candidate has no shadow evidence and G8 returns ``NOT_RUN``.
    When present, the recorded ``shadow_artefact_hash_sha256`` is
    compared against a recomputed disk hash; mismatch → G8 FAIL.
    """

    bundle_id: str
    bundle_hash_sha256: str
    atlas_report_path: str
    atlas_report_hash_sha256: str
    data_availability_report_path: str
    data_availability_report_hash_sha256: str
    walk_forward_run_paths: tuple[str, ...]
    # symbol -> {"years_total": int, "years_passing": int,
    #            "negative_sign_years": tuple[int, ...]}
    year_replication: dict[str, dict[str, Any]]
    cross_symbol_count: int
    halt_event_count: int
    no_strategy_change: bool
    # Ticket 2 — Step 6: optional shadow-artefact join key. None when
    # no shadow run has been recorded for the candidate.
    shadow_artefact_path: str | None = None
    shadow_artefact_hash_sha256: str | None = None
    # Ticket 4 v2 follow-on T4-F1 — registry append-only audit
    # state. None when the operator did not supply registry audit
    # information; G8 then proceeds without the new gate. When set
    # to a :class:`RegistryAuditState` with
    # ``registry_append_only_violation=True``, G8 ABSTAINs every
    # candidate this round regardless of artefact verdict.
    registry_audit: "Any | None" = None


@dataclass(frozen=True)
class PromotionGateResult:
    gate_id: str
    status: GateStatus
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateManifest:
    manifest_schema_version: str
    candidate_id: str
    title: str
    author: str
    created_at: str
    state: CandidateState
    diff: CandidateDiff
    evidence_bundle: EvidenceBundle | None
    gates: tuple[PromotionGateResult, ...]
    result: OverallResult
    blocking_reasons: tuple[str, ...]
    next_data_needs: tuple[str, ...]
    required_next_data_or_policy: str
    human_approval_required_for_state_transitions_above: str
    audit_trail: tuple[dict[str, Any], ...]


# ---------------------------------------------------------------------------
# JSON IO
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses / enums / tuples into JSON-safe
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


def manifest_to_dict(manifest: CandidateManifest) -> dict[str, Any]:
    return _to_jsonable(manifest)  # type: ignore[return-value]


def _coerce_dict_recursively(value: Any) -> Any:
    """Convert lists back to tuples on load to preserve immutability
    invariants of the dataclasses (they declare tuple fields)."""
    if isinstance(value, list):
        return tuple(_coerce_dict_recursively(v) for v in value)
    if isinstance(value, dict):
        return {k: _coerce_dict_recursively(v) for k, v in value.items()}
    return value


def manifest_from_dict(data: dict[str, Any]) -> CandidateManifest:
    if "manifest_schema_version" not in data:
        raise KeyError("manifest_schema_version is required")
    version = data["manifest_schema_version"]
    if not isinstance(version, str):
        raise ValueError(f"manifest_schema_version must be string, got {type(version)}")
    major = version.split(".", 1)[0]
    if major != str(MANIFEST_SCHEMA_MAJOR):
        raise ValueError(
            f"unsupported manifest_schema_version major: {version} "
            f"(this loader supports v{MANIFEST_SCHEMA_MAJOR}.x)"
        )

    diff_raw = data["diff"]
    scope_raw = diff_raw["scope"]
    scope = CandidateDiffScope(
        regimes_affected=tuple(scope_raw.get("regimes_affected", [])),
        affects_halt_mode=bool(scope_raw.get("affects_halt_mode", False)),
        affects_classifier_or_rule_engine=bool(
            scope_raw.get("affects_classifier_or_rule_engine", True)
        ),
        raises_gross_exposure=bool(scope_raw.get("raises_gross_exposure", False)),
        raises_leverage=bool(scope_raw.get("raises_leverage", False)),
        raises_max_open_positions=bool(
            scope_raw.get("raises_max_open_positions", False)
        ),
        raises_max_recovery_multiplier=bool(
            scope_raw.get("raises_max_recovery_multiplier", False)
        ),
        raises_max_grid_density=bool(scope_raw.get("raises_max_grid_density", False)),
        interfaces_touched=tuple(scope_raw.get("interfaces_touched", [])),
    )
    diff = CandidateDiff(
        kind=diff_raw["kind"],
        target=diff_raw["target"],
        baseline_value=diff_raw.get("baseline_value"),
        proposed_value=diff_raw.get("proposed_value"),
        scope=scope,
    )

    eb_raw = data.get("evidence_bundle")
    evidence_bundle: EvidenceBundle | None = None
    if eb_raw is not None:
        # year_replication: keep as dict but coerce inner negative_sign_years to tuple
        yr = {}
        for sym, body in eb_raw.get("year_replication", {}).items():
            yr[sym] = {
                "years_total": int(body["years_total"]),
                "years_passing": int(body["years_passing"]),
                "negative_sign_years": tuple(body.get("negative_sign_years", [])),
            }
        evidence_bundle = EvidenceBundle(
            bundle_id=eb_raw["bundle_id"],
            bundle_hash_sha256=eb_raw["bundle_hash_sha256"],
            atlas_report_path=eb_raw["atlas_report_path"],
            atlas_report_hash_sha256=eb_raw["atlas_report_hash_sha256"],
            data_availability_report_path=eb_raw["data_availability_report_path"],
            data_availability_report_hash_sha256=eb_raw[
                "data_availability_report_hash_sha256"
            ],
            walk_forward_run_paths=tuple(eb_raw.get("walk_forward_run_paths", [])),
            year_replication=yr,
            cross_symbol_count=int(eb_raw["cross_symbol_count"]),
            halt_event_count=int(eb_raw["halt_event_count"]),
            no_strategy_change=bool(eb_raw["no_strategy_change"]),
        )

    gates: list[PromotionGateResult] = []
    for g_raw in data.get("gates", []):
        gates.append(PromotionGateResult(
            gate_id=g_raw["gate_id"],
            status=GateStatus(g_raw["status"]),
            reason=g_raw["reason"],
            details=dict(g_raw.get("details", {})),
        ))

    return CandidateManifest(
        manifest_schema_version=version,
        candidate_id=data["candidate_id"],
        title=data["title"],
        author=data["author"],
        created_at=data["created_at"],
        state=CandidateState(data["state"]),
        diff=diff,
        evidence_bundle=evidence_bundle,
        gates=tuple(gates),
        result=OverallResult(data["result"]),
        blocking_reasons=tuple(data.get("blocking_reasons", [])),
        next_data_needs=tuple(data.get("next_data_needs", [])),
        required_next_data_or_policy=data.get("required_next_data_or_policy", ""),
        human_approval_required_for_state_transitions_above=data.get(
            "human_approval_required_for_state_transitions_above", "tested"
        ),
        audit_trail=tuple(data.get("audit_trail", [])),
    )


def compute_canonical_candidate_hash(manifest: CandidateManifest) -> str:
    """Hash the candidate's MENU IDENTITY only.

    Used by Ticket 2's R5 double-key join. Evaluation-derived
    fields (``evidence_bundle``, ``gates``, ``result``,
    ``blocking_reasons``, ``next_data_needs``,
    ``required_next_data_or_policy``, ``audit_trail``) are stripped
    so two callsites that build the same menu candidate but attach
    different evaluation state still hash to the same id.

    The point of the join key is "is this artefact about THIS menu
    entry?", not "did the artefact run with the same evidence
    bundle?" — the latter is checked separately by
    ``shadow_artefact.data_slice``.
    """
    from dataclasses import replace as _replace
    canonical_form = _replace(
        manifest,
        evidence_bundle=None,
        gates=(),
        result=OverallResult.PROMOTION_BLOCKED,
        blocking_reasons=(),
        next_data_needs=(),
        required_next_data_or_policy="",
        audit_trail=(),
    )
    payload = manifest_to_dict(canonical_form)
    canonical = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_payload_json(manifest: CandidateManifest) -> str:
    """Deterministic canonical serialisation of the dataclass payload.
    Used both to generate the on-disk bytes AND as the hash input."""
    return json.dumps(
        manifest_to_dict(manifest),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )


def _wrap_with_hash(payload_canonical: str, payload_dict: dict[str, Any]) -> str:
    """Wrap the canonical payload in an outer envelope with an
    embedded ``content_sha256`` so value-level tampering is detectable
    on load. The hash covers the inner payload bytes only."""
    h = hashlib.sha256(payload_canonical.encode("utf-8")).hexdigest()
    wrapper = {"content_sha256": h, "manifest": payload_dict}
    return json.dumps(wrapper, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def dump_manifest(manifest: CandidateManifest, path: Path) -> None:
    """Write manifest as deterministic JSON wrapped in a content-
    hashed envelope, mode 0444. Refuses to overwrite an existing
    file — Plan §6.3 / RFC §4 immutability.

    On-disk shape::

        {
          "content_sha256": "<sha256 of inner manifest payload>",
          "manifest": { ... CandidateManifest fields ... }
        }
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"manifest already exists at {path}; manifests are "
            "immutable once written"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_dict = manifest_to_dict(manifest)
    payload_canonical = json.dumps(
        payload_dict, indent=2, sort_keys=True, ensure_ascii=False,
    )
    text = _wrap_with_hash(payload_canonical, payload_dict)
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o444)


def _verify_envelope_and_unwrap(raw: Any) -> dict[str, Any]:
    """Validate the wrapped layout and return the inner manifest
    payload. Raises :class:`ManifestIntegrityError` on:

      - non-dict top level
      - missing ``content_sha256`` or ``manifest`` keys (bare layout
        is rejected — use :func:`manifest_from_dict` directly when a
        test needs the bare path)
      - malformed ``content_sha256``
      - SHA-256 of the canonical inner payload not matching the
        recorded value (value-level tamper detection)
    """
    if not isinstance(raw, dict):
        raise ManifestIntegrityError(
            "manifest file top-level must be a JSON object"
        )
    if "content_sha256" not in raw or "manifest" not in raw:
        raise ManifestIntegrityError(
            "manifest is bare (no `content_sha256` envelope); "
            "registry-managed manifests must be wrapped — use "
            "manifest_from_dict() directly if you need the bare path"
        )
    expected = raw["content_sha256"]
    if not isinstance(expected, str) or not expected:
        raise ManifestIntegrityError(
            f"content_sha256 must be a non-empty string, got {type(expected)}"
        )
    inner = raw["manifest"]
    if not isinstance(inner, dict):
        raise ManifestIntegrityError(
            "envelope `manifest` field must be a JSON object"
        )
    canonical = json.dumps(
        inner, indent=2, sort_keys=True, ensure_ascii=False,
    )
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual != expected:
        raise ManifestIntegrityError(
            f"manifest content_sha256 mismatch: stored "
            f"{expected[:16]}…, recomputed {actual[:16]}…"
        )
    return inner


def load_manifest(path: Path) -> CandidateManifest:
    """Strict manifest loader.

    On-disk shape MUST be the wrapped envelope ``{"content_sha256":
    ..., "manifest": {...}}``. The hash is verified BEFORE unwrapping;
    a mismatch raises :class:`ManifestIntegrityError`. After the
    envelope is verified, the inner payload is parsed by
    :func:`manifest_from_dict` (which still enforces the schema-major
    contract).

    Hand-authored test fixtures that need the bare layout should call
    :func:`manifest_from_dict` directly; the registry layer always
    writes (and reads) the wrapped layout.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ManifestIntegrityError(
            f"manifest is not valid JSON: {e}"
        ) from e
    payload = _verify_envelope_and_unwrap(raw)
    return manifest_from_dict(payload)


def verify_manifest_unchanged(path: Path) -> bool:
    """Returns True when the file at ``path`` is a wrapped manifest
    whose ``content_sha256`` matches the recomputed hash of the inner
    payload. Returns False on any failure mode — tamper, missing
    envelope, malformed JSON, missing file, schema-version mismatch.

    Implemented as ``load_manifest`` + exception trap; the strict
    loader is the single source of truth for "is this file
    untampered".
    """
    try:
        load_manifest(Path(path))
    except (ManifestIntegrityError, FileNotFoundError, KeyError, ValueError):
        return False
    except OSError:
        return False
    return True
