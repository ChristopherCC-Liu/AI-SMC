"""A/B arm classifier for the post-bake calibration runner (R10 cal-1).

Splits a flat journal-entry stream into named A/B arms based on each
trade's recorded R10 feature flags + magic number. The downstream
``run_promotion_gate.py`` runner then computes per-arm PF/CI/win-rate
and the consec_loss histogram needed to inform P3.1 sizing decay.

Design contract (locked with foundation-impl-lead pre-implementation):

- **(A) ArmMembership carries pnl_usd**: Each classified trade view
  satisfies BOTH the local ``ArmMembership`` Protocol AND
  ``smc.eval.promotion_gate.TradeLike`` via structural subtyping. The
  runner can pass classified arms straight into ``evaluate_promotion``
  with no DTO conversion. The single ``pnl_usd`` field threads the
  whole pipeline; no duplication.
- **(A2) Strict matching default + Literal kwarg**: ``ArmDefinition.matching_mode``
  defaults to ``"strict"`` (Phase 1 clean A/B). ``"subset"`` opt-in is
  reserved for Phase 2+ ablation arms where a treatment cell has only
  one or two flags ON but allows the others to vary.
- **(A3) Symmetric diagnostic buckets**: ``_control_with_unexpected_flag_on``
  catches a deployment-bug trade entered on the control magic but with
  one or more R10 flags accidentally ON. ``_treatment_with_missing_flag``
  is the mirror — treatment magic but expected flag(s) OFF. ``_unmatched``
  catches everything else (e.g., manual operator entries on a
  third magic). Three buckets means deployment bugs surface as concrete
  trade counts, not silent reclassification.

- **(α) Backward-compat for pre-cal-0b legacy records**: Records written
  before R10 cal-0b are missing the ``loss_count_in_window_at_open``
  field. The ``_extract_loss_count_from_record`` helper returns ``None``
  for those records — calibration v1 still runs against mixed
  pre/post-cal-0b journals.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

__all__ = [
    "FeatureFlagSnapshot",
    "ArmMembership",
    "TradeView",
    "ArmDefinition",
    "ClassificationResult",
    "classify_trades",
    "DIAGNOSTIC_CONTROL_UNEXPECTED",
    "DIAGNOSTIC_TREATMENT_MISSING",
    "DIAGNOSTIC_UNMATCHED",
]


# ---------------------------------------------------------------------------
# Public Protocols + dataclasses
# ---------------------------------------------------------------------------


class FeatureFlagSnapshot(Protocol):
    """Read-only view of which R10 feature flags were ON for a given trade.

    ``consec_loss_window_size`` is an ``int`` (not bool) because the field
    has a numeric domain rather than a binary ON/OFF — control may be at
    legacy-3, treatment at the R10 default 6. Strict matching compares
    by equality.
    """

    range_trend_filter_enabled: bool
    range_ai_regime_gate_enabled: bool
    spread_gate_enabled: bool
    persistent_dd_breaker_enabled: bool
    consec_loss_window_size: int


class ArmMembership(Protocol):
    """A trade record annotated with its A/B arm + flag snapshot.

    Satisfies ``smc.eval.promotion_gate.TradeLike`` via the ``pnl_usd``
    attribute (Python's structural subtyping doesn't require explicit
    inheritance), so instances flow directly into ``evaluate_promotion``
    without unwrapping.
    """

    trade_id: str
    timestamp_utc: str
    arm_label: str
    flag_snapshot: FeatureFlagSnapshot
    pnl_usd: float
    loss_count_in_window_at_open: int | None


@dataclass(frozen=True)
class _FlagSnapshot:
    """Concrete frozen-dataclass implementation of ``FeatureFlagSnapshot``.

    Hidden behind the Protocol — callers should annotate against
    ``FeatureFlagSnapshot`` rather than this concrete type.
    """

    range_trend_filter_enabled: bool
    range_ai_regime_gate_enabled: bool
    spread_gate_enabled: bool
    persistent_dd_breaker_enabled: bool
    consec_loss_window_size: int


@dataclass(frozen=True)
class TradeView:
    """Concrete frozen-dataclass implementation of ``ArmMembership``.

    Concrete type for the classifier's output. Tests can construct
    instances directly; production callers receive these via
    :func:`classify_trades`.
    """

    trade_id: str
    timestamp_utc: str
    arm_label: str
    flag_snapshot: FeatureFlagSnapshot
    pnl_usd: float
    loss_count_in_window_at_open: int | None


@dataclass(frozen=True)
class ArmDefinition:
    """Defines an A/B arm by its required flag values + magic number.

    - ``required_flags`` is a subset of ``FeatureFlagSnapshot`` field
      names mapped to their expected values. In ``"strict"`` mode the
      classifier requires EXACT match on all named keys; in ``"subset"``
      mode any unspecified field is allowed to vary (Phase 2+ ablation).
    - ``required_magic`` pins the arm to a specific MT5 magic number
      (control vs treatment process identity). Set to ``None`` to skip
      the magic check (rarely useful — keep the safety check on).
    """

    arm_label: str
    required_flags: Mapping[str, bool | int]
    required_magic: int | None = None
    matching_mode: Literal["strict", "subset"] = "strict"


@dataclass(frozen=True)
class ClassificationResult:
    """Output of :func:`classify_trades`.

    ``arms`` maps each defined arm's label to its trades in journal order.
    ``diagnostics`` separately collects trades that fell into one of the
    three deployment-bug buckets.
    """

    arms: Mapping[str, Sequence[TradeView]]
    diagnostics: Mapping[str, Sequence[TradeView]]


# Diagnostic bucket labels — exposed as constants so test code and the
# downstream runner can reference them by name rather than string literal.
DIAGNOSTIC_CONTROL_UNEXPECTED = "_control_with_unexpected_flag_on"
DIAGNOSTIC_TREATMENT_MISSING = "_treatment_with_missing_flag"
DIAGNOSTIC_UNMATCHED = "_unmatched"


# ---------------------------------------------------------------------------
# Field-extraction helpers (per-record robustness against legacy formats)
# ---------------------------------------------------------------------------


def _extract_pnl_usd(entry: Mapping[str, object]) -> float:
    """Read ``pnl_usd`` as a top-level float from a journal entry.

    Per the ``live_demo.py:2013`` grep-contract comment, the field is
    canonical and stable. This helper exists only to centralise the
    ``float()`` cast + missing-key handling, not to abstract the path.
    """
    raw = entry.get("pnl_usd", 0.0)
    return float(raw) if raw is not None else 0.0


def _extract_loss_count_from_record(entry: Mapping[str, object]) -> int | None:
    """Read the cal-0b ``loss_count_in_window_at_open`` field.

    Returns ``None`` for legacy pre-cal-0b records (missing field) so the
    calibration v1 runs against any mix of pre/post journals. Cast to
    ``int`` defensively in case the JSON parser deserialised a numeric
    string.
    """
    raw = entry.get("loss_count_in_window_at_open")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _extract_flag_snapshot(entry: Mapping[str, object]) -> FeatureFlagSnapshot | None:
    """Build a ``_FlagSnapshot`` from foundation cal-0's per-trade flags dict.

    Returns ``None`` if the entry is missing the ``flags`` sub-dict
    (legacy pre-cal-0 record) — the classifier falls back to the
    boot-time snapshot in that case.

    Foundation's cal-0 commits the per-trade dict under the key
    ``flags`` (per their schema_version=2 spec). Each value is read with
    a typed default; missing keys are treated as ``False`` for booleans
    and ``0`` for the integer window size. A future cal-0 evolution
    that nests differently would need this helper updated.
    """
    raw_flags = entry.get("flags")
    if not isinstance(raw_flags, Mapping):
        return None
    return _FlagSnapshot(
        range_trend_filter_enabled=bool(raw_flags.get("range_trend_filter_enabled", False)),
        range_ai_regime_gate_enabled=bool(raw_flags.get("range_ai_regime_gate_enabled", False)),
        spread_gate_enabled=bool(raw_flags.get("spread_gate_enabled", False)),
        persistent_dd_breaker_enabled=bool(raw_flags.get("persistent_dd_breaker_enabled", False)),
        consec_loss_window_size=int(raw_flags.get("consec_loss_window_size", 0)),
    )


# ---------------------------------------------------------------------------
# Arm-matching logic
# ---------------------------------------------------------------------------


def _flags_match(
    snapshot: FeatureFlagSnapshot,
    required: Mapping[str, bool | int],
    *,
    mode: Literal["strict", "subset"],
) -> bool:
    """Check whether ``snapshot`` matches ``required`` under the given mode.

    - ``"strict"``: EVERY field of ``FeatureFlagSnapshot`` must equal its
      ``required`` value. If a field is not in ``required`` we treat the
      arm definition as incomplete and fail-closed (no match).
    - ``"subset"``: only the keys named in ``required`` are checked;
      everything else may vary. Phase 2 ablation arms use this.
    """
    if mode == "strict":
        # Strict requires every documented FeatureFlagSnapshot field to
        # appear in required_flags. Fail-closed if the arm definition is
        # under-specified (better to crash loud than silently match).
        expected_fields = {
            "range_trend_filter_enabled",
            "range_ai_regime_gate_enabled",
            "spread_gate_enabled",
            "persistent_dd_breaker_enabled",
            "consec_loss_window_size",
        }
        if set(required) != expected_fields:
            return False
    for field_name, expected in required.items():
        actual = getattr(snapshot, field_name, None)
        if actual != expected:
            return False
    return True


def _classify_one_trade(
    entry: Mapping[str, object],
    *,
    arm_definitions: Sequence[ArmDefinition],
    fallback_snapshot: FeatureFlagSnapshot | None,
    control_magic: int | None,
    treatment_magic: int | None,
) -> tuple[str, TradeView]:
    """Assign a single journal entry to an arm or diagnostic bucket.

    Returns ``(label, view)`` where ``label`` is either the matched arm
    name or one of the three ``DIAGNOSTIC_*`` constants. The classifier
    walks ``arm_definitions`` in order; first match wins. If no arm
    matches and the trade's magic identifies it as control/treatment
    with the wrong flag state, it lands in the corresponding diagnostic
    bucket. Otherwise it lands in ``_unmatched``.
    """
    snapshot = _extract_flag_snapshot(entry) or fallback_snapshot
    magic = entry.get("magic")
    try:
        magic_int = int(magic) if magic is not None else None
    except (TypeError, ValueError):
        magic_int = None

    # Candidate arm walk: first arm whose required_magic AND flags match wins.
    matched_label: str | None = None
    for arm in arm_definitions:
        if arm.required_magic is not None and magic_int != arm.required_magic:
            continue
        if snapshot is None:
            # No flags to compare; can't match in strict mode.
            continue
        if _flags_match(snapshot, arm.required_flags, mode=arm.matching_mode):
            matched_label = arm.arm_label
            break

    label = matched_label or _diagnostic_bucket(
        magic_int=magic_int,
        snapshot=snapshot,
        control_magic=control_magic,
        treatment_magic=treatment_magic,
    )

    view = TradeView(
        trade_id=str(entry.get("trade_id") or entry.get("ticket") or entry.get("time", "")),
        timestamp_utc=str(entry.get("time", "")),
        arm_label=label,
        flag_snapshot=snapshot or _empty_snapshot(),
        pnl_usd=_extract_pnl_usd(entry),
        loss_count_in_window_at_open=_extract_loss_count_from_record(entry),
    )
    return label, view


def _diagnostic_bucket(
    *,
    magic_int: int | None,
    snapshot: FeatureFlagSnapshot | None,
    control_magic: int | None,
    treatment_magic: int | None,
) -> str:
    """Pick the right diagnostic bucket for an unmatched trade.

    Rules:
    - magic == control_magic AND any flag is ON → control_with_unexpected_flag_on
    - magic == treatment_magic AND any expected flag is OFF → treatment_with_missing_flag
    - otherwise → unmatched (third magic, manual entry, or no snapshot)
    """
    if magic_int is not None and snapshot is not None:
        if magic_int == control_magic and _any_flag_on(snapshot):
            return DIAGNOSTIC_CONTROL_UNEXPECTED
        if magic_int == treatment_magic and not _all_flags_on(snapshot):
            return DIAGNOSTIC_TREATMENT_MISSING
    return DIAGNOSTIC_UNMATCHED


def _any_flag_on(snapshot: FeatureFlagSnapshot) -> bool:
    """True if at least one R10 boolean flag is ON in the snapshot."""
    return (
        snapshot.range_trend_filter_enabled
        or snapshot.range_ai_regime_gate_enabled
        or snapshot.spread_gate_enabled
        or snapshot.persistent_dd_breaker_enabled
    )


def _all_flags_on(snapshot: FeatureFlagSnapshot) -> bool:
    """True if all 4 R10 boolean flags are ON in the snapshot.

    consec_loss_window_size is not a boolean — its ``treatment`` value
    being non-default (>3) IS the toggle, so a window_size > 3 contributes
    "treatment-leaning". We treat any window > legacy 3 as the treatment
    setting for the purposes of this diagnostic-bucket heuristic.
    """
    return (
        snapshot.range_trend_filter_enabled
        and snapshot.range_ai_regime_gate_enabled
        and snapshot.spread_gate_enabled
        and snapshot.persistent_dd_breaker_enabled
        and snapshot.consec_loss_window_size > 3
    )


def _empty_snapshot() -> _FlagSnapshot:
    """Return a synthetic all-OFF snapshot for trades with no flags dict.

    Used for diagnostic-bucket trades so the TradeView always carries a
    valid (if uninformative) snapshot — downstream consumers don't have
    to guard against ``None`` on every field access.
    """
    return _FlagSnapshot(
        range_trend_filter_enabled=False,
        range_ai_regime_gate_enabled=False,
        spread_gate_enabled=False,
        persistent_dd_breaker_enabled=False,
        consec_loss_window_size=0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_trades(
    journal_entries: Sequence[Mapping[str, object]],
    *,
    arm_definitions: Sequence[ArmDefinition],
    fallback_snapshot: FeatureFlagSnapshot | None = None,
    control_magic: int | None = None,
    treatment_magic: int | None = None,
) -> ClassificationResult:
    """Group journal entries by arm membership.

    Args:
        journal_entries: Raw journal records (e.g., parsed jsonl rows
            from ``data/journal/live_trades.jsonl``). Each entry should
            be a ``Mapping[str, object]`` with at least ``time``,
            ``magic``, ``pnl_usd`` keys; optionally ``flags`` and
            ``loss_count_in_window_at_open`` (post-cal-0 / cal-0b).
        arm_definitions: Ordered list of arms to attempt matching against.
            First match wins per trade. Empty input yields all-unmatched.
        fallback_snapshot: Boot-time snapshot used when an entry lacks
            its own ``flags`` dict (pre-cal-0 legacy). If ``None``, those
            entries fall through to the diagnostic buckets.
        control_magic / treatment_magic: Used only by the diagnostic
            bucket logic to assign deployment-bug trades to the
            symmetric ``_control_with_unexpected_flag_on`` /
            ``_treatment_with_missing_flag`` buckets.

    Returns:
        :class:`ClassificationResult` with ``arms`` (label → trade list)
        and ``diagnostics`` (bucket → trade list). Per-arm lists preserve
        the input journal order so the runner can compute time-ordered
        statistics like rolling consec-loss histograms.
    """
    arms: dict[str, list[TradeView]] = {arm.arm_label: [] for arm in arm_definitions}
    diagnostics: dict[str, list[TradeView]] = {
        DIAGNOSTIC_CONTROL_UNEXPECTED: [],
        DIAGNOSTIC_TREATMENT_MISSING: [],
        DIAGNOSTIC_UNMATCHED: [],
    }

    for entry in journal_entries:
        label, view = _classify_one_trade(
            entry,
            arm_definitions=arm_definitions,
            fallback_snapshot=fallback_snapshot,
            control_magic=control_magic,
            treatment_magic=treatment_magic,
        )
        if label in arms:
            arms[label].append(view)
        else:
            diagnostics[label].append(view)

    return ClassificationResult(
        arms={k: tuple(v) for k, v in arms.items()},
        diagnostics={k: tuple(v) for k, v in diagnostics.items()},
    )
