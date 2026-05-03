"""End-to-end smoke check for HedgeRock Phase 1 — no MT5 required.

Spins up the decision server in-process with a scripted mock provider,
hits every endpoint, validates the cache writer round-trip, and prints
a pass/fail summary. Exit code 0 = green, non-zero = something broke.

Usage:
    python scripts/smoke_hedgerock_phase1.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from smc.hedgerock.cache_writer import write_envelope
from smc.hedgerock.decision_server import PrevRegimeStore, create_app
from smc.hedgerock.mock_provider import (
    ScriptedMockProvider,
    default_static_features,
)
from smc.hedgerock.schemas import SignalEnvelope


def _check(condition: bool, label: str) -> bool:
    flag = "OK   " if condition else "FAIL "
    print(f"  [{flag}] {label}")
    return condition


def main() -> int:
    print("HedgeRock Phase 1 smoke check\n")

    # 1. Boot the decision server
    print("Step 1: Boot decision server with scripted provider")
    provider = ScriptedMockProvider(
        [
            default_static_features("TREND_UP"),
            default_static_features("TREND_DOWN"),  # extreme reversal → 7200s lock
            default_static_features("TREND_DOWN"),  # same regime → no lock
        ]
    )
    store = PrevRegimeStore()
    app = create_app(provider, store)
    client = TestClient(app)
    _check(True, "decision server booted")

    # 2. /healthz
    print("\nStep 2: /healthz")
    h = client.get("/healthz")
    pass_ok = True
    pass_ok &= _check(h.status_code == 200, f"status 200 (got {h.status_code})")
    pass_ok &= _check(h.json()["status"] == "ok", "body status=ok")

    # 3. First /signal — no prev → no lock
    print("\nStep 3: First /signal call (no prior regime)")
    r1 = client.get("/signal", params={"symbol": "XAUUSD"})
    pass_ok &= _check(r1.status_code == 200, f"status 200 (got {r1.status_code})")
    body1 = r1.json()
    pass_ok &= _check(body1["regime"] == "trend_up", f"regime=trend_up (got {body1['regime']})")
    pass_ok &= _check(body1["prev_regime"] is None, "prev_regime is None")
    pass_ok &= _check(
        body1["transition_lock_until_ts"] is None, "no transition lock"
    )

    # 4. Second /signal — extreme reversal → max lock
    print("\nStep 4: Second /signal call (TREND_UP → TREND_DOWN extreme reversal)")
    r2 = client.get("/signal", params={"symbol": "XAUUSD"})
    pass_ok &= _check(r2.status_code == 200, f"status 200 (got {r2.status_code})")
    body2 = r2.json()
    pass_ok &= _check(body2["prev_regime"] == "trend_up", "prev_regime=trend_up")
    pass_ok &= _check(body2["regime"] == "trend_down", "regime=trend_down")
    pass_ok &= _check(
        body2["transition_lock_until_ts"] is not None,
        "transition lock applied",
    )

    # 5. Third /signal — same regime → no new lock
    print("\nStep 5: Third /signal call (TREND_DOWN → TREND_DOWN, same regime)")
    r3 = client.get("/signal", params={"symbol": "XAUUSD"})
    pass_ok &= _check(r3.status_code == 200, f"status 200 (got {r3.status_code})")
    body3 = r3.json()
    pass_ok &= _check(
        body3["transition_lock_until_ts"] is None, "no new lock for same regime"
    )

    # 6. Cache writer round-trip
    print("\nStep 6: Cache writer round-trip")
    envelope = SignalEnvelope.model_validate(body2)
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "RegimeCache.json"
        written = write_envelope(envelope, target)
        pass_ok &= _check(written.exists(), f"file written to {written}")
        loaded = json.loads(written.read_text(encoding="utf-8"))
        pass_ok &= _check(
            loaded == envelope.model_dump(mode="json"),
            "round-tripped JSON matches envelope",
        )

    # 7. Unknown symbol → 404
    print("\nStep 7: Unknown symbol returns 404")
    r404 = client.get("/signal", params={"symbol": "UNKNOWN"})
    pass_ok &= _check(r404.status_code == 404, f"status 404 (got {r404.status_code})")

    # 8. /status reflects regime
    print("\nStep 8: /status reflects last seen regime")
    s = client.get("/status").json()
    pass_ok &= _check(
        s["tracked_regimes"]["XAUUSD"] == "TREND_DOWN",
        f"tracked_regimes XAUUSD=TREND_DOWN (got {s['tracked_regimes']['XAUUSD']})",
    )

    print("\n" + ("=" * 50))
    if pass_ok:
        print("ALL CHECKS PASSED")
        return 0
    print("SMOKE CHECK FAILED — see [FAIL] lines above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
