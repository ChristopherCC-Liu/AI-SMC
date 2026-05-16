"""P1-1 — Bayesian confidence calibrator (Beta–Bernoulli conjugate).

Each parameter class carries an independent ``Beta(α, β)`` posterior.
The prior is uninformative ``Beta(1, 1)`` (uniform on [0, 1]); each
paper-test outcome (1 = success, 0 = failure) updates the posterior
to ``Beta(α + n_success, β + n_failure)``.

Calibrated confidence is the posterior mean ``α / (α + β)``.

JSON serialisable + deterministic; no scipy dependency (we only
need first moments of the Beta family).

Isolation: this module does NOT import ``rule_engine`` or the
Tier-1 unsealed prod modules. It's a pure data sidecar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


__all__ = [
    "BayesianCalibrator",
    "DEFAULT_CALIBRATOR_PATH_BASENAME",
    "DEFAULT_PARAMETER_CLASSES",
    "ParameterClassPrior",
    "load_calibrator",
]


DEFAULT_PARAMETER_CLASSES: tuple[str, ...] = (
    "confidence_threshold_observe",
    "confidence_threshold_aggressive",
    "confidence_threshold_range_2",
    "halt_expiry_observe_hours",
)
DEFAULT_CALIBRATOR_PATH_BASENAME: str = "calibrator_state.json"


@dataclass(frozen=True)
class ParameterClassPrior:
    """Beta(α, β) posterior for one parameter class.

    Posterior mean = α / (α + β). Variance is computed for the
    JSON-emitted summary only; downstream consumers use the mean.
    """

    parameter_class: str
    alpha: float = 1.0
    beta: float = 1.0
    n_observations: int = 0
    last_updated_at: str | None = None

    @property
    def mean(self) -> float:
        denom = self.alpha + self.beta
        if denom <= 0:
            return 0.5
        return self.alpha / denom

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        denom = (a + b) ** 2 * (a + b + 1.0)
        if denom <= 0:
            return 0.0
        return (a * b) / denom

    def update(
        self, *, n_success: int, n_failure: int,
        now: datetime | None = None,
    ) -> "ParameterClassPrior":
        if n_success < 0 or n_failure < 0:
            raise ValueError("counts must be non-negative")
        ts = (now or datetime.now(timezone.utc)).isoformat()
        return replace(
            self,
            alpha=self.alpha + float(n_success),
            beta=self.beta + float(n_failure),
            n_observations=self.n_observations + n_success + n_failure,
            last_updated_at=ts,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_class": self.parameter_class,
            "alpha": self.alpha,
            "beta": self.beta,
            "n_observations": self.n_observations,
            "last_updated_at": self.last_updated_at,
            "mean": round(self.mean, 6),
            "variance": round(self.variance, 9),
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "ParameterClassPrior":
        return cls(
            parameter_class=str(obj["parameter_class"]),
            alpha=float(obj.get("alpha", 1.0)),
            beta=float(obj.get("beta", 1.0)),
            n_observations=int(obj.get("n_observations", 0)),
            last_updated_at=obj.get("last_updated_at"),
        )


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


@dataclass
class BayesianCalibrator:
    """Holds one ``ParameterClassPrior`` per known class plus a JSON
    save/load round-trip."""

    priors: dict[str, ParameterClassPrior] = field(default_factory=dict)

    @classmethod
    def with_uninformative_priors(
        cls,
        parameter_classes: Iterable[str] = DEFAULT_PARAMETER_CLASSES,
    ) -> "BayesianCalibrator":
        return cls(priors={
            pc: ParameterClassPrior(parameter_class=pc)
            for pc in parameter_classes
        })

    def get(self, parameter_class: str) -> ParameterClassPrior:
        prior = self.priors.get(parameter_class)
        if prior is None:
            prior = ParameterClassPrior(parameter_class=parameter_class)
            self.priors[parameter_class] = prior
        return prior

    def calibrated_confidence(self, parameter_class: str) -> float:
        return self.get(parameter_class).mean

    def update_from_outcome(
        self,
        *,
        parameter_class: str,
        success: bool,
        weight: int = 1,
        now: datetime | None = None,
    ) -> ParameterClassPrior:
        if weight < 1:
            raise ValueError("weight must be >= 1")
        prior = self.get(parameter_class)
        new = prior.update(
            n_success=weight if success else 0,
            n_failure=0 if success else weight,
            now=now,
        )
        self.priors[parameter_class] = new
        return new

    def update_from_paper_test_summary(
        self,
        *,
        parameter_class: str,
        n_trades: int,
        pnl_sum: float,
        max_drawdown: float,
        drawdown_floor: float = -50.0,
        now: datetime | None = None,
    ) -> ParameterClassPrior:
        """Convert one paper-test summary into a Bernoulli sample.

        success = ``pnl_sum > 0`` AND ``max_drawdown >= drawdown_floor``
        AND ``n_trades > 0``. Edge cases (no trades) are recorded as
        a single failure so a stalled candidate's posterior decays
        toward ``β > α`` over repeated runs.
        """
        if n_trades <= 0:
            success = False
        else:
            success = pnl_sum > 0 and max_drawdown >= drawdown_floor
        return self.update_from_outcome(
            parameter_class=parameter_class,
            success=success, weight=1, now=now,
        )

    # ------------------------------------------------------------------
    # JSON persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "bayesian_calibrator/v1",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "priors": [p.to_dict() for p in self.priors.values()],
        }

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> "BayesianCalibrator":
        priors_list = obj.get("priors", []) or []
        priors: dict[str, ParameterClassPrior] = {}
        for raw in priors_list:
            prior = ParameterClassPrior.from_dict(raw)
            priors[prior.parameter_class] = prior
        return cls(priors=priors)

    def save(self, path: Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: Path) -> "BayesianCalibrator":
        p = Path(path)
        if not p.exists():
            return cls.with_uninformative_priors()
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls.with_uninformative_priors()
        return cls.from_dict(obj)


def load_calibrator(path: Path | None = None) -> BayesianCalibrator:
    """Convenience wrapper. Returns an uninformative calibrator
    when ``path`` is None or the file is missing."""
    if path is None:
        return BayesianCalibrator.with_uninformative_priors()
    return BayesianCalibrator.load(path)
