"""Scientific verification claims and convergence evidence.

Numerical execution and scientific trust are deliberately separate.  A
successful solve establishes that a quantity was computed; a verification
claim records why that quantity may be accepted, rejected, or remains
inconclusive for a declared use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log
from typing import Iterable, Literal, Mapping

import numpy as np


ClaimStatus = Literal["passed", "failed", "inconclusive"]
ClaimKind = Literal["verification", "validation"]
TrustLevel = Literal[
    "not_computed",
    "computed",
    "converged",
    "verified",
    "validated",
]

_TRUST_ORDER = {
    "not_computed": 0,
    "computed": 1,
    "converged": 2,
    "verified": 3,
    "validated": 4,
}


def trust_rank(level: str) -> int:
    """Return the ordered rank of one public trust level."""

    try:
        return _TRUST_ORDER[str(level)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown trust level {level!r}; expected {tuple(_TRUST_ORDER)}."
        ) from exc


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class VerificationClaim:
    """One explicit, machine-readable scientific acceptance claim."""

    name: str
    observable: str
    reference: str
    status: ClaimStatus
    criterion: str
    kind: ClaimKind = "verification"
    actual: object | None = None
    expected: object | None = None
    relative_tolerance: float | None = None
    absolute_tolerance: float | None = None
    validity_domain: str = ""
    evidence: Mapping[str, object] = field(default_factory=dict)
    message: str = ""

    def __post_init__(self) -> None:
        for attribute in ("name", "observable", "reference", "criterion"):
            if not str(getattr(self, attribute)).strip():
                raise ValueError(f"VerificationClaim.{attribute} must not be empty.")
        if self.status not in {"passed", "failed", "inconclusive"}:
            raise ValueError("Claim status must be passed, failed, or inconclusive.")
        if self.kind not in {"verification", "validation"}:
            raise ValueError("Claim kind must be verification or validation.")
        for attribute in ("relative_tolerance", "absolute_tolerance"):
            value = getattr(self, attribute)
            if value is not None and (not isfinite(value) or value < 0.0):
                raise ValueError(f"{attribute} must be finite and nonnegative.")

    @classmethod
    def compare(
        cls,
        *,
        name: str,
        observable: str,
        actual,
        expected,
        reference: str,
        relative_tolerance: float = 0.0,
        absolute_tolerance: float = 0.0,
        validity_domain: str = "",
        applicable: bool = True,
        kind: ClaimKind = "verification",
        evidence: Mapping[str, object] | None = None,
    ) -> "VerificationClaim":
        """Compare an observable with a reference under explicit tolerances."""

        actual_array = np.asarray(actual, dtype=float)
        expected_array = np.asarray(expected, dtype=float)
        finite = bool(
            np.all(np.isfinite(actual_array))
            and np.all(np.isfinite(expected_array))
        )
        accepted = finite and bool(
            np.allclose(
                actual_array,
                expected_array,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
        )
        if not applicable:
            status: ClaimStatus = "inconclusive"
            message = "The declared reference is outside its validity domain."
        elif accepted:
            status = "passed"
            message = "The observable satisfies the declared numerical contract."
        else:
            status = "failed"
            message = "The observable is outside the declared numerical contract."
        return cls(
            name=name,
            observable=observable,
            reference=reference,
            status=status,
            criterion=(
                f"allclose(rtol={float(relative_tolerance):g}, "
                f"atol={float(absolute_tolerance):g})"
            ),
            kind=kind,
            actual=actual_array,
            expected=expected_array,
            relative_tolerance=float(relative_tolerance),
            absolute_tolerance=float(absolute_tolerance),
            validity_domain=validity_domain,
            evidence={} if evidence is None else dict(evidence),
            message=message,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "observable": self.observable,
            "reference": self.reference,
            "status": self.status,
            "criterion": self.criterion,
            "kind": self.kind,
            "actual": _json_value(self.actual),
            "expected": _json_value(self.expected),
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "validity_domain": self.validity_domain,
            "evidence": _json_value(self.evidence),
            "message": self.message,
        }


@dataclass(frozen=True)
class VerificationReport:
    """Trust decision derived from execution state and scientific claims."""

    claims: tuple[VerificationClaim, ...] = ()
    computed: bool = True
    converged: bool = False
    scope: str = "simulation"

    def __post_init__(self) -> None:
        object.__setattr__(self, "claims", tuple(self.claims))
        if self.converged and not self.computed:
            raise ValueError("A converged report must also be computed.")
        if not str(self.scope).strip():
            raise ValueError("VerificationReport.scope must not be empty.")

    @property
    def trust_level(self) -> TrustLevel:
        if not self.computed:
            return "not_computed"
        if not self.converged:
            return "computed"
        if not self.claims or any(item.status != "passed" for item in self.claims):
            return "converged"
        if any(item.kind == "validation" for item in self.claims):
            return "validated"
        return "verified"

    @property
    def acceptable(self) -> bool:
        """Return true only when every declared claim passes."""

        return bool(self.claims) and all(
            item.status == "passed" for item in self.claims
        )

    def require(self, minimum: TrustLevel = "verified") -> None:
        if trust_rank(self.trust_level) < trust_rank(minimum):
            failed = tuple(
                item.name for item in self.claims if item.status != "passed"
            )
            raise RuntimeError(
                f"Trust level {self.trust_level!r} is below required "
                f"{minimum!r}; unresolved claims={failed}."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "agentfem.verification-report",
            "schema_version": "0.1.0",
            "scope": self.scope,
            "computed": self.computed,
            "converged": self.converged,
            "trust_level": self.trust_level,
            "acceptable": self.acceptable,
            "claims": [item.as_dict() for item in self.claims],
        }


@dataclass(frozen=True)
class ConvergenceSample:
    """One observable evaluated at a declared discretization size."""

    characteristic_size: float
    value: object
    label: str = ""
    evidence: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        size = float(self.characteristic_size)
        values = np.asarray(self.value, dtype=float)
        if not isfinite(size) or size <= 0.0:
            raise ValueError("Convergence characteristic_size must be positive.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Convergence sample values must be finite.")
        object.__setattr__(self, "characteristic_size", size)
        object.__setattr__(self, "value", values.copy())


@dataclass(frozen=True)
class ConvergenceStudy:
    """A coarse-to-fine mesh or time-step convergence sequence."""

    name: str
    observable: str
    samples: tuple[ConvergenceSample, ...]
    discretization: str = "mesh"

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if len(samples) < 2:
            raise ValueError("ConvergenceStudy requires at least two samples.")
        sizes = tuple(item.characteristic_size for item in samples)
        if any(fine >= coarse for coarse, fine in zip(sizes, sizes[1:])):
            raise ValueError(
                "Convergence samples must be ordered coarse-to-fine with "
                "strictly decreasing characteristic_size."
            )
        shapes = {np.asarray(item.value).shape for item in samples}
        if len(shapes) != 1:
            raise ValueError("Convergence sample values must share one shape.")
        object.__setattr__(self, "samples", samples)

    @property
    def finest_relative_change(self) -> float:
        coarse = np.asarray(self.samples[-2].value, dtype=float)
        fine = np.asarray(self.samples[-1].value, dtype=float)
        numerator = float(np.linalg.norm(fine - coarse))
        denominator = max(float(np.linalg.norm(fine)), np.finfo(float).tiny)
        return numerator / denominator

    @property
    def observed_order(self) -> float | None:
        """Return an observed order for the last three uniformly refined runs."""

        if len(self.samples) < 3:
            return None
        first, second, third = self.samples[-3:]
        ratio_1 = first.characteristic_size / second.characteristic_size
        ratio_2 = second.characteristic_size / third.characteristic_size
        if not np.isclose(ratio_1, ratio_2, rtol=1.0e-6, atol=1.0e-12):
            return None
        difference_1 = float(np.linalg.norm(first.value - second.value))
        difference_2 = float(np.linalg.norm(second.value - third.value))
        if difference_1 <= 0.0 or difference_2 <= 0.0:
            return None
        return log(difference_1 / difference_2) / log(ratio_1)

    def verify(
        self,
        *,
        maximum_relative_change: float,
        minimum_observed_order: float | None = None,
        reference: str = "successive discretization refinement",
    ) -> VerificationClaim:
        """Create a claim without treating solver convergence as mesh convergence."""

        limit = float(maximum_relative_change)
        if not isfinite(limit) or limit < 0.0:
            raise ValueError("maximum_relative_change must be nonnegative.")
        order = self.observed_order
        change_passes = self.finest_relative_change <= limit
        order_available = minimum_observed_order is None or order is not None
        order_passes = (
            minimum_observed_order is None
            or (order is not None and order >= float(minimum_observed_order))
        )
        status: ClaimStatus
        if not order_available:
            status = "inconclusive"
        else:
            status = "passed" if change_passes and order_passes else "failed"
        criterion = f"finest relative change <= {limit:g}"
        if minimum_observed_order is not None:
            criterion += f" and observed order >= {float(minimum_observed_order):g}"
        return VerificationClaim(
            name=self.name,
            observable=self.observable,
            reference=reference,
            status=status,
            criterion=criterion,
            actual=self.finest_relative_change,
            expected=limit,
            evidence=self.as_dict(),
            message=(
                "The discretization sequence satisfies the declared contract."
                if status == "passed"
                else "The discretization sequence does not establish the declared accuracy."
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "convergence_study",
            "name": self.name,
            "observable": self.observable,
            "discretization": self.discretization,
            "sample_count": len(self.samples),
            "characteristic_sizes": [
                item.characteristic_size for item in self.samples
            ],
            "values": [_json_value(item.value) for item in self.samples],
            "finest_relative_change": self.finest_relative_change,
            "observed_order": self.observed_order,
            "samples": [
                {
                    "label": item.label,
                    "evidence": _json_value(item.evidence),
                }
                for item in self.samples
            ],
        }


def report(
    *claims: VerificationClaim,
    computed: bool = True,
    converged: bool = True,
    scope: str = "simulation",
) -> VerificationReport:
    """Concise public constructor for a verification report."""

    return VerificationReport(
        claims=tuple(claims),
        computed=computed,
        converged=converged,
        scope=scope,
    )


def convergence_study(
    name: str,
    observable: str,
    samples: Iterable[ConvergenceSample],
    *,
    discretization: str = "mesh",
) -> ConvergenceStudy:
    return ConvergenceStudy(
        name=name,
        observable=observable,
        samples=tuple(samples),
        discretization=discretization,
    )


__all__ = [
    "ClaimKind",
    "ClaimStatus",
    "ConvergenceSample",
    "ConvergenceStudy",
    "TrustLevel",
    "VerificationClaim",
    "VerificationReport",
    "convergence_study",
    "report",
    "trust_rank",
]
