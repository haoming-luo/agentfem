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
ClaimKind = Literal["runtime", "verification", "validation"]
QualityPreset = Literal["exploratory", "engineering", "release"]
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
        if self.kind not in {"runtime", "verification", "validation"}:
            raise ValueError(
                "Claim kind must be runtime, verification, or validation."
            )
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
    quality_policy: str | None = None
    minimum_trust_level: TrustLevel | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "claims", tuple(self.claims))
        if self.converged and not self.computed:
            raise ValueError("A converged report must also be computed.")
        if not str(self.scope).strip():
            raise ValueError("VerificationReport.scope must not be empty.")
        if self.minimum_trust_level is not None:
            trust_rank(self.minimum_trust_level)

    @property
    def trust_level(self) -> TrustLevel:
        if not self.computed:
            return "not_computed"
        if not self.converged:
            return "computed"
        if any(item.status != "passed" for item in self.claims):
            return "converged"
        scientific = tuple(
            item for item in self.claims if item.kind != "runtime"
        )
        if not scientific:
            return "converged"
        if any(item.kind == "validation" for item in scientific):
            return "validated"
        return "verified"

    @property
    def acceptable(self) -> bool:
        """Return true only when every declared claim passes."""

        claims_pass = bool(self.claims) and all(
            item.status == "passed" for item in self.claims
        )
        if self.minimum_trust_level is None:
            return claims_pass
        return claims_pass and trust_rank(self.trust_level) >= trust_rank(
            self.minimum_trust_level
        )

    def require(self, minimum: TrustLevel | None = None) -> None:
        selected = minimum or self.minimum_trust_level or "verified"
        if trust_rank(self.trust_level) < trust_rank(selected):
            failed = tuple(
                item.name for item in self.claims if item.status != "passed"
            )
            raise RuntimeError(
                f"Trust level {self.trust_level!r} is below required "
                f"{selected!r}; unresolved claims={failed}."
            )
        failed = tuple(item.name for item in self.claims if item.status != "passed")
        if failed:
            raise RuntimeError(
                f"Quality checks did not pass; unresolved claims={failed}."
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
            "quality_policy": self.quality_policy,
            "minimum_trust_level": self.minimum_trust_level,
            "claims": [item.as_dict() for item in self.claims],
        }


@dataclass(frozen=True)
class QualityPolicy:
    """Low-ceremony acceptance policy for one result or dataset boundary."""

    name: QualityPreset
    minimum_trust_level: TrustLevel
    description: str

    def __post_init__(self) -> None:
        if self.name not in {"exploratory", "engineering", "release"}:
            raise ValueError(f"Unknown quality policy {self.name!r}.")
        trust_rank(self.minimum_trust_level)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "minimum_trust_level": self.minimum_trust_level,
            "description": self.description,
        }


_QUALITY_POLICIES = {
    "exploratory": QualityPolicy(
        "exploratory",
        "computed",
        "Finite, inspectable output from a completed exploratory run.",
    ),
    "engineering": QualityPolicy(
        "engineering",
        "converged",
        "Converged procedure with finite payload and declared runtime checks.",
    ),
    "release": QualityPolicy(
        "release",
        "verified",
        "Engineering checks plus passed scientific reference evidence.",
    ),
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


def quality_policy(value: str | QualityPolicy) -> QualityPolicy:
    """Return one named public quality policy."""

    if isinstance(value, QualityPolicy):
        return value
    selected = str(value).strip().lower().replace("-", "_")
    try:
        return _QUALITY_POLICIES[selected]
    except KeyError as exc:
        raise ValueError(
            f"Unknown quality policy {value!r}; expected "
            f"{tuple(_QUALITY_POLICIES)}."
        ) from exc


def assess(
    result,
    quality: str | QualityPolicy = "engineering",
    *,
    claims: Iterable[VerificationClaim] = (),
    converged: bool | None = None,
    required_quantities: Iterable[str] = (),
    required_histories: Iterable[str] = (),
    required_artifacts: Iterable[str] = (),
    attach: bool = True,
) -> VerificationReport:
    """Apply a quality preset and inexpensive deterministic result checks.

    This function does not manufacture scientific evidence. Runtime checks
    have ``kind='runtime'`` and can establish that a converged result is fit
    for an engineering workflow, but only an explicit verification/reference
    claim can promote it to ``verified`` for the release policy.
    """

    policy = quality_policy(quality)
    status = str(getattr(result, "status", "")).strip().lower()
    computed = status == "completed"
    existing = getattr(result, "verification", None)
    previous_claims = ()
    if isinstance(existing, VerificationReport):
        previous_claims = tuple(
            item for item in existing.claims if item.kind != "runtime"
        )
    scientific_claims = _unique_claims((*previous_claims, *tuple(claims)))
    runtime_claims = _runtime_checks(
        result,
        required_quantities=tuple(required_quantities),
        required_histories=tuple(required_histories),
        required_artifacts=tuple(required_artifacts),
    )
    selected_convergence = (
        _infer_converged(result)
        if converged is None
        else bool(converged)
    )
    selected_convergence = bool(computed and selected_convergence)
    selected_report = VerificationReport(
        claims=(*scientific_claims, *runtime_claims),
        computed=computed,
        converged=selected_convergence,
        scope=getattr(existing, "scope", "simulation"),
        quality_policy=policy.name,
        minimum_trust_level=policy.minimum_trust_level,
    )
    if attach:
        add = getattr(result, "add_verification", None)
        if not callable(add):
            raise TypeError(
                "verification.assess requires a SimulationResult-like object "
                "with add_verification()."
            )
        add(selected_report)
    return selected_report


def _runtime_checks(
    result,
    *,
    required_quantities: tuple[str, ...],
    required_histories: tuple[str, ...],
    required_artifacts: tuple[str, ...],
) -> tuple[VerificationClaim, ...]:
    quantities = getattr(result, "quantities", {})
    fields = getattr(result, "fields", {})
    histories = getattr(result, "histories", {})
    artifacts = getattr(result, "artifacts", {})
    status_ok = str(getattr(result, "status", "")).lower() == "completed"
    claims = [
        _runtime_claim(
            "execution_completed",
            status_ok,
            observable="result.status",
            actual=getattr(result, "status", None),
            expected="completed",
            criterion="status == 'completed'",
        )
    ]
    payload_count = len(quantities) + len(fields) + len(histories) + len(artifacts)
    claims.append(
        _runtime_claim(
            "result_payload_present",
            payload_count > 0,
            observable="registered result payload",
            actual=payload_count,
            expected="> 0",
            criterion="at least one quantity, field, history, or artifact",
        )
    )

    finite, field_evidence = _finite_live_fields(fields)
    claims.append(
        _runtime_claim(
            "finite_live_fields",
            finite,
            observable="live field coefficients",
            actual=field_evidence,
            expected="all finite",
            criterion="all inspectable owned field coefficients are finite",
        )
    )

    required = {
        "quantities": (required_quantities, quantities),
        "histories": (required_histories, histories),
        "artifacts": (required_artifacts, artifacts),
    }
    missing = {
        kind: tuple(name for name in names if name not in records)
        for kind, (names, records) in required.items()
    }
    missing = {kind: names for kind, names in missing.items() if names}
    claims.append(
        _runtime_claim(
            "required_outputs_present",
            not missing,
            observable="declared result contract",
            actual=missing,
            expected="no missing records",
            criterion="all required quantities, histories, and artifacts exist",
        )
    )

    missing_files = tuple(
        name
        for name in required_artifacts
        if name in artifacts and not _artifact_exists(artifacts[name])
    )
    claims.append(
        _runtime_claim(
            "required_artifacts_materialized",
            not missing_files,
            observable="required artifact paths",
            actual=missing_files,
            expected="all materialized",
            criterion="every required artifact path exists",
        )
    )

    execution = getattr(result, "metadata", {}).get("execution")
    if isinstance(execution, Mapping):
        events = execution.get("events", ())
        kinds = tuple(
            item.get("kind") for item in events if isinstance(item, Mapping)
        )
        failed = "step_failed" in kinds
        started = any(kind in kinds for kind in ("step_started", "transient_started"))
        finished = any(
            kind in kinds
            for kind in ("step_completed", "step_paused", "transient_completed")
        )
        claims.append(
            _runtime_claim(
                "execution_trace_complete",
                not failed and (not started or finished),
                observable="structured solve-event trace",
                actual={
                    "event_count": len(kinds),
                    "failed": failed,
                    "started": started,
                    "finished": finished,
                },
                expected="no failure and every started procedure finished",
                criterion="structured event trace has no unfinished or failed step",
            )
        )
    return tuple(claims)


def _runtime_claim(
    name: str,
    passed: bool,
    *,
    observable: str,
    actual,
    expected,
    criterion: str,
) -> VerificationClaim:
    return VerificationClaim(
        name=name,
        observable=observable,
        reference="AgentFEM deterministic runtime integrity contract",
        status="passed" if passed else "failed",
        criterion=criterion,
        kind="runtime",
        actual=actual,
        expected=expected,
        message=(
            "The runtime integrity check passed."
            if passed
            else "The runtime integrity check failed."
        ),
    )


def _finite_live_fields(fields: Mapping[str, object]) -> tuple[bool, dict[str, object]]:
    inspected = []
    nonfinite = []
    for name, record in fields.items():
        selected = getattr(record, "field", None)
        if selected is None:
            continue
        selected = getattr(selected, "value", selected)
        vector = getattr(selected, "x", None)
        values = getattr(vector, "array", None)
        if values is None:
            continue
        array = np.asarray(values)
        space = getattr(selected, "function_space", None)
        dofmap = getattr(space, "dofmap", None)
        index_map = getattr(dofmap, "index_map", None)
        block_size = int(getattr(dofmap, "index_map_bs", 1))
        if index_map is not None:
            array = array[: int(index_map.size_local) * block_size]
        local_finite = bool(np.all(np.isfinite(array)))
        comm = getattr(getattr(space, "mesh", None), "comm", None)
        if comm is not None and getattr(comm, "size", 1) > 1:
            from mpi4py import MPI

            local_finite = bool(comm.allreduce(local_finite, op=MPI.LAND))
        inspected.append(str(name))
        if not local_finite:
            nonfinite.append(str(name))
    return not nonfinite, {
        "inspected": tuple(inspected),
        "nonfinite": tuple(nonfinite),
    }


def _artifact_exists(path) -> bool:
    from pathlib import Path

    return Path(path).exists()


def _infer_converged(result) -> bool:
    existing = getattr(result, "verification", None)
    if isinstance(existing, VerificationReport) and existing.converged:
        return True
    metadata = getattr(result, "metadata", {})
    decisions = []

    def visit(value):
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"converged", "completed_step"} and isinstance(
                    item, (bool, np.bool_)
                ):
                    decisions.append(bool(item))
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(metadata)
    execution = metadata.get("execution") if isinstance(metadata, Mapping) else None
    if isinstance(execution, Mapping):
        kinds = tuple(
            item.get("kind")
            for item in execution.get("events", ())
            if isinstance(item, Mapping)
        )
        if "step_failed" in kinds:
            return False
        if any(kind in kinds for kind in ("step_completed", "transient_completed")):
            return True
    return bool(decisions) and all(decisions)


def _unique_claims(claims: Iterable[VerificationClaim]) -> tuple[VerificationClaim, ...]:
    selected = {}
    for claim in claims:
        if not isinstance(claim, VerificationClaim):
            raise TypeError("Scientific claims must be VerificationClaim objects.")
        selected[claim.name] = claim
    return tuple(selected.values())


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
    "QualityPolicy",
    "QualityPreset",
    "TrustLevel",
    "VerificationClaim",
    "VerificationReport",
    "assess",
    "convergence_study",
    "report",
    "quality_policy",
    "trust_rank",
]
