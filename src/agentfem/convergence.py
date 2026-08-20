"""Multi-axis scientific convergence certificates for campaign evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import isfinite, log
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from .campaigns import CampaignReport, CaseRunRecord
from .ir.schema import to_json_safe
from .provenance import content_fingerprint


Comparison = Literal["relative", "absolute", "exact"]
Source = Literal["output", "provenance"]
Characteristic = Literal["value", "inverse"]


@dataclass(frozen=True)
class ConvergenceAxis:
    """One refinement coordinate with all other coordinates fixed explicitly."""

    parameter: str
    fixed: Mapping[str, object] = field(default_factory=dict)
    discretization: str = "mesh"
    characteristic: Characteristic = "value"

    def __post_init__(self) -> None:
        parameter = str(self.parameter).strip()
        if not parameter:
            raise ValueError("ConvergenceAxis.parameter must not be empty.")
        selected = str(self.characteristic).strip().lower().replace("-", "_")
        if selected not in {"value", "inverse"}:
            raise ValueError("Axis characteristic must be value or inverse.")
        if parameter in self.fixed:
            raise ValueError("A convergence axis cannot also fix its own parameter.")
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "fixed", dict(self.fixed))
        object.__setattr__(self, "characteristic", selected)

    def size(self, parameters: Mapping[str, object]) -> float:
        value = parameters[self.parameter]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"Convergence axis {self.parameter!r} must be numeric."
            )
        selected = float(value)
        if not isfinite(selected) or selected <= 0.0:
            raise ValueError(
                f"Convergence axis {self.parameter!r} must be positive and finite."
            )
        return selected if self.characteristic == "value" else 1.0 / selected

    def summary(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "fixed": dict(self.fixed),
            "discretization": self.discretization,
            "characteristic": self.characteristic,
        }


@dataclass(frozen=True)
class ObservablePolicy:
    """How one scalar, vector, event, or topology record is compared."""

    name: str
    comparison: Comparison = "relative"
    tolerance: float | None = None
    source: Source = "output"
    path: str | None = None
    minimum_observed_order: float | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("ObservablePolicy.name must not be empty.")
        comparison = str(self.comparison).strip().lower().replace("-", "_")
        if comparison not in {"relative", "absolute", "exact"}:
            raise ValueError(
                "Observable comparison must be relative, absolute, or exact."
            )
        source = str(self.source).strip().lower().replace("-", "_")
        if source not in {"output", "provenance"}:
            raise ValueError("Observable source must be output or provenance.")
        tolerance = None if self.tolerance is None else float(self.tolerance)
        if comparison == "exact":
            if tolerance is not None or self.minimum_observed_order is not None:
                raise ValueError("Exact comparison does not accept numeric tolerances.")
        elif tolerance is None or not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Numeric convergence requires a nonnegative tolerance.")
        order = (
            None
            if self.minimum_observed_order is None
            else float(self.minimum_observed_order)
        )
        if order is not None and not isfinite(order):
            raise ValueError("minimum_observed_order must be finite.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "comparison", comparison)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "minimum_observed_order", order)
        object.__setattr__(self, "path", str(self.path or name))

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "comparison": self.comparison,
            "tolerance": self.tolerance,
            "source": self.source,
            "path": self.path,
            "minimum_observed_order": self.minimum_observed_order,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class ConvergenceCheck:
    """One observable checked along one explicitly selected refinement axis."""

    axis: str
    observable: str
    status: str
    comparison: str
    metric: float | None
    tolerance: float | None
    observed_order: float | None
    characteristic_sizes: tuple[float, ...]
    case_ids: tuple[str, ...]
    values: tuple[object, ...]
    failed_case_ids: tuple[str, ...] = ()
    missing_case_ids: tuple[str, ...] = ()
    message: str = ""

    def summary(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "observable": self.observable,
            "status": self.status,
            "comparison": self.comparison,
            "metric": self.metric,
            "tolerance": self.tolerance,
            "observed_order": self.observed_order,
            "characteristic_sizes": self.characteristic_sizes,
            "case_ids": self.case_ids,
            "values": self.values,
            "failed_case_ids": self.failed_case_ids,
            "missing_case_ids": self.missing_case_ids,
            "message": self.message,
        }


@dataclass(frozen=True)
class ConvergenceCertificate:
    """Auditable multi-axis convergence decision for one CampaignReport."""

    campaign: str
    status: str
    checks: tuple[ConvergenceCheck, ...]
    axes: tuple[ConvergenceAxis, ...]
    observables: tuple[ObservablePolicy, ...]
    campaign_fingerprint: str

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def summary(self) -> dict[str, object]:
        record = {
            "schema": "agentfem.convergence-certificate",
            "schema_version": "0.1.0",
            "campaign": self.campaign,
            "status": self.status,
            "passed": self.passed,
            "campaign_fingerprint": self.campaign_fingerprint,
            "axes": [item.summary() for item in self.axes],
            "observables": [item.summary() for item in self.observables],
            "checks": [item.summary() for item in self.checks],
        }
        return {**record, "certificate_id": content_fingerprint(record)}

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                to_json_safe(self.summary()),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
        return output


def axis(
    parameter: str,
    *,
    fixed: Mapping[str, object] | None = None,
    discretization: str = "mesh",
    characteristic: Characteristic = "value",
) -> ConvergenceAxis:
    return ConvergenceAxis(
        parameter=parameter,
        fixed={} if fixed is None else fixed,
        discretization=discretization,
        characteristic=characteristic,
    )


def observable(
    name: str,
    *,
    comparison: Comparison = "relative",
    tolerance: float | None = None,
    source: Source = "output",
    path: str | None = None,
    minimum_observed_order: float | None = None,
    unit: str | None = None,
) -> ObservablePolicy:
    return ObservablePolicy(
        name=name,
        comparison=comparison,
        tolerance=tolerance,
        source=source,
        path=path,
        minimum_observed_order=minimum_observed_order,
        unit=unit,
    )


def audit(
    report: CampaignReport,
    *,
    axes: tuple[ConvergenceAxis, ...],
    observables: tuple[ObservablePolicy, ...],
    output: str | Path | None = None,
) -> ConvergenceCertificate:
    """Build a conservative convergence certificate from campaign evidence."""

    selected_axes = tuple(axes)
    selected_observables = tuple(observables)
    if not selected_axes or not selected_observables:
        raise ValueError("Convergence audit requires axes and observables.")
    if len({item.parameter for item in selected_axes}) != len(selected_axes):
        raise ValueError("Convergence axis parameters must be unique.")
    if len({item.name for item in selected_observables}) != len(
        selected_observables
    ):
        raise ValueError("Convergence observable names must be unique.")
    parameter_names = set(report.plan.parameter_space.names)
    for selected_axis in selected_axes:
        unknown = ({selected_axis.parameter} | set(selected_axis.fixed)) - parameter_names
        if unknown:
            raise ValueError(f"Convergence axis references unknown parameters {unknown}.")

    records = {record.case.case_id: record for record in report.records}
    checks = []
    for selected_axis in selected_axes:
        planned = tuple(
            case
            for case in report.plan.cases
            if _matches_fixed(case.parameters, selected_axis.fixed)
        )
        selected_records = tuple(records.get(case.case_id) for case in planned)
        uncontrolled = tuple(
            name
            for name in report.plan.parameter_space.names
            if name != selected_axis.parameter
            and name not in selected_axis.fixed
            and len({_stable_value(case.parameters[name]) for case in planned}) > 1
        )
        for selected_observable in selected_observables:
            if uncontrolled:
                successful = tuple(
                    record
                    for record in selected_records
                    if record is not None and record.successful
                )
                checks.append(
                    _inconclusive(
                        selected_axis,
                        selected_observable,
                        successful,
                        message=(
                            "The refinement slice changes uncontrolled parameters "
                            f"{uncontrolled}; fix them explicitly."
                        ),
                    )
                )
                continue
            checks.append(
                _check_axis(
                    selected_axis,
                    selected_observable,
                    planned,
                    selected_records,
                )
            )
    statuses = {item.status for item in checks}
    status = (
        "failed"
        if "failed" in statuses
        else "inconclusive"
        if "inconclusive" in statuses
        else "passed"
    )
    certificate = ConvergenceCertificate(
        campaign=report.name,
        status=status,
        checks=tuple(checks),
        axes=selected_axes,
        observables=selected_observables,
        campaign_fingerprint=content_fingerprint(report.plan.summary()),
    )
    if output is not None:
        certificate.write(output)
    return certificate


def _check_axis(axis, policy, planned, selected_records) -> ConvergenceCheck:
    missing = tuple(
        case.case_id
        for case, record in zip(planned, selected_records, strict=True)
        if record is None
    )
    failed = tuple(
        record.case.case_id
        for record in selected_records
        if record is not None and not record.successful
    )
    successful = tuple(
        record
        for record in selected_records
        if record is not None and record.successful
    )
    if missing or failed:
        return _inconclusive(
            axis,
            policy,
            successful,
            failed=failed,
            missing=missing,
            message="The selected refinement sequence contains missing or failed cases.",
        )
    if len(successful) < 2:
        return _inconclusive(
            axis,
            policy,
            successful,
            message="At least two successful refinement cases are required.",
        )
    try:
        ordered = tuple(
            sorted(
                successful,
                key=lambda record: axis.size(record.case.parameters),
                reverse=True,
            )
        )
        sizes = tuple(axis.size(record.case.parameters) for record in ordered)
        if len(set(sizes)) != len(sizes):
            return _inconclusive(
                axis,
                policy,
                ordered,
                message=(
                    "The selected slice has duplicate characteristic sizes; "
                    "fix every non-refined parameter explicitly."
                ),
            )
        values = tuple(_extract(record, policy) for record in ordered)
        if policy.comparison == "exact":
            passed = all(_equal(values[0], value) for value in values[1:])
            return ConvergenceCheck(
                axis=axis.parameter,
                observable=policy.name,
                status="passed" if passed else "failed",
                comparison=policy.comparison,
                metric=0.0 if passed else 1.0,
                tolerance=None,
                observed_order=None,
                characteristic_sizes=sizes,
                case_ids=tuple(record.case.case_id for record in ordered),
                values=tuple(to_json_safe(value) for value in values),
                message=(
                    "The categorical/event sequence is invariant under refinement."
                    if passed
                    else "The categorical/event sequence changes under refinement."
                ),
            )
        arrays = tuple(_numeric(value) for value in values)
        difference = float(np.linalg.norm(arrays[-1] - arrays[-2]))
        metric = (
            difference
            if policy.comparison == "absolute"
            else difference
            / max(float(np.linalg.norm(arrays[-1])), np.finfo(float).tiny)
        )
        order = _observed_order(sizes, arrays)
        change_passes = metric <= float(policy.tolerance)
        order_available = policy.minimum_observed_order is None or order is not None
        order_passes = (
            policy.minimum_observed_order is None
            or (order is not None and order >= policy.minimum_observed_order)
        )
        status = (
            "inconclusive"
            if change_passes and not order_available
            else "passed"
            if change_passes and order_passes
            else "failed"
        )
        return ConvergenceCheck(
            axis=axis.parameter,
            observable=policy.name,
            status=status,
            comparison=policy.comparison,
            metric=metric,
            tolerance=policy.tolerance,
            observed_order=order,
            characteristic_sizes=sizes,
            case_ids=tuple(record.case.case_id for record in ordered),
            values=tuple(to_json_safe(value) for value in values),
            message=(
                "The declared refinement criterion is satisfied."
                if status == "passed"
                else "The available sequence does not establish the declared criterion."
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _inconclusive(
            axis,
            policy,
            successful,
            message=f"Observable extraction/comparison is incomplete: {exc}",
        )


def _inconclusive(
    axis,
    policy,
    records,
    *,
    failed=(),
    missing=(),
    message,
) -> ConvergenceCheck:
    return ConvergenceCheck(
        axis=axis.parameter,
        observable=policy.name,
        status="inconclusive",
        comparison=policy.comparison,
        metric=None,
        tolerance=policy.tolerance,
        observed_order=None,
        characteristic_sizes=tuple(
            axis.size(record.case.parameters) for record in records
        ),
        case_ids=tuple(record.case.case_id for record in records),
        values=(),
        failed_case_ids=tuple(failed),
        missing_case_ids=tuple(missing),
        message=message,
    )


def _matches_fixed(parameters, fixed) -> bool:
    return all(_equal(parameters.get(name), value) for name, value in fixed.items())


def _stable_value(value) -> str:
    return json.dumps(
        to_json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _extract(record: CaseRunRecord, policy: ObservablePolicy):
    if record.outcome is None:
        raise ValueError("successful case has no outcome")
    root = (
        record.outcome.outputs
        if policy.source == "output"
        else record.outcome.provenance
    )
    selected = root
    for part in str(policy.path).split("."):
        if not isinstance(selected, Mapping) or part not in selected:
            raise KeyError(f"{policy.source} path {policy.path!r} is unavailable")
        selected = selected[part]
    return selected


def _numeric(value) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(selected)):
        raise ValueError("numeric observable contains non-finite values")
    return selected


def _equal(left, right) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, rtol=1.0e-12, atol=1.0e-15))
    try:
        return to_json_safe(left) == to_json_safe(right)
    except (TypeError, ValueError):
        return False


def _observed_order(sizes, values) -> float | None:
    if len(sizes) < 3:
        return None
    h1, h2, h3 = sizes[-3:]
    ratio_1 = h1 / h2
    ratio_2 = h2 / h3
    if not np.isclose(ratio_1, ratio_2, rtol=1.0e-6, atol=1.0e-12):
        return None
    difference_1 = float(np.linalg.norm(values[-3] - values[-2]))
    difference_2 = float(np.linalg.norm(values[-2] - values[-1]))
    if difference_1 <= 0.0 or difference_2 <= 0.0:
        return None
    return log(difference_1 / difference_2) / log(ratio_1)


__all__ = [
    "ConvergenceAxis",
    "ConvergenceCertificate",
    "ConvergenceCheck",
    "ObservablePolicy",
    "audit",
    "axis",
    "observable",
]
