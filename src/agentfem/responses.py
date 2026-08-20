"""Traceable response experiments built on AgentFEM campaigns."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import isfinite
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from . import campaigns
from .campaigns import CampaignReport, ParameterSpace, RealParameter
from .datasets import Quantity
from .provenance import content_fingerprint


@dataclass(frozen=True)
class ResponseReport:
    """A finite-difference Jacobian and the cases that support it."""

    name: str
    status: str
    parameter_names: tuple[str, ...]
    output_names: tuple[str, ...]
    steps: Mapping[str, float]
    jacobian: np.ndarray | None
    baseline: np.ndarray | None
    singular_values: np.ndarray | None
    rank: int | None
    condition_number: float | None
    conditioning_basis: str = "not_available"
    nonlinearity: Mapping[str, float] = field(default_factory=dict)
    nonlinearity_by_output: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    derivative_units: Mapping[str, str | None] = field(default_factory=dict)
    missing_cases: tuple[str, ...] = ()
    case_ids: Mapping[str, str] = field(default_factory=dict)
    design_fingerprint: str | None = None

    @property
    def complete(self) -> bool:
        return self.status == "completed" and self.jacobian is not None

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.response-report",
            "schema_version": "0.1.0",
            "name": self.name,
            "status": self.status,
            "complete": self.complete,
            "parameter_names": self.parameter_names,
            "output_names": self.output_names,
            "steps": dict(self.steps),
            "jacobian": None if self.jacobian is None else self.jacobian.tolist(),
            "baseline": None if self.baseline is None else self.baseline.tolist(),
            "singular_values": (
                None if self.singular_values is None else self.singular_values.tolist()
            ),
            "rank": self.rank,
            "condition_number": self.condition_number,
            "conditioning_basis": self.conditioning_basis,
            "nonlinearity": dict(self.nonlinearity),
            "nonlinearity_by_output": {
                parameter: dict(values)
                for parameter, values in self.nonlinearity_by_output.items()
            },
            "derivative_units": dict(self.derivative_units),
            "missing_cases": self.missing_cases,
            "case_ids": dict(self.case_ids),
            "design_fingerprint": self.design_fingerprint,
        }

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                self.summary(),
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


@dataclass(frozen=True)
class FiniteDifferenceResponse:
    """A method-neutral response contract with a finite-difference provider."""

    parameter_space: ParameterSpace
    baseline: Mapping[str, object]
    outputs: tuple[Quantity, ...]
    perturbation: float | Mapping[str, float] = 0.01
    scheme: str = "central"
    step_mode: str = "relative"
    name: str = "response_operator"

    def __post_init__(self) -> None:
        baseline = self.parameter_space.validate(self.baseline)
        outputs = tuple(self.outputs)
        if not outputs:
            raise ValueError("FiniteDifferenceResponse requires output quantities.")
        if len({item.name for item in outputs}) != len(outputs):
            raise ValueError("Response output names must be unique.")
        if any(not isinstance(item, RealParameter) for item in self.parameter_space.parameters):
            raise TypeError(
                "Finite-difference response parameters must currently be RealParameter objects."
            )
        if any(item.scale != "linear" for item in self.parameter_space.parameters):
            raise ValueError(
                "The first finite-difference provider supports linear parameters only; "
                "logarithmic coordinates require a declared transformed derivative."
            )
        scheme = str(self.scheme).strip().lower().replace("-", "_")
        if scheme not in {"central", "forward", "backward"}:
            raise ValueError("Response scheme must be central, forward, or backward.")
        step_mode = str(self.step_mode).strip().lower().replace("-", "_")
        if step_mode not in {"relative", "absolute"}:
            raise ValueError("Response step_mode must be relative or absolute.")
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "step_mode", step_mode)
        object.__setattr__(self, "name", str(self.name).strip() or "response_operator")
        self._steps()

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.parameter_space.names

    @property
    def output_names(self) -> tuple[str, ...]:
        names = []
        for quantity in self.outputs:
            if not quantity.shape:
                names.append(quantity.name)
                continue
            names.extend(
                f"{quantity.name}[{','.join(str(item) for item in index)}]"
                for index in np.ndindex(quantity.shape)
            )
        return tuple(names)

    @property
    def fingerprint(self) -> str:
        return content_fingerprint(self.summary())

    def sampling(self):
        """Return the immutable baseline/perturbation design as a campaign plan."""

        steps = self._steps()
        samples: list[dict[str, object]] = [dict(self.baseline)]
        labels = ["baseline"]
        for parameter in self.parameter_space.parameters:
            name = parameter.name
            if self.scheme in {"central", "forward"}:
                sample = dict(self.baseline)
                sample[name] = float(sample[name]) + steps[name]
                samples.append(sample)
                labels.append(f"{name}:plus")
            if self.scheme in {"central", "backward"}:
                sample = dict(self.baseline)
                sample[name] = float(sample[name]) - steps[name]
                samples.append(sample)
                labels.append(f"{name}:minus")
        return campaigns.explicit(
            self.parameter_space,
            samples,
            metadata={
                "purpose": "finite_difference_response",
                "scheme": self.scheme,
                "step_mode": self.step_mode,
                "steps": steps,
                "case_labels": labels,
                "design_fingerprint": self.fingerprint,
            },
        )

    def campaign(
        self,
        *,
        evaluate: Callable,
        build: Callable[[Mapping[str, object]], object] | None = None,
        execution: campaigns.ExecutionPolicy | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> campaigns.Campaign:
        """Create the ordinary Campaign that executes this response design."""

        return campaigns.create(
            name=self.name,
            parameter_space=self.parameter_space,
            outputs=self.outputs,
            evaluate=evaluate,
            build=build,
            execution=execution,
            metadata={
                **dict(metadata or {}),
                "response_design": self.summary(),
                "response_design_fingerprint": self.fingerprint,
            },
        )

    def run(
        self,
        *,
        evaluate: Callable,
        build: Callable[[Mapping[str, object]], object] | None = None,
        output_directory: str | Path | None = None,
        execution: campaigns.ExecutionPolicy | None = None,
        comm=None,
    ) -> tuple[CampaignReport, ResponseReport]:
        """Execute through Campaign and analyze one traceable response matrix."""

        campaign = self.campaign(
            evaluate=evaluate,
            build=build,
            execution=execution,
        )
        campaign_report = campaign.run(
            self.sampling(),
            output_directory=output_directory,
            comm=comm,
        )
        response = self.analyze(campaign_report)
        rank = getattr(comm, "rank", 0)
        if output_directory is not None and rank == 0:
            response.write(Path(output_directory) / "response.json")
        return campaign_report, response

    def analyze(self, report: CampaignReport) -> ResponseReport:
        """Recover the Jacobian without hiding missing or failed cases."""

        expected_sampling = self.sampling()
        expected_labels = tuple(expected_sampling.metadata["case_labels"])
        expected_samples = tuple(expected_sampling.samples)
        completed = {
            _parameter_key(record.case.parameters): record
            for record in report.records
            if record.successful and record.outcome is not None
        }
        records = []
        missing = []
        case_ids = {}
        for label, sample in zip(expected_labels, expected_samples, strict=True):
            record = completed.get(_parameter_key(sample))
            if record is None:
                missing.append(label)
                records.append(None)
                continue
            records.append(record)
            case_ids[label] = record.case.case_id
        if missing:
            return ResponseReport(
                name=self.name,
                status="incomplete",
                parameter_names=self.parameter_names,
                output_names=self.output_names,
                steps=self._steps(),
                jacobian=None,
                baseline=None,
                singular_values=None,
                rank=None,
                condition_number=None,
                conditioning_basis="not_available",
                derivative_units=self._derivative_units(),
                missing_cases=tuple(missing),
                case_ids=case_ids,
                design_fingerprint=self.fingerprint,
            )

        vectors = [_output_vector(record.outcome.outputs, self.outputs) for record in records]
        baseline = vectors[0]
        columns = []
        nonlinearity = {}
        nonlinearity_by_output = {}
        cursor = 1
        steps = self._steps()
        for parameter in self.parameter_space.parameters:
            step = steps[parameter.name]
            if self.scheme == "central":
                plus, minus = vectors[cursor], vectors[cursor + 1]
                cursor += 2
                columns.append((plus - minus) / (2.0 * step))
                by_output = _central_nonlinearity_by_output(
                    plus,
                    baseline,
                    minus,
                    self.outputs,
                )
                nonlinearity_by_output[parameter.name] = by_output
                nonlinearity[parameter.name] = max(by_output.values(), default=0.0)
            elif self.scheme == "forward":
                plus = vectors[cursor]
                cursor += 1
                columns.append((plus - baseline) / step)
            else:
                minus = vectors[cursor]
                cursor += 1
                columns.append((baseline - minus) / step)
        jacobian = np.column_stack(columns)
        rank = int(np.linalg.matrix_rank(jacobian))
        homogeneous = self._homogeneous_units()
        singular_values = (
            np.linalg.svd(jacobian, compute_uv=False) if homogeneous else None
        )
        condition = float(np.linalg.cond(jacobian)) if homogeneous else None
        if condition is not None and not isfinite(condition):
            condition = None
        return ResponseReport(
            name=self.name,
            status="completed",
            parameter_names=self.parameter_names,
            output_names=self.output_names,
            steps=steps,
            jacobian=jacobian,
            baseline=baseline,
            singular_values=singular_values,
            rank=rank,
            condition_number=condition,
            conditioning_basis=(
                "unscaled_homogeneous_units"
                if homogeneous
                else "mixed_units_require_explicit_scaling"
            ),
            nonlinearity=nonlinearity,
            nonlinearity_by_output=nonlinearity_by_output,
            derivative_units=self._derivative_units(),
            case_ids=case_ids,
            design_fingerprint=self.fingerprint,
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema": "agentfem.response-design",
            "schema_version": "0.1.0",
            "name": self.name,
            "provider": "finite_difference",
            "scheme": self.scheme,
            "step_mode": self.step_mode,
            "relative_step_reference": "max(abs(baseline), parameter_span)",
            "parameter_space": self.parameter_space.summary(),
            "baseline": dict(self.baseline),
            "steps": self._steps(),
            "outputs": [item.summary() for item in self.outputs],
        }

    def _steps(self) -> dict[str, float]:
        if isinstance(self.perturbation, Mapping):
            missing = [name for name in self.parameter_names if name not in self.perturbation]
            extra = [name for name in self.perturbation if name not in self.parameter_names]
            if missing or extra:
                raise ValueError(
                    f"Response perturbations differ from parameters; missing={missing}, extra={extra}."
                )
            raw = {name: float(self.perturbation[name]) for name in self.parameter_names}
        else:
            raw = {name: float(self.perturbation) for name in self.parameter_names}
        steps = {}
        for parameter in self.parameter_space.parameters:
            value = float(self.baseline[parameter.name])
            selected = raw[parameter.name]
            if not isfinite(selected) or selected <= 0.0:
                raise ValueError("Response perturbations must be positive and finite.")
            scale = max(abs(value), parameter.upper - parameter.lower)
            step = selected * scale if self.step_mode == "relative" else selected
            lower = value - step if self.scheme in {"central", "backward"} else value
            upper = value + step if self.scheme in {"central", "forward"} else value
            if lower < parameter.lower or upper > parameter.upper:
                raise ValueError(
                    f"Response perturbation for {parameter.name!r} leaves its bounds; "
                    "reduce the step or select a one-sided scheme."
                )
            steps[parameter.name] = step
        return steps

    def _derivative_units(self) -> dict[str, str | None]:
        output_units = []
        cursor = 0
        for quantity in self.outputs:
            for _ in range(quantity.size):
                output_units.append((self.output_names[cursor], quantity.unit))
                cursor += 1
        parameter_units = {
            parameter.name: parameter.unit
            for parameter in self.parameter_space.parameters
        }
        return {
            f"{output_name} wrt {parameter_name}": _quotient_unit(
                output_unit,
                parameter_units[parameter_name],
            )
            for output_name, output_unit in output_units
            for parameter_name in self.parameter_names
        }

    def _homogeneous_units(self) -> bool:
        return (
            len({quantity.unit for quantity in self.outputs}) == 1
            and len({parameter.unit for parameter in self.parameter_space.parameters}) == 1
        )


def finite_difference(**kwargs) -> FiniteDifferenceResponse:
    """Create a campaign-backed finite-difference response experiment."""

    return FiniteDifferenceResponse(**kwargs)


def _parameter_key(values: Mapping[str, object]) -> str:
    return json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _output_vector(
    values: Mapping[str, object],
    quantities: tuple[Quantity, ...],
) -> np.ndarray:
    return np.concatenate(
        [quantity.validate(values[quantity.name]).reshape(-1) for quantity in quantities]
    )


def _quotient_unit(output: str | None, parameter: str | None) -> str | None:
    if output is None and parameter is None:
        return None
    if parameter is None:
        return output
    if output is None:
        return f"1/{parameter}"
    return f"{output}/{parameter}"


def _central_nonlinearity_by_output(
    plus: np.ndarray,
    baseline: np.ndarray,
    minus: np.ndarray,
    quantities: tuple[Quantity, ...],
) -> dict[str, float]:
    """Measure curvature per quantity without combining unlike physical units."""

    metrics = {}
    cursor = 0
    for quantity in quantities:
        selected = slice(cursor, cursor + quantity.size)
        curvature = float(
            np.linalg.norm(plus[selected] - 2.0 * baseline[selected] + minus[selected])
        )
        response_span = max(
            float(np.linalg.norm(plus[selected] - minus[selected])),
            np.finfo(float).tiny,
        )
        metrics[quantity.name] = curvature / response_span
        cursor += quantity.size
    return metrics


__all__ = [
    "FiniteDifferenceResponse",
    "ResponseReport",
    "finite_difference",
]
