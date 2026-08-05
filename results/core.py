"""A small, solver-independent result layer.

The result layer is intentionally distinct from file writers.  XDMF, CSV, and
NumPy files are artifacts; a :class:`SimulationResult` is the scientific view
of one completed analysis and is the bridge to campaigns and datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field, field as dc_field
import json
from math import isfinite
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def _name(value: str, *, label: str = "name") -> str:
    selected = str(value).strip()
    if not selected:
        raise ValueError(f"{label} must not be empty.")
    return selected


def _finite_array(value, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values.")
    return array


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Result metadata cannot contain non-finite floats.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    return str(value)


@dataclass(frozen=True)
class ResultQuantity:
    """One scalar or fixed-shape quantity of interest."""

    name: str
    value: object
    unit: str | None = None
    kind: str = "qoi"
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name))
        object.__setattr__(self, "kind", _name(self.kind, label="kind"))
        array = _finite_array(self.value, label=f"ResultQuantity {self.name!r}")
        if array.ndim == 0:
            object.__setattr__(self, "value", float(array))
        else:
            object.__setattr__(self, "value", array.copy())

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(np.asarray(self.value).shape)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": _json_value(self.value),
            "unit": self.unit,
            "kind": self.kind,
            "shape": self.shape,
            "description": self.description,
        }


@dataclass(frozen=True)
class HistoryResult:
    """Time, load, or iteration history with a fixed value shape."""

    name: str
    abscissa: object
    values: object
    unit: str | None = None
    abscissa_name: str = "time"
    abscissa_unit: str | None = "s"
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name))
        object.__setattr__(
            self,
            "abscissa_name",
            _name(self.abscissa_name, label="abscissa_name"),
        )
        x = _finite_array(self.abscissa, label=f"HistoryResult {self.name!r} abscissa")
        y = _finite_array(self.values, label=f"HistoryResult {self.name!r} values")
        if x.ndim != 1:
            raise ValueError("HistoryResult.abscissa must be one-dimensional.")
        if y.ndim == 0 or y.shape[0] != x.size:
            raise ValueError(
                "HistoryResult.values first dimension must match abscissa length."
            )
        if x.size > 1 and np.any(np.diff(x) <= 0.0):
            raise ValueError("HistoryResult.abscissa must be strictly increasing.")
        object.__setattr__(self, "abscissa", x.copy())
        object.__setattr__(self, "values", y.copy())

    @property
    def value_shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape[1:])

    @property
    def latest(self):
        if self.values.shape[0] == 0:
            raise ValueError(f"HistoryResult {self.name!r} is empty.")
        value = self.values[-1]
        return float(value) if np.asarray(value).ndim == 0 else value.copy()

    def as_dict(self, *, include_values: bool = True) -> dict[str, object]:
        record = {
            "name": self.name,
            "unit": self.unit,
            "abscissa_name": self.abscissa_name,
            "abscissa_unit": self.abscissa_unit,
            "sample_count": int(self.abscissa.size),
            "value_shape": self.value_shape,
            "description": self.description,
        }
        if include_values:
            record["abscissa"] = self.abscissa.tolist()
            record["values"] = self.values.tolist()
        return record


@dataclass(frozen=True)
class FieldResult:
    """A named live field or an external field artifact."""

    name: str
    field: object | None = None
    unit: str | None = None
    location: str = "nodes"
    artifact: str | Path | None = None
    description: str = ""
    processing: Mapping[str, object] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name))
        object.__setattr__(self, "processing", dict(self.processing))
        if self.field is None and self.artifact is None:
            raise ValueError("FieldResult requires a live field or artifact.")

    def as_dict(self) -> dict[str, object]:
        value_shape = getattr(self.field, "ufl_shape", None)
        if value_shape is None and hasattr(self.field, "value"):
            value_shape = getattr(self.field.value, "ufl_shape", None)
        return {
            "name": self.name,
            "unit": self.unit,
            "location": self.location,
            "value_shape": None if value_shape is None else tuple(value_shape),
            "live": self.field is not None,
            "artifact": None if self.artifact is None else str(self.artifact),
            "description": self.description,
            "processing": _json_value(self.processing),
        }


@dataclass(frozen=True)
class CheckpointRecord:
    """One restart asset with an explicit portability boundary."""

    name: str
    path: str | Path
    schema: str
    step_name: str
    coordinate_name: str
    coordinate_value: float
    portable: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name))
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "schema", _name(self.schema, label="schema"))
        object.__setattr__(self, "step_name", _name(self.step_name, label="step_name"))
        object.__setattr__(
            self,
            "coordinate_name",
            _name(self.coordinate_name, label="coordinate_name"),
        )
        value = float(self.coordinate_value)
        if not isfinite(value):
            raise ValueError("Checkpoint coordinate_value must be finite.")
        object.__setattr__(self, "coordinate_value", value)

    def as_dict(self, *, artifact_base: str | Path | None = None) -> dict[str, object]:
        selected_path = str(self.path)
        if artifact_base is not None:
            selected_path = _portable_artifact_path(
                self.path,
                Path(artifact_base).resolve(),
            )
        return {
            "name": self.name,
            "path": selected_path,
            "schema": self.schema,
            "step_name": self.step_name,
            "coordinate_name": self.coordinate_name,
            "coordinate_value": self.coordinate_value,
            "portable": bool(self.portable),
            "metadata": _json_value(self.metadata),
        }

    def write_manifest(self, path: str | Path | None = None) -> Path:
        output = (
            self.path.with_suffix(self.path.suffix + ".checkpoint.json")
            if path is None
            else Path(path)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                self.as_dict(artifact_base=output.parent),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return output


@dataclass
class SimulationResult:
    """Scientific results and artifacts from one simulation."""

    name: str
    status: str = "completed"
    quantities: dict[str, ResultQuantity] = field(default_factory=dict)
    fields: dict[str, FieldResult] = field(default_factory=dict)
    histories: dict[str, HistoryResult] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    checkpoints: dict[str, CheckpointRecord] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    verification: object | None = None

    def __post_init__(self) -> None:
        self.name = _name(self.name)
        self.status = _name(self.status, label="status")

    def add_quantity(
        self,
        name: str,
        value,
        *,
        unit: str | None = None,
        kind: str = "qoi",
        description: str = "",
    ) -> ResultQuantity:
        quantity = ResultQuantity(name, value, unit, kind, description)
        self.quantities[quantity.name] = quantity
        return quantity

    def add_quantities(
        self,
        values: Mapping[str, object],
        *,
        units: Mapping[str, str | None] | None = None,
        kind: str = "qoi",
        descriptions: Mapping[str, str] | None = None,
    ) -> dict[str, ResultQuantity]:
        """Record a named quantity mapping without repetitive calls."""

        selected_units = units or {}
        selected_descriptions = descriptions or {}
        return {
            name: self.add_quantity(
                name,
                value,
                unit=selected_units.get(name),
                kind=kind,
                description=selected_descriptions.get(name, ""),
            )
            for name, value in values.items()
        }

    def add_field(
        self,
        name: str,
        value=None,
        *,
        unit: str | None = None,
        location: str = "nodes",
        artifact: str | Path | None = None,
        description: str = "",
        processing: Mapping[str, object] | None = None,
    ) -> FieldResult:
        item = FieldResult(
            name,
            value,
            unit,
            location,
            artifact,
            description,
            {} if processing is None else processing,
        )
        self.fields[item.name] = item
        return item

    def add_history(
        self,
        name: str,
        abscissa,
        values,
        *,
        unit: str | None = None,
        abscissa_name: str = "time",
        abscissa_unit: str | None = "s",
        description: str = "",
    ) -> HistoryResult:
        item = HistoryResult(
            name,
            abscissa,
            values,
            unit,
            abscissa_name,
            abscissa_unit,
            description,
        )
        self.histories[item.name] = item
        return item

    def add_histories(
        self,
        abscissa,
        values: Mapping[str, object],
        *,
        units: Mapping[str, str | None] | None = None,
        abscissa_name: str = "time",
        abscissa_unit: str | None = "s",
        descriptions: Mapping[str, str] | None = None,
    ) -> dict[str, HistoryResult]:
        """Record histories sharing one time, load, or frequency axis."""

        selected_units = units or {}
        selected_descriptions = descriptions or {}
        return {
            name: self.add_history(
                name,
                abscissa,
                history_values,
                unit=selected_units.get(name),
                abscissa_name=abscissa_name,
                abscissa_unit=abscissa_unit,
                description=selected_descriptions.get(name, ""),
            )
            for name, history_values in values.items()
        }

    def add_artifact(self, name: str, path: str | Path) -> Path:
        selected = Path(path)
        self.artifacts[_name(name)] = selected
        return selected

    def add_checkpoint(self, checkpoint: CheckpointRecord) -> CheckpointRecord:
        """Register restart state without confusing it with a result field."""

        self.checkpoints[checkpoint.name] = checkpoint
        self.add_artifact(f"checkpoint_{checkpoint.name}", checkpoint.path)
        return checkpoint

    def add_verification(self, report):
        """Attach scientific trust evidence without changing solver status.

        ``status='completed'`` describes execution.  The verification report
        separately distinguishes computed, converged, verified, and validated
        results so a successful solver call cannot silently imply scientific
        acceptance.
        """

        from ..verification import VerificationReport

        if not isinstance(report, VerificationReport):
            raise TypeError("add_verification requires a VerificationReport.")
        self.verification = report
        return report

    def verify(
        self,
        quality="engineering",
        *,
        claims=(),
        converged: bool | None = None,
        required_quantities=(),
        required_histories=(),
        required_artifacts=(),
    ):
        """Apply a named quality preset and attach its evidence report.

        The default engineering preset runs deterministic payload checks and
        requires solver-convergence evidence. The release preset additionally
        requires an explicit scientific verification claim.
        """

        from ..verification import assess

        return assess(
            self,
            quality,
            claims=claims,
            converged=converged,
            required_quantities=required_quantities,
            required_histories=required_histories,
            required_artifacts=required_artifacts,
        )

    @property
    def trust_level(self) -> str:
        if self.verification is not None:
            return self.verification.trust_level
        return "computed" if self.status == "completed" else "not_computed"

    def add_dof_statistics(
        self,
        field,
        *,
        prefix: str | None = None,
        unit: str | None = None,
    ) -> dict[str, float | int]:
        """Record global coefficient statistics as scalar dataset-ready QoIs.

        These are interpolation-coefficient statistics, not domain integrals.
        Use :mod:`agentfem.results.quantities` for physical integrals/averages.
        """

        selected_name = prefix or getattr(
            getattr(field, "value", field),
            "name",
            "field",
        )
        statistics = dof_statistics(field)
        for statistic, value in statistics.items():
            self.add_quantity(
                f"{selected_name}_{statistic}",
                value,
                unit=None if statistic == "dof_count" else unit,
                kind="dof_statistic",
            )
        return statistics

    def quantity(self, name: str):
        try:
            return self.quantities[name].value
        except KeyError as exc:
            raise KeyError(
                f"Unknown result quantity {name!r}. "
                f"Available: {tuple(self.quantities)}."
            ) from exc

    def field(self, name: str):
        try:
            return self.fields[name].field
        except KeyError as exc:
            raise KeyError(
                f"Unknown result field {name!r}. Available: {tuple(self.fields)}."
            ) from exc

    def outputs(self, names: Iterable[str] | None = None) -> dict[str, object]:
        selected = tuple(self.quantities) if names is None else tuple(names)
        return {name: self.quantity(name) for name in selected}

    def to_sample(
        self,
        *,
        case_id: str,
        inputs: Mapping[str, object],
        outputs: Iterable[str] | None = None,
        provenance: Mapping[str, object] | None = None,
    ):
        """Create a dataset sample without copying live finite-element fields."""

        from agentfem.datasets import Sample

        evidence = {
            "simulation_result": self.summary(),
            **({} if provenance is None else dict(provenance)),
        }
        return Sample(
            case_id=case_id,
            inputs=dict(inputs),
            outputs=self.outputs(outputs),
            provenance=evidence,
            artifacts={
                name: str(path) for name, path in self.artifacts.items()
            },
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "simulation_result",
            "name": self.name,
            "status": self.status,
            "trust_level": self.trust_level,
            "quantities": tuple(self.quantities),
            "fields": tuple(self.fields),
            "histories": tuple(self.histories),
            "artifacts": {key: str(value) for key, value in self.artifacts.items()},
            "checkpoints": tuple(self.checkpoints),
            "metadata": _json_value(self.metadata),
            "verification": (
                None
                if self.verification is None
                else self.verification.as_dict()
            ),
        }

    def format(self) -> str:
        """Return a concise human-facing completion summary.

        Full numeric records remain available through ``manifest()`` and
        ``write_manifest()``; they are intentionally not dumped to a terminal.
        """

        lines = [
            f"Result: {self.name}",
            f"  status: {self.status}",
            f"  trust: {self.trust_level}",
            f"  quantities: {len(self.quantities)}",
            f"  fields: {len(self.fields)}",
            f"  histories: {len(self.histories)}",
            f"  artifacts: {len(self.artifacts)}",
        ]
        if self.verification is not None and self.verification.quality_policy:
            lines.insert(
                3,
                f"  quality: {self.verification.quality_policy} "
                f"({'accepted' if self.verification.acceptable else 'not accepted'})",
            )
        if self.artifacts:
            lines.append(
                "  files: " + ", ".join(sorted(self.artifacts))
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()

    def manifest(
        self,
        *,
        include_histories: bool = False,
        artifact_base: str | Path | None = None,
    ) -> dict[str, object]:
        record = {
            "schema": "agentfem.simulation-result",
            "schema_version": "0.1.0",
            **self.summary(),
            "quantity_records": [
                item.as_dict() for item in self.quantities.values()
            ],
            "field_records": [item.as_dict() for item in self.fields.values()],
            "history_records": [
                item.as_dict(include_values=include_histories)
                for item in self.histories.values()
            ],
            "checkpoint_records": [
                item.as_dict(artifact_base=artifact_base)
                for item in self.checkpoints.values()
            ],
        }
        if artifact_base is not None:
            base = Path(artifact_base).resolve()
            record["artifacts"] = {
                name: _portable_artifact_path(path, base)
                for name, path in self.artifacts.items()
            }
        return record

    def write_manifest(
        self,
        path: str | Path,
        *,
        include_histories: bool = False,
        relative_artifacts: bool = True,
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                self.manifest(
                    include_histories=include_histories,
                    artifact_base=output.parent if relative_artifacts else None,
                ),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return output


def _portable_artifact_path(path: str | Path, base: Path) -> str:
    selected = Path(path)
    resolved = selected.resolve() if selected.is_absolute() else (base / selected).resolve()
    try:
        return str(resolved.relative_to(base))
    except ValueError:
        return str(selected)


def from_solution(
    solution,
    *,
    name: str = "result",
    field_name: str | None = None,
    unit: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SimulationResult:
    """Wrap one solved field in a :class:`SimulationResult`."""

    result = SimulationResult(
        name=name,
        metadata={} if metadata is None else dict(metadata),
    )
    selected_name = field_name or getattr(solution, "name", "solution")
    result.add_field(
        selected_name,
        solution,
        unit=unit,
        processing={
            "method": "primary_finite_element_solution",
            "representation": "finite_element_dofs",
            "postprocessed": False,
        },
    )
    return result


def dof_statistics(field) -> dict[str, float | int]:
    """Return global finite dof statistics for a DOLFINx-like field.

    These are coefficient statistics, not physical domain integrals.  Ghost
    entries are excluded when the function-space index map is available.
    """

    selected = getattr(field, "value", field)
    values = np.asarray(selected.x.array)
    space = selected.function_space
    index_map = getattr(space.dofmap, "index_map", None)
    block_size = int(getattr(space.dofmap, "index_map_bs", 1))
    if index_map is not None:
        values = values[: int(index_map.size_local) * block_size]
    if values.size == 0:
        local_min = np.inf
        local_max = -np.inf
        local_abs = 0.0
    else:
        local_min = float(np.min(values))
        local_max = float(np.max(values))
        local_abs = float(np.max(np.abs(values)))
    comm = getattr(space.mesh, "comm", None)
    if comm is None:
        global_min, global_max, global_abs, global_count = (
            local_min,
            local_max,
            local_abs,
            int(values.size),
        )
    else:
        from mpi4py import MPI

        global_min = comm.allreduce(local_min, op=MPI.MIN)
        global_max = comm.allreduce(local_max, op=MPI.MAX)
        global_abs = comm.allreduce(local_abs, op=MPI.MAX)
        global_count = comm.allreduce(int(values.size), op=MPI.SUM)
    return {
        "minimum": float(global_min),
        "maximum": float(global_max),
        "max_abs": float(global_abs),
        "dof_count": int(global_count),
    }
