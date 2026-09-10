"""Result-owned response quantities sampled during harmonic sweeps."""

from __future__ import annotations

from dataclasses import dataclass

from dolfinx import fem, geometry as geometry_api
import numpy as np
import ufl

from .. import _axisymmetric
from .. import fields as field_api
from ..provenance import collective_call


_REDUCTIONS = {"identical", "sum", "mean", "weighted_mean"}


def _response_name(value: str) -> str:
    selected = str(value).strip()
    if not selected:
        raise ValueError("Harmonic response name must not be empty.")
    return selected


def _finite_scalar(value, *, name: str) -> complex:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"Harmonic response {name!r} must return one scalar.")
    selected = complex(array.item())
    if not np.isfinite(selected.real) or not np.isfinite(selected.imag):
        raise ValueError(
            f"Harmonic response {name!r} returned a non-finite value."
        )
    return selected


@dataclass(frozen=True)
class HarmonicResponse:
    """One scalar response computed locally and reduced by AgentFEM.

    A plain evaluator remains deliberately simple in serial. Under MPI the
    caller must state how rank-local values are combined. The evaluator must
    not enter an MPI collective itself; AgentFEM first exchanges local errors
    and only then performs the declared reduction.
    """

    name: str
    evaluate: object
    unit: str | None = None
    description: str = ""
    reduction: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _response_name(self.name))
        if not callable(self.evaluate):
            raise TypeError("HarmonicResponse.evaluate must be callable.")
        reduction = None if self.reduction is None else str(self.reduction).lower()
        if reduction is not None and reduction not in _REDUCTIONS:
            raise ValueError(
                "HarmonicResponse.reduction must be identical, sum, mean, "
                "weighted_mean, or None."
            )
        object.__setattr__(self, "reduction", reduction)

    def sample(self, step) -> complex:
        comm = step.solution_real.function_space.mesh.comm
        if comm.size > 1 and self.reduction is None:
            raise NotImplementedError(
                "AFM-HARMONIC-RESPONSE-001: a callable harmonic response is "
                "serial-only unless reduction= declares how rank-local values "
                "are combined. Use harmonic_average_response or "
                "harmonic_probe_response for finite-element observations."
            )

        if self.reduction == "weighted_mean":
            payload = collective_call(
                lambda: self._weighted_payload(step),
                comm=comm,
                label=f"Harmonic response {self.name!r} local evaluation",
            )
            gathered = tuple(comm.allgather(payload))
            denominator = float(sum(item[2] for item in gathered))
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError(
                    f"Harmonic response {self.name!r} requires positive total weight."
                )
            return _finite_scalar(
                complex(
                    sum(item[0] for item in gathered),
                    sum(item[1] for item in gathered),
                )
                / denominator,
                name=self.name,
            )

        selected = collective_call(
            lambda: _finite_scalar(self.evaluate(step), name=self.name),
            comm=comm,
            label=f"Harmonic response {self.name!r} local evaluation",
        )
        gathered = tuple(comm.allgather((selected.real, selected.imag)))
        if self.reduction in {None, "identical"}:
            if any(
                not np.allclose(item, gathered[0], rtol=1.0e-12, atol=1.0e-14)
                for item in gathered[1:]
            ):
                raise RuntimeError(
                    f"Harmonic response {self.name!r} differs across MPI ranks."
                )
            real, imaginary = gathered[0]
        elif self.reduction == "sum":
            real = sum(item[0] for item in gathered)
            imaginary = sum(item[1] for item in gathered)
        else:  # mean
            real = sum(item[0] for item in gathered) / len(gathered)
            imaginary = sum(item[1] for item in gathered) / len(gathered)
        return complex(real, imaginary)

    def _weighted_payload(self, step) -> tuple[float, float, float]:
        value = self.evaluate(step)
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(
                f"Harmonic response {self.name!r} with weighted_mean must "
                "return (weighted_value, local_weight)."
            )
        weighted = _finite_scalar(value[0], name=self.name)
        weight = float(value[1])
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"Harmonic response {self.name!r} returned an invalid local weight."
            )
        return weighted.real, weighted.imag, weight

    def to_ir(self) -> dict[str, object]:
        return {
            "kind": "rank_local_scalar_harmonic_response",
            "name": self.name,
            "unit": self.unit,
            "description": self.description,
            "reduction": self.reduction,
            "evaluate": self.evaluate,
        }

    def summary(self) -> dict[str, object]:
        return {
            "kind": "rank_local_scalar_harmonic_response",
            "name": self.name,
            "unit": self.unit,
            "description": self.description,
            "reduction": self.reduction,
            "mpi_contract": (
                "serial_only"
                if self.reduction is None
                else "local_then_framework_reduce"
            ),
            "evaluator": _evaluator_summary(self.evaluate),
        }


@dataclass(frozen=True)
class HarmonicAverageResponse:
    """Measure-weighted complex average with rank-local assembly."""

    name: str
    evaluate: object
    on: object
    study: object | None = None
    unit: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _response_name(self.name))
        if not callable(self.evaluate):
            raise TypeError("HarmonicAverageResponse.evaluate must be callable.")
        if getattr(self.on, "domain", None) is None or getattr(
            self.on, "measure", None
        ) is None:
            raise TypeError("harmonic_average_response on= requires a mesh region.")

    def sample(self, step) -> complex:
        comm = step.solution_real.function_space.mesh.comm

        def assemble_local() -> tuple[float, float, float]:
            if self.on.domain is not step.solution_real.function_space.mesh:
                raise ValueError(
                    "Harmonic average region must belong to the solved mesh."
                )
            expressions = self.evaluate(step)
            if not isinstance(expressions, (tuple, list)) or len(expressions) != 2:
                raise ValueError(
                    f"Harmonic average {self.name!r} evaluator must return "
                    "(real_expression, imaginary_expression)."
                )
            real_expression, imaginary_expression = expressions
            if tuple(getattr(real_expression, "ufl_shape", ())) or tuple(
                getattr(imaginary_expression, "ufl_shape", ())
            ):
                raise ValueError(
                    f"Harmonic average {self.name!r} requires scalar expressions."
                )
            weight = _axisymmetric.integration_weight(self.on.domain, self.study)
            measure = self.on.measure
            values = (
                fem.assemble_scalar(fem.form(weight * real_expression * measure)),
                fem.assemble_scalar(fem.form(weight * imaginary_expression * measure)),
                fem.assemble_scalar(
                    fem.form(weight * ufl.as_ufl(1.0) * measure)
                ),
            )
            selected = tuple(float(value) for value in values)
            if not np.all(np.isfinite(selected)):
                raise ValueError(
                    f"Harmonic average {self.name!r} produced non-finite local integrals."
                )
            return selected

        local = collective_call(
            assemble_local,
            comm=comm,
            label=f"Harmonic average {self.name!r} local assembly",
        )
        gathered = tuple(comm.allgather(local))
        denominator = float(sum(item[2] for item in gathered))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(
                f"Harmonic average {self.name!r} requires positive total measure."
            )
        return _finite_scalar(
            complex(
                sum(item[0] for item in gathered),
                sum(item[1] for item in gathered),
            )
            / denominator,
            name=self.name,
        )

    def to_ir(self) -> dict[str, object]:
        return {
            "kind": "harmonic_region_average",
            "name": self.name,
            "unit": self.unit,
            "description": self.description,
            "evaluate": self.evaluate,
            "region": self.on,
            "study": self.study,
            "reduction": "sum_local_integrals_then_divide",
        }

    def summary(self) -> dict[str, object]:
        return {
            "kind": "harmonic_region_average",
            "name": self.name,
            "unit": self.unit,
            "description": self.description,
            "region": getattr(self.on, "name", None),
            "reduction": "sum_local_integrals_then_divide",
            "mpi_contract": "local_assembly_then_framework_reduce",
            "evaluator": _evaluator_summary(self.evaluate),
        }


@dataclass(frozen=True)
class HarmonicProbeResponse:
    """Complex finite-element point probe with framework-owned MPI selection."""

    name: str
    evaluate: object
    at: tuple[float, ...]
    component: int | None = None
    padding: float = 1.0e-10
    unit: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _response_name(self.name))
        if not callable(self.evaluate):
            raise TypeError("HarmonicProbeResponse.evaluate must be callable.")
        point = tuple(float(value) for value in np.asarray(self.at).reshape(-1))
        if not point or not np.all(np.isfinite(point)):
            raise ValueError("Harmonic probe coordinates must be finite.")
        padding = float(self.padding)
        if not np.isfinite(padding) or padding < 0.0:
            raise ValueError("Harmonic probe padding must be finite and nonnegative.")
        component = None if self.component is None else int(self.component)
        if component is not None and component < 0:
            raise ValueError("Harmonic probe component must be nonnegative.")
        object.__setattr__(self, "at", point)
        object.__setattr__(self, "padding", padding)
        object.__setattr__(self, "component", component)

    def sample(self, step) -> complex:
        comm = step.solution_real.function_space.mesh.comm
        prepare = getattr(self.evaluate, "prepare_harmonic_response", None)
        if callable(prepare):
            # Only a structured evaluator may own a collective preparation
            # such as a reusable global L2 projection. Plain callbacks never
            # enter this path.
            prepare(step)

        def evaluate_local():
            fields = self.evaluate(step)
            if not isinstance(fields, (tuple, list)) or len(fields) != 2:
                raise ValueError(
                    f"Harmonic probe {self.name!r} evaluator must return "
                    "(real_field, imaginary_field)."
                )
            real = _local_probe_candidate(fields[0], self.at, self.padding)
            imaginary = _local_probe_candidate(fields[1], self.at, self.padding)
            if (real is None) != (imaginary is None):
                raise RuntimeError(
                    f"Harmonic probe {self.name!r} real and imaginary ownership differ."
                )
            if real is None:
                return None
            return (
                _select_probe_component(real, self.component),
                _select_probe_component(imaginary, self.component),
            )

        candidate = collective_call(
            evaluate_local,
            comm=comm,
            label=f"Harmonic probe {self.name!r} local evaluation",
        )
        candidates = tuple(comm.allgather(candidate))
        owners = [rank for rank, value in enumerate(candidates) if value is not None]
        if not owners:
            raise ValueError(
                f"Harmonic probe {self.name!r} could not locate point {self.at}."
            )
        real, imaginary = candidates[owners[0]]
        return _finite_scalar(complex(real, imaginary), name=self.name)

    def to_ir(self) -> dict[str, object]:
        return {
            "kind": "harmonic_point_probe",
            "name": self.name,
            "unit": self.unit,
            "description": self.description,
            "evaluate": self.evaluate,
            "point": self.at,
            "component": self.component,
            "padding": self.padding,
            "owner_selection": "lowest_rank_owned_cell",
        }

    def summary(self) -> dict[str, object]:
        return {
            "kind": "harmonic_point_probe",
            "name": self.name,
            "unit": self.unit,
            "description": self.description,
            "point": self.at,
            "component": self.component,
            "padding": self.padding,
            "owner_selection": "lowest_rank_owned_cell",
            "mpi_contract": "local_owned_cell_probe_then_framework_select",
            "evaluator": _evaluator_summary(self.evaluate),
        }


def harmonic_response(
    name: str,
    evaluate,
    *,
    unit: str | None = None,
    description: str = "",
    reduction: str | None = None,
) -> HarmonicResponse:
    """Declare a scalar response with an explicit rank-local reduction."""

    return HarmonicResponse(
        name=name,
        evaluate=evaluate,
        unit=unit,
        description=description,
        reduction=reduction,
    )


def harmonic_average_response(
    name: str,
    evaluate,
    *,
    on,
    study=None,
    unit: str | None = None,
    description: str = "",
) -> HarmonicAverageResponse:
    """Declare a complex region average safe for serial and MPI sweeps."""

    return HarmonicAverageResponse(
        name=name,
        evaluate=evaluate,
        on=on,
        study=study,
        unit=unit,
        description=description,
    )


def harmonic_probe_response(
    name: str,
    evaluate,
    *,
    at,
    component: int | None = None,
    padding: float = 1.0e-10,
    unit: str | None = None,
    description: str = "",
) -> HarmonicProbeResponse:
    """Declare a complex point response safe for serial and MPI sweeps."""

    return HarmonicProbeResponse(
        name=name,
        evaluate=evaluate,
        at=tuple(at),
        component=component,
        padding=padding,
        unit=unit,
        description=description,
    )


def _local_probe_candidate(field, point, padding: float):
    function = field_api.unwrap(field)
    domain = function.function_space.mesh
    selected = np.asarray(point, dtype=domain.geometry.x.dtype)
    geometric_dimension = int(domain.geometry.dim)
    storage_dimension = int(domain.geometry.x.shape[1])
    if selected.shape != (geometric_dimension,):
        if selected.shape != (storage_dimension,):
            raise ValueError(
                "Harmonic probe coordinates must match the mesh geometric dimension."
            )
        coordinate = selected
    else:
        coordinate = np.zeros(storage_dimension, dtype=domain.geometry.x.dtype)
        coordinate[:geometric_dimension] = selected
    topology = domain.topology
    owned_count = int(topology.index_map(topology.dim).size_local)
    if owned_count == 0:
        return None
    owned = np.arange(owned_count, dtype=np.int32)
    points = np.ascontiguousarray(coordinate.reshape(1, -1))
    tree = geometry_api.bb_tree(
        domain,
        topology.dim,
        padding=padding,
        entities=owned,
    )
    candidates = geometry_api.compute_collisions_points(tree, points)
    collisions = geometry_api.compute_colliding_cells(domain, candidates, points)
    cells = np.asarray(collisions.links(0), dtype=np.int32)
    cells = cells[cells < owned_count]
    if not cells.size:
        return None
    cell = int(np.min(cells))
    value_shape = tuple(getattr(function, "ufl_shape", ()))
    value_size = int(np.prod(value_shape, dtype=int)) if value_shape else 1
    evaluated = np.asarray(
        function.eval(points, np.asarray([cell], dtype=np.int32))
    ).reshape(1, value_size)
    return evaluated[0]


def _select_probe_component(value, component: int | None) -> float:
    array = np.asarray(value)
    if component is None:
        if array.shape not in {(), (1,)}:
            raise ValueError(
                "A vector harmonic probe requires component= to select one scalar."
            )
        return float(array.reshape(-1)[0])
    flat = array.reshape(-1)
    if component >= flat.size:
        raise ValueError(
            f"Harmonic probe component {component} is outside value size {flat.size}."
        )
    return float(flat[component])


def _evaluator_summary(evaluate) -> dict[str, object]:
    return {
        "module": getattr(evaluate, "__module__", None),
        "qualname": getattr(
            evaluate,
            "__qualname__",
            getattr(evaluate, "__name__", type(evaluate).__name__),
        ),
        "identity_contract": "scientific_input_manifest",
    }


__all__ = [
    "HarmonicAverageResponse",
    "HarmonicProbeResponse",
    "HarmonicResponse",
    "harmonic_average_response",
    "harmonic_probe_response",
    "harmonic_response",
]
