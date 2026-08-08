"""Explicit contracts for neural operators and physics-informed learning.

These records are design and validation contracts, not claims that arbitrary
AgentFEM/UFL models can already be trained as neural operators or PINNs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class AffineCoordinateMap:
    """Explicit affine map from observation coordinates to model coordinates.

    Field observations, publication images, experiments, and a finite-element
    mesh often use different origins, axes, or length units.  This record keeps
    that registration outside plotting code.  For row-wise points ``q`` the
    convention is ``x_model = q @ matrix.T + offset``.
    """

    matrix: np.ndarray
    offset: np.ndarray
    source_coordinate_system: str = "observation"
    target_coordinate_system: str = "model"
    source_unit: str | None = None
    target_unit: str | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        offset = np.asarray(self.offset, dtype=float).reshape(-1)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("AffineCoordinateMap.matrix must be square.")
        if matrix.shape[0] not in {1, 2, 3}:
            raise ValueError("AffineCoordinateMap supports one, two, or three dimensions.")
        if offset.shape != (matrix.shape[0],):
            raise ValueError("AffineCoordinateMap.offset dimension must match matrix.")
        if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(offset)):
            raise ValueError("AffineCoordinateMap coefficients must be finite.")
        determinant = float(np.linalg.det(matrix))
        if not np.isfinite(determinant) or abs(determinant) <= np.finfo(float).eps:
            raise ValueError("AffineCoordinateMap.matrix must be invertible.")
        source = str(self.source_coordinate_system).strip()
        target = str(self.target_coordinate_system).strip()
        if not source or not target:
            raise ValueError("AffineCoordinateMap coordinate-system names must not be empty.")
        object.__setattr__(self, "matrix", matrix.copy())
        object.__setattr__(self, "offset", offset.copy())
        object.__setattr__(self, "source_coordinate_system", source)
        object.__setattr__(self, "target_coordinate_system", target)
        source_unit = None if self.source_unit is None else str(self.source_unit).strip()
        target_unit = None if self.target_unit is None else str(self.target_unit).strip()
        if source_unit == "" or target_unit == "":
            raise ValueError("AffineCoordinateMap units must not be empty strings.")
        object.__setattr__(self, "source_unit", source_unit)
        object.__setattr__(self, "target_unit", target_unit)

    @classmethod
    def identity(
        cls,
        dimension: int,
        *,
        coordinate_system: str = "cartesian",
        unit: str | None = None,
    ) -> "AffineCoordinateMap":
        selected = int(dimension)
        if selected not in {1, 2, 3}:
            raise ValueError("AffineCoordinateMap dimension must be one, two, or three.")
        return cls(
            np.eye(selected),
            np.zeros(selected),
            source_coordinate_system=coordinate_system,
            target_coordinate_system=coordinate_system,
            source_unit=unit,
            target_unit=unit,
        )

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0])

    def map_points(self, points) -> np.ndarray:
        """Map row-wise observation points into model coordinates."""

        selected = np.asarray(points, dtype=float)
        if selected.ndim == 1:
            selected = selected.reshape((1, -1))
        if selected.ndim != 2 or selected.shape[1] != self.dimension:
            raise ValueError(
                "AffineCoordinateMap points must have shape "
                f"(count, {self.dimension})."
            )
        if np.any(~np.isfinite(selected)):
            raise ValueError("AffineCoordinateMap points must be finite.")
        return selected @ self.matrix.T + self.offset

    def inverse(self) -> "AffineCoordinateMap":
        """Return the exact inverse map with source and target exchanged."""

        inverse = np.linalg.inv(self.matrix)
        return AffineCoordinateMap(
            inverse,
            -inverse @ self.offset,
            source_coordinate_system=self.target_coordinate_system,
            target_coordinate_system=self.source_coordinate_system,
            source_unit=self.target_unit,
            target_unit=self.source_unit,
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "affine_coordinate_map",
            "convention": "target = source @ matrix.T + offset",
            "dimension": self.dimension,
            "matrix": self.matrix.tolist(),
            "offset": self.offset.tolist(),
            "source_coordinate_system": self.source_coordinate_system,
            "target_coordinate_system": self.target_coordinate_system,
            "source_unit": self.source_unit,
            "target_unit": self.target_unit,
        }


@dataclass(frozen=True)
class ObservationGrid:
    """Mesh-independent Cartesian coordinates for field learning and sensing."""

    axes: tuple[np.ndarray, ...]
    axis_names: tuple[str, ...]
    coordinate_system: str = "cartesian"
    order: str = "C"
    coordinate_unit: str | None = None

    def __post_init__(self) -> None:
        axes = tuple(np.asarray(axis, dtype=float).reshape(-1) for axis in self.axes)
        names = tuple(str(name).strip() for name in self.axis_names)
        if not 1 <= len(axes) <= 3:
            raise ValueError("ObservationGrid requires one, two, or three axes.")
        if len(names) != len(axes) or any(not name for name in names):
            raise ValueError("ObservationGrid requires one non-empty name per axis.")
        if len(set(names)) != len(names):
            raise ValueError("ObservationGrid axis names must be unique.")
        for name, axis in zip(names, axes):
            if axis.size < 2:
                raise ValueError(f"ObservationGrid axis {name!r} needs at least two points.")
            if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
                raise ValueError(
                    f"ObservationGrid axis {name!r} must be finite and strictly increasing."
                )
        selected_order = str(self.order).upper()
        if selected_order not in {"C", "F"}:
            raise ValueError("ObservationGrid order must be 'C' or 'F'.")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "axis_names", names)
        object.__setattr__(self, "coordinate_system", str(self.coordinate_system))
        object.__setattr__(self, "order", selected_order)
        coordinate_unit = (
            None if self.coordinate_unit is None else str(self.coordinate_unit).strip()
        )
        if coordinate_unit == "":
            raise ValueError("ObservationGrid.coordinate_unit must not be empty.")
        object.__setattr__(self, "coordinate_unit", coordinate_unit)

    @classmethod
    def from_axes(cls, **axes):
        """Create a grid from named coordinate arrays, for example ``x=..., y=...``."""

        return cls(tuple(axes.values()), tuple(axes))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(axis.size) for axis in self.axes)

    @property
    def point_count(self) -> int:
        return int(np.prod(self.shape, dtype=int))

    def points(self) -> np.ndarray:
        """Return flattened physical coordinates in the declared array order."""

        coordinates = np.meshgrid(*self.axes, indexing="ij")
        return np.column_stack(
            [component.reshape(-1, order=self.order) for component in coordinates]
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "observation_grid",
            "coordinate_system": self.coordinate_system,
            "coordinate_unit": self.coordinate_unit,
            "axis_names": self.axis_names,
            "axes": {
                name: axis.tolist()
                for name, axis in zip(self.axis_names, self.axes)
            },
            "shape": self.shape,
            "order": self.order,
            "point_count": self.point_count,
        }


def regular_grid(
    *,
    bounds,
    shape,
    axis_names=None,
    coordinate_system: str = "cartesian",
    order: str = "C",
    coordinate_unit: str | None = None,
) -> ObservationGrid:
    """Create an evenly spaced observation grid from physical bounds."""

    selected_bounds = tuple(tuple(float(value) for value in pair) for pair in bounds)
    selected_shape = tuple(int(value) for value in shape)
    if len(selected_bounds) != len(selected_shape):
        raise ValueError("regular_grid bounds and shape must have the same dimension.")
    names = tuple(axis_names or ("x", "y", "z")[: len(selected_shape)])
    axes = []
    for (lower, upper), count in zip(selected_bounds, selected_shape):
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError("regular_grid bounds must be finite and increasing.")
        if count < 2:
            raise ValueError("regular_grid shape entries must be at least two.")
        axes.append(np.linspace(lower, upper, count))
    return ObservationGrid(
        tuple(axes),
        names,
        coordinate_system=coordinate_system,
        order=order,
        coordinate_unit=coordinate_unit,
    )


@dataclass(frozen=True)
class FieldEncoding:
    """How a physical field becomes a machine-learning tensor."""

    name: str
    role: str
    unit: str | None
    components: tuple[str, ...] = ()
    representation: str = "mesh_dofs"
    shape: tuple[int, ...] | None = None
    geometry_encoding: str = "coordinates"
    mesh_policy: str = "fixed_mesh"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("FieldEncoding.name must not be empty.")
        role = self.role.lower().replace("-", "_")
        if role not in {"input", "output", "condition", "coordinate"}:
            raise ValueError(
                "FieldEncoding.role must be input, output, condition, or coordinate."
            )
        representation = self.representation.lower().replace("-", "_")
        supported = {
            "mesh_dofs",
            "structured_grid",
            "point_samples",
            "graph",
            "basis_coefficients",
        }
        if representation not in supported:
            raise ValueError(
                f"Unknown field representation {self.representation!r}; "
                f"expected one of {sorted(supported)}."
            )
        mesh_policy = self.mesh_policy.lower().replace("-", "_")
        if mesh_policy not in {
            "fixed_mesh",
            "registered_mesh_family",
            "mesh_independent_coordinates",
        }:
            raise ValueError(f"Unknown mesh policy {self.mesh_policy!r}.")
        if self.shape is not None and any(int(value) <= 0 for value in self.shape):
            raise ValueError("FieldEncoding.shape dimensions must be positive.")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "mesh_policy", mesh_policy)
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(
            self,
            "shape",
            None if self.shape is None else tuple(int(value) for value in self.shape),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "unit": self.unit,
            "components": self.components,
            "representation": self.representation,
            "shape": self.shape,
            "geometry_encoding": self.geometry_encoding,
            "mesh_policy": self.mesh_policy,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class NeuralOperatorSpec:
    """Function-to-function learning contract for an external trainer."""

    architecture: str
    inputs: tuple[FieldEncoding, ...]
    outputs: tuple[FieldEncoding, ...]
    boundary_encoding: str
    parameter_inputs: tuple[str, ...] = ()
    temporal_encoding: str | None = None
    required_checks: tuple[str, ...] = (
        "held_out_field_error",
        "boundary_condition_error",
        "conservation_or_balance_error",
        "out_of_distribution_test",
    )
    status: str = field(default="contract_only", init=False)

    def __post_init__(self) -> None:
        if not self.inputs or not self.outputs:
            raise ValueError("NeuralOperatorSpec requires input and output fields.")
        if any(field.role not in {"input", "condition", "coordinate"} for field in self.inputs):
            raise ValueError("Neural-operator inputs have incompatible field roles.")
        if any(field.role != "output" for field in self.outputs):
            raise ValueError("Neural-operator outputs must use role='output'.")
        architecture = self.architecture.lower().replace("-", "_")
        if architecture in {"fno", "fourier_neural_operator"}:
            incompatible = [
                field.name
                for field in (*self.inputs, *self.outputs)
                if field.representation != "structured_grid"
            ]
            if incompatible:
                raise ValueError(
                    "A basic FNO contract requires structured_grid encodings; "
                    f"incompatible fields={incompatible}."
                )
        object.__setattr__(self, "architecture", architecture)
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "parameter_inputs", tuple(self.parameter_inputs))
        object.__setattr__(self, "required_checks", tuple(self.required_checks))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "neural_operator_spec",
            "status": self.status,
            "architecture": self.architecture,
            "inputs": [item.summary() for item in self.inputs],
            "outputs": [item.summary() for item in self.outputs],
            "parameter_inputs": self.parameter_inputs,
            "boundary_encoding": self.boundary_encoding,
            "temporal_encoding": self.temporal_encoding,
            "required_checks": self.required_checks,
        }


@dataclass(frozen=True)
class PhysicsResidual:
    """One explicit differentiable residual used in a physics loss."""

    name: str
    equation: str
    form: str
    dependent_fields: tuple[str, ...]
    independent_variables: tuple[str, ...]
    unit: str | None = None
    weight: float = 1.0
    implementation: str | None = None

    def __post_init__(self) -> None:
        form = self.form.lower().replace("-", "_")
        if form not in {"strong", "weak", "discrete"}:
            raise ValueError("PhysicsResidual.form must be strong, weak, or discrete.")
        if not self.equation.strip():
            raise ValueError("PhysicsResidual.equation must be explicit.")
        if self.weight <= 0.0:
            raise ValueError("PhysicsResidual.weight must be positive.")
        object.__setattr__(self, "form", form)
        object.__setattr__(self, "dependent_fields", tuple(self.dependent_fields))
        object.__setattr__(
            self,
            "independent_variables",
            tuple(self.independent_variables),
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "equation": self.equation,
            "form": self.form,
            "dependent_fields": self.dependent_fields,
            "independent_variables": self.independent_variables,
            "unit": self.unit,
            "weight": self.weight,
            "implementation": self.implementation,
        }


@dataclass(frozen=True)
class PhysicsCondition:
    """Boundary, initial, interface, or observation condition in a loss."""

    name: str
    kind: str
    target: str
    location: str
    value: object | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        kind = self.kind.lower().replace("-", "_")
        if kind not in {"boundary", "initial", "interface", "observation"}:
            raise ValueError(f"Unknown physics-condition kind {self.kind!r}.")
        if self.weight <= 0.0:
            raise ValueError("PhysicsCondition.weight must be positive.")
        object.__setattr__(self, "kind", kind)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "target": self.target,
            "location": self.location,
            "value": self.value,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class PINNSpec:
    """Physics-informed training contract for selected explicit residuals."""

    fields: tuple[FieldEncoding, ...]
    residuals: tuple[PhysicsResidual, ...]
    conditions: tuple[PhysicsCondition, ...]
    purpose: str = "inverse_or_data_physics_fusion"
    autodiff_backend: str = "external"
    collocation_policy: Mapping[str, object] = field(default_factory=dict)
    required_checks: tuple[str, ...] = (
        "independent_solution_error",
        "condition_error",
        "residual_distribution",
        "parameter_identifiability",
    )
    status: str = field(default="contract_only", init=False)

    def __post_init__(self) -> None:
        if not self.fields or not self.residuals:
            raise ValueError("PINNSpec requires fields and explicit residuals.")
        if not self.conditions:
            raise ValueError(
                "PINNSpec requires boundary, initial, interface, or observation conditions."
            )
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "residuals", tuple(self.residuals))
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "collocation_policy", dict(self.collocation_policy))
        object.__setattr__(self, "required_checks", tuple(self.required_checks))

    def summary(self) -> dict[str, object]:
        return {
            "kind": "pinn_spec",
            "status": self.status,
            "purpose": self.purpose,
            "autodiff_backend": self.autodiff_backend,
            "fields": [item.summary() for item in self.fields],
            "residuals": [item.summary() for item in self.residuals],
            "conditions": [item.summary() for item in self.conditions],
            "collocation_policy": dict(self.collocation_policy),
            "required_checks": self.required_checks,
        }


__all__ = [
    "AffineCoordinateMap",
    "FieldEncoding",
    "NeuralOperatorSpec",
    "ObservationGrid",
    "PINNSpec",
    "PhysicsCondition",
    "PhysicsResidual",
    "regular_grid",
]
