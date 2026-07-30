"""Explicit contracts for neural operators and physics-informed learning.

These records are design and validation contracts, not claims that arbitrary
AgentFEM/UFL models can already be trained as neural operators or PINNs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


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
    "FieldEncoding",
    "NeuralOperatorSpec",
    "PINNSpec",
    "PhysicsCondition",
    "PhysicsResidual",
]
