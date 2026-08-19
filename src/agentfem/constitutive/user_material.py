"""Contracts for future Abaqus UMAT/UHYPER interoperability.

This module deliberately defines and validates the material-point boundary;
it does not claim that arbitrary Abaqus user subroutines can already be
executed by AgentFEM.  A real bridge additionally needs quadrature-state
storage, a global constitutive driver, compiler/ABI adapters, and reference
comparisons against Abaqus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class MaterialPointInput:
    """Solver-neutral finite-strain input for one material-point update."""

    deformation_gradient_old: np.ndarray
    deformation_gradient_new: np.ndarray
    time: float
    time_increment: float
    properties: np.ndarray
    state_old: np.ndarray
    temperature: float | None = None
    temperature_increment: float | None = None
    field_variables: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("deformation_gradient_old", "deformation_gradient_new"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3, 3) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite 3x3 matrix.")
            if np.linalg.det(value) <= 0.0:
                raise ValueError(f"{name} must have a positive determinant.")
            object.__setattr__(self, name, value.copy())
        if not np.isfinite(self.time) or not np.isfinite(self.time_increment):
            raise ValueError("Material-point time values must be finite.")
        if self.time_increment <= 0.0:
            raise ValueError("Material-point time_increment must be positive.")
        for name in ("properties", "state_old"):
            value = np.asarray(getattr(self, name), dtype=float).reshape(-1)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain finite values.")
            object.__setattr__(self, name, value.copy())
        if self.field_variables is not None:
            fields = np.asarray(self.field_variables, dtype=float).reshape(-1)
            if not np.all(np.isfinite(fields)):
                raise ValueError("field_variables must contain finite values.")
            object.__setattr__(self, "field_variables", fields.copy())


@dataclass(frozen=True)
class MaterialPointOutput:
    """Constitutive response returned to a nonlinear finite-element driver."""

    cauchy_stress: np.ndarray
    consistent_tangent: np.ndarray
    state_new: np.ndarray
    strain_energy_density: float | None = None
    suggested_time_scale: float = 1.0

    def __post_init__(self) -> None:
        stress = np.asarray(self.cauchy_stress, dtype=float)
        tangent = np.asarray(self.consistent_tangent, dtype=float)
        state = np.asarray(self.state_new, dtype=float).reshape(-1)
        if stress.shape != (3, 3) or not np.all(np.isfinite(stress)):
            raise ValueError("cauchy_stress must be a finite 3x3 tensor.")
        if tangent.shape != (6, 6) or not np.all(np.isfinite(tangent)):
            raise ValueError("consistent_tangent must be a finite 6x6 matrix.")
        if not np.all(np.isfinite(state)):
            raise ValueError("state_new must contain finite values.")
        if (
            self.strain_energy_density is not None
            and not np.isfinite(self.strain_energy_density)
        ):
            raise ValueError("strain_energy_density must be finite when provided.")
        if (
            not np.isfinite(self.suggested_time_scale)
            or self.suggested_time_scale <= 0.0
        ):
            raise ValueError("suggested_time_scale must be finite and positive.")
        object.__setattr__(self, "cauchy_stress", stress.copy())
        object.__setattr__(self, "consistent_tangent", tangent.copy())
        object.__setattr__(self, "state_new", state.copy())


@runtime_checkable
class UserMaterial(Protocol):
    """Protocol implemented by native or adapted material-point models."""

    name: str

    def update(self, point: MaterialPointInput) -> MaterialPointOutput:
        """Advance one integration point and return stress, state, and tangent."""


@dataclass(frozen=True)
class AbaqusUserMaterialBridge:
    """Truthful capability description for an intended UMAT/UHYPER adapter."""

    kind: str
    source: str
    material_name: str
    property_count: int
    state_variable_count: int = 0
    tensor_order: str = "11,22,33,12,13,23"
    status: str = "adapter_specification"

    def __post_init__(self) -> None:
        kind = self.kind.upper()
        if kind not in {"UMAT", "UHYPER"}:
            raise ValueError("Abaqus user-material kind must be UMAT or UHYPER.")
        if not self.source:
            raise ValueError("Abaqus user-material source must be named.")
        if self.property_count < 0 or self.state_variable_count < 0:
            raise ValueError("Property and state-variable counts cannot be negative.")
        object.__setattr__(self, "kind", kind)

    @property
    def executable(self) -> bool:
        """Return false until a compiled adapter and global driver exist."""

        return False

    def summary(self) -> dict[str, object]:
        return {
            "kind": "abaqus_user_material_bridge",
            "interface": self.kind,
            "source": self.source,
            "material_name": self.material_name,
            "property_count": self.property_count,
            "state_variable_count": self.state_variable_count,
            "tensor_order": self.tensor_order,
            "status": self.status,
            "executable": self.executable,
            "required_runtime": (
                "quadrature state driver",
                "compiler/ABI adapter",
                "Abaqus-to-AgentFEM tensor and rotation semantics",
                "consistent-tangent verification",
            ),
        }
