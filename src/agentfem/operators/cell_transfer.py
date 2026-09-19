# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Exact FEM transfers between continuous fields and cell-average gradients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc

from agentfem import assembly
from agentfem.mesh import owned_cell_measures


@dataclass(frozen=True)
class ConvectedCellFiberKinematics:
    """Current cell tangents, stretch, and unit direction for one fibre family."""

    current_tangents: np.ndarray
    stretch: np.ndarray
    direction: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "convected_cell_fiber_kinematics",
            "local_and_ghost_cells": int(self.stretch.size),
            "minimum_stretch": float(np.min(self.stretch)),
            "maximum_stretch": float(np.max(self.stretch)),
            "derivation": "reference_tangent_coordinates_convected_by_cell_tangents",
        }


@dataclass(frozen=True)
class ConvectedCellFiberIncrement:
    """Directional derivative of convected cell-fibre kinematics."""

    tangent_increment: np.ndarray
    stretch_increment: np.ndarray
    direction_increment: np.ndarray


@dataclass(frozen=True)
class ConvectedCellFiberOperator:
    """Displacement-derived surface tangents and one convected fibre family."""

    transfer: "CellAverageGradientOperator"
    reference_tangents: np.ndarray
    reference_tangent_coordinates: np.ndarray

    def __post_init__(self) -> None:
        if tuple(self.transfer.source_space.element.value_shape) != (3,):
            raise ValueError("Convected cell fibres require a 3-component source field.")
        if self.transfer.value_shape != (3, 2):
            raise ValueError(
                "Convected cell fibres require a 2D parameter mesh and 3x2 gradients."
            )
        domain = self.transfer.source_space.mesh
        cell_map = domain.topology.index_map(domain.topology.dim)
        total = int(cell_map.size_local + cell_map.num_ghosts)
        tangents = np.asarray(self.reference_tangents, dtype=float)
        coordinates = np.asarray(self.reference_tangent_coordinates, dtype=float)
        if tangents.shape == (3, 2):
            tangents = np.broadcast_to(tangents, (total, 3, 2)).copy()
        if coordinates.shape == (2,):
            coordinates = np.broadcast_to(coordinates, (total, 2)).copy()
        if tangents.shape != (total, 3, 2) or not np.all(np.isfinite(tangents)):
            raise ValueError(
                "reference_tangents must be finite with shape (cells, 3, 2)."
            )
        if coordinates.shape != (total, 2) or not np.all(np.isfinite(coordinates)):
            raise ValueError(
                "reference_tangent_coordinates must be finite with shape (cells, 2)."
            )
        for cell, item in enumerate(tangents):
            if np.linalg.matrix_rank(item) != 2:
                raise ValueError(f"reference_tangents[{cell}] must have rank two.")
            reference_vector = item @ coordinates[cell]
            if np.linalg.norm(reference_vector) <= np.finfo(float).eps:
                raise ValueError(
                    f"reference_tangent_coordinates[{cell}] define a zero fibre."
                )
        object.__setattr__(self, "reference_tangents", tangents.copy())
        object.__setattr__(
            self, "reference_tangent_coordinates", coordinates.copy()
        )

    @property
    def total_cells(self) -> int:
        return int(self.reference_tangents.shape[0])

    def _gradient_values(self, field) -> np.ndarray:
        gradient = self.transfer.apply(field)
        block_size = int(gradient.function_space.dofmap.index_map_bs)
        values = np.empty((self.total_cells, 3, 2), dtype=float)
        for cell in range(self.total_cells):
            dof = int(gradient.function_space.dofmap.cell_dofs(cell)[0])
            start = dof * block_size
            values[cell] = gradient.x.array[
                start : start + block_size
            ].reshape((3, 2))
        return values

    def apply(self, displacement) -> ConvectedCellFiberKinematics:
        """Evaluate current tangents, fibre stretch, and unit direction."""

        gradients = self._gradient_values(displacement)
        current = self.reference_tangents + gradients
        vectors = np.einsum(
            "cij,cj->ci", current, self.reference_tangent_coordinates
        )
        stretch = np.linalg.norm(vectors, axis=1)
        if np.any(stretch <= np.finfo(float).eps):
            raise ValueError("Current deformation collapses at least one fibre.")
        return ConvectedCellFiberKinematics(
            current_tangents=current,
            stretch=stretch,
            direction=vectors / stretch[:, None],
        )

    def directional_derivative(
        self, displacement, increment
    ) -> ConvectedCellFiberIncrement:
        """Differentiate the kinematics along one displacement increment."""

        state = self.apply(displacement)
        tangent_increment = self._gradient_values(increment)
        vector_increment = np.einsum(
            "cij,cj->ci",
            tangent_increment,
            self.reference_tangent_coordinates,
        )
        stretch_increment = np.sum(state.direction * vector_increment, axis=1)
        direction_increment = (
            vector_increment - state.direction * stretch_increment[:, None]
        ) / state.stretch[:, None]
        return ConvectedCellFiberIncrement(
            tangent_increment=tangent_increment,
            stretch_increment=stretch_increment,
            direction_increment=direction_increment,
        )

    def apply_adjoint(
        self,
        displacement,
        *,
        direction_duals,
        tangent_duals=None,
        stretch_duals=None,
    ):
        """Return the exact source-space adjoint of declared kinematic duals."""

        state = self.apply(displacement)
        direction_duals = np.asarray(direction_duals, dtype=float)
        if direction_duals.shape != (self.total_cells, 3) or not np.all(
            np.isfinite(direction_duals)
        ):
            raise ValueError("direction_duals must have shape (cells, 3).")
        if tangent_duals is None:
            tangent_duals = np.zeros((self.total_cells, 3, 2), dtype=float)
        else:
            tangent_duals = np.asarray(tangent_duals, dtype=float)
        if tangent_duals.shape != (self.total_cells, 3, 2) or not np.all(
            np.isfinite(tangent_duals)
        ):
            raise ValueError("tangent_duals must have shape (cells, 3, 2).")
        if stretch_duals is None:
            stretch_duals = np.zeros(self.total_cells, dtype=float)
        else:
            stretch_duals = np.asarray(stretch_duals, dtype=float)
        if stretch_duals.shape != (self.total_cells,) or not np.all(
            np.isfinite(stretch_duals)
        ):
            raise ValueError("stretch_duals must have shape (cells,).")
        gradient_duals = tangent_duals.copy()
        for cell, direction in enumerate(state.direction):
            projector = np.eye(3) - np.outer(direction, direction)
            vector_dual = (
                projector @ direction_duals[cell] / state.stretch[cell]
                + stretch_duals[cell] * direction
            )
            gradient_duals[cell] += np.outer(
                vector_dual,
                self.reference_tangent_coordinates[cell],
            )
        return self.transfer.apply_adjoint(gradient_duals)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "convected_cell_fiber_operator",
            "primary_field": "three_component_surface_displacement",
            "outputs": ("current_tangents", "fiber_stretch", "fiber_direction"),
            "linearization": "exact_directional_derivative",
            "adjoint": "exact_to_source_displacement_space",
            "scope": "kinematic_transfer_not_shell_equilibrium",
            "gradient_transfer": self.transfer.as_dict(),
        }


@dataclass(frozen=True)
class CellAverageGradientOperator:
    """Map a FEM field to DG0 cell-average gradients and apply its transpose."""

    source_space: object
    target_space: object
    matrix: object
    cell_weights: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.cell_weights, dtype=float)
        domain = self.source_space.mesh
        owned = int(domain.topology.index_map(domain.topology.dim).size_local)
        if weights.shape != (owned,) or not np.all(np.isfinite(weights)):
            raise ValueError("cell_weights require one finite value per owned cell.")
        if np.any(weights <= 0.0):
            raise ValueError("cell_weights must be strictly positive.")
        if self.target_space.mesh is not domain:
            raise ValueError("source_space and target_space must share one mesh.")
        object.__setattr__(self, "cell_weights", weights.copy())

    @property
    def value_shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.target_space.element.value_shape)

    def apply(self, field) -> object:
        """Return the DG0 cell-average gradient of one source-space field."""

        if field.function_space is not self.source_space:
            raise ValueError("field must belong to the operator source_space.")
        integrated = self.matrix.createVecLeft()
        self.matrix.mult(field.x.petsc_vec, integrated)
        result = fem.Function(self.target_space, name=f"GRAD_{field.name or 'FIELD'}")
        block_size = int(self.target_space.dofmap.index_map_bs)
        for cell, weight in enumerate(self.cell_weights):
            dofs = np.asarray(self.target_space.dofmap.cell_dofs(cell), dtype=np.int32)
            if dofs.shape != (1,):
                integrated.destroy()
                raise RuntimeError("DG0 gradient transfer requires one block dof per cell.")
            start = int(dofs[0]) * block_size
            result.x.array[start : start + block_size] = (
                integrated.array_r[start : start + block_size] / weight
            )
        result.x.scatter_forward()
        integrated.destroy()
        return result

    def apply_adjoint(self, cell_gradient_duals):
        """Map local-plus-ghost cell-gradient duals to a source PETSc vector."""

        assembled = assembly.assemble_cell_residual(
            self.target_space,
            cell_gradient_duals,
        )
        scaled = self.matrix.createVecLeft()
        scaled.set(0.0)
        block_size = int(self.target_space.dofmap.index_map_bs)
        for cell, weight in enumerate(self.cell_weights):
            dof = int(self.target_space.dofmap.cell_dofs(cell)[0])
            start = dof * block_size
            scaled.array[start : start + block_size] = (
                assembled.array_r[start : start + block_size] / weight
            )
        result = self.matrix.createVecRight()
        self.matrix.multTranspose(scaled, result)
        assembled.destroy()
        scaled.destroy()
        return result

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "cell_average_gradient_operator",
            "source_value_shape": list(self.source_space.element.value_shape),
            "target_value_shape": list(self.target_space.element.value_shape),
            "owned_cells": int(self.cell_weights.size),
            "forward": "dg0_cell_average_of_ufl_gradient",
            "adjoint": "exact_mass_inverse_weighted_matrix_transpose",
            "mpi_semantics": "cell_duals_reverse_scattered_before_transpose",
        }


def cell_average_gradient(source_space) -> CellAverageGradientOperator:
    """Assemble a reusable FEM-to-DG0 cell-average gradient transfer."""

    domain = source_space.mesh
    source_shape = tuple(int(value) for value in source_space.element.value_shape)
    target_shape = (*source_shape, int(domain.geometry.dim))
    target_space = fem.functionspace(domain, ("DG", 0, target_shape))
    trial = ufl.TrialFunction(source_space)
    test = ufl.TestFunction(target_space)
    matrix = fem_petsc.assemble_matrix(
        fem.form(ufl.inner(ufl.grad(trial), test) * ufl.dx(domain=domain))
    )
    matrix.assemble()
    return CellAverageGradientOperator(
        source_space=source_space,
        target_space=target_space,
        matrix=matrix,
        cell_weights=owned_cell_measures(domain),
    )


def convected_cell_fiber(
    transfer: CellAverageGradientOperator,
    *,
    reference_tangents,
    reference_tangent_coordinates,
) -> ConvectedCellFiberOperator:
    """Create a displacement-derived cell-fibre kinematic transfer."""

    return ConvectedCellFiberOperator(
        transfer=transfer,
        reference_tangents=reference_tangents,
        reference_tangent_coordinates=reference_tangent_coordinates,
    )


__all__ = [
    "CellAverageGradientOperator",
    "ConvectedCellFiberIncrement",
    "ConvectedCellFiberKinematics",
    "ConvectedCellFiberOperator",
    "cell_average_gradient",
    "convected_cell_fiber",
]
