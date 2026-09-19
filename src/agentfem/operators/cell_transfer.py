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


__all__ = ["CellAverageGradientOperator", "cell_average_gradient"]
