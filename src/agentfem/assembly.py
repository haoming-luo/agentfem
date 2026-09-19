# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Assembly helpers for standard finite-element workflows."""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

import dolfinx.fem.petsc as fem_petsc


def make_form(ufl_form):
    """Compile a UFL form for assembly."""

    return fem.form(ufl_form)


def assemble_vector(form):
    """Assemble a vector and accumulate ghost contributions to owned entries."""

    vector = fem_petsc.assemble_vector(form)
    vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return vector


def assemble_cell_residual(space, cell_contributions):
    """Accumulate local-and-ghost cell contributions into a DG0 PETSc vector.

    ``cell_contributions`` must provide one scalar or value-shaped entry for
    every local and ghost cell. Contributions at ghost cells are reverse-
    scattered with addition to their owner, then forward-scattered so the
    returned ghosted vector has consistent owned and ghost entries.

    This adapter is intentionally restricted to discontinuous degree-zero
    spaces: their one block dof per cell makes the cell identity explicit.
    """

    element = space.element
    basix_element = element.basix_element
    if int(basix_element.degree) != 0 or not bool(basix_element.discontinuous):
        raise ValueError("assemble_cell_residual requires a discontinuous DG0 space.")
    value_shape = tuple(int(value) for value in element.value_shape)
    block_size = int(space.dofmap.index_map_bs)
    expected_block_size = int(np.prod(value_shape, dtype=int)) if value_shape else 1
    if block_size != expected_block_size:
        raise ValueError("DG0 space block size does not match its value shape.")
    domain = space.mesh
    cell_map = domain.topology.index_map(domain.topology.dim)
    total_cells = int(cell_map.size_local + cell_map.num_ghosts)
    contributions = np.asarray(cell_contributions, dtype=float)
    expected_shape = (total_cells, *value_shape)
    if contributions.shape != expected_shape or not np.all(np.isfinite(contributions)):
        raise ValueError(
            f"cell_contributions must be finite with shape {expected_shape}."
        )
    flattened = contributions.reshape((total_cells, block_size))
    holder = fem.Function(space)
    vector = holder.x.petsc_vec.duplicate()
    with vector.localForm() as local:
        local.set(0.0)
        local_values = local.array
        for cell in range(total_cells):
            dofs = np.asarray(space.dofmap.cell_dofs(cell), dtype=np.int32)
            if dofs.shape != (1,):
                vector.destroy()
                raise RuntimeError("DG0 cell residual requires one block dof per cell.")
            start = int(dofs[0]) * block_size
            local_values[start : start + block_size] += flattened[cell]
    vector.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )
    vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT_VALUES,
        mode=PETSc.ScatterMode.FORWARD,
    )
    return vector


def assemble_matrix(form, bcs=None):
    """Assemble a matrix and apply optional strong Dirichlet BC structure."""

    matrix = fem_petsc.assemble_matrix(form, bcs=[] if bcs is None else bcs)
    matrix.assemble()
    return matrix


def assemble_lumped_operator(V, coefficient=1.0, measure=ufl.dx) -> np.ndarray:
    """Assemble a diagonal/lumped operator vector on ``V``.

    ``V`` is the DOLFINx function space that owns the degrees of freedom. The
    coefficient may be a Python scalar, ``fem.Constant``, ``fem.Function``, or
    any compatible UFL expression.
    """

    test_function = ufl.TestFunction(V)
    ones = fem.Function(V)
    ones.x.array[:] = 1.0
    lumped_form = fem.form(ufl.inner(coefficient * ones, test_function) * measure)
    lumped_vec = assemble_vector(lumped_form)
    lumped = lumped_vec.array.copy()
    lumped_vec.destroy()
    return lumped


def assemble_lumped_mass(V, density=1.0, measure=ufl.dx) -> np.ndarray:
    """Assemble a lumped mass vector for a scalar or vector space."""

    return assemble_lumped_operator(V, coefficient=density, measure=measure)


def inverse_diagonal(diagonal: np.ndarray) -> np.ndarray:
    """Return a safe inverse for a diagonal vector."""

    safe = diagonal.copy()
    safe[safe <= 0.0] = np.inf
    return 1.0 / safe
