"""FEniCSx/SLEPc execution backend for structural modal analysis.

This module owns distributed matrix reduction, eigensolver configuration, and
the lifetime of PETSc/SLEPc resources.  The public mechanics procedure remains
responsible for scientific intent and the result layer remains responsible for
publishing evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from dolfinx import fem
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

from .._modal_fem import (
    orient_mode_deterministically,
    require_symmetric_operator,
)
from ..dependencies import require


@dataclass(frozen=True)
class ModalBackendCandidate:
    """One real eigenpair materialized by the numerical backend."""

    eigenvalue: float
    residual_norm: float
    mode_shape: object
    orientation_anchor_dof: int


@dataclass(frozen=True)
class ModalBackendResult:
    """Raw eigensolver output and operator evidence for modal selection."""

    candidates: tuple[ModalBackendCandidate, ...]
    converged_eigenpairs: int
    constrained_dofs: int
    free_dofs: int
    eigensolver: str
    mass_gram: np.ndarray
    stiffness_gram: np.ndarray
    operator_symmetry_relative_tolerance: float
    stiffness_symmetry_absolute_tolerance: float
    mass_symmetry_absolute_tolerance: float


def solve_modal_eigenproblem(
    *,
    target,
    stiffness,
    mass,
    modes: int,
    bcs: tuple[object, ...],
    target_frequency: float | None,
    tolerance: float,
    maximum_iterations: int,
) -> ModalBackendResult:
    """Solve one constrained generalized Hermitian eigenproblem.

    Every PETSc/SLEPc object created here is destroyed before this function
    returns or raises.  DOLFINx fields in the returned result own only copied
    modal coefficients and therefore remain live after backend teardown.
    """

    SLEPc = require(
        "slepc4py.SLEPc",
        extra="modal",
        capability="distributed structural modal analysis",
    )
    solution = getattr(target, "value", target)
    V = solution.function_space
    comm = V.mesh.comm

    stiffness_matrix = None
    mass_matrix = None
    free_is = None
    reduced_stiffness = None
    reduced_mass = None
    eps = None
    reduced_vector = None
    try:
        stiffness_matrix = stiffness.assemble_matrix(bcs=None)
        mass_matrix = mass.assemble_matrix(bcs=None)

        block_size = int(V.dofmap.index_map_bs)
        owned_blocks = int(V.dofmap.index_map.size_local)
        owned_scalar = owned_blocks * block_size
        constrained_local = []
        for bc in bcs:
            indices, first_ghost = bc.dof_indices()
            constrained_local.extend(
                np.asarray(indices[:first_ghost], dtype=np.int64)
            )
        constrained_local = np.unique(
            np.asarray(constrained_local, dtype=np.int64)
        )
        constrained_local = constrained_local[constrained_local < owned_scalar]
        free_mask = np.ones(owned_scalar, dtype=bool)
        free_mask[constrained_local] = False
        free_local = np.flatnonzero(free_mask).astype(np.int32)

        local_blocks = free_local // block_size
        components = free_local % block_size
        global_blocks = V.dofmap.index_map.local_to_global(local_blocks)
        free_global = (
            np.asarray(global_blocks, dtype=PETSc.IntType) * block_size
            + components.astype(PETSc.IntType)
        )
        free_count = int(comm.allreduce(free_local.size, op=MPI.SUM))
        constrained_count = int(
            comm.allreduce(constrained_local.size, op=MPI.SUM)
        )
        if free_count <= modes:
            raise ValueError(
                f"Modal analysis has {free_count} free dofs but requests "
                f"{modes} modes."
            )

        free_is = PETSc.IS().createGeneral(free_global, comm=comm)
        reduced_stiffness = stiffness_matrix.createSubMatrix(free_is, free_is)
        reduced_mass = mass_matrix.createSubMatrix(free_is, free_is)

        relative_symmetry_tolerance = max(
            100.0 * np.finfo(float).eps,
            min(1.0e-8, 10.0 * tolerance),
        )
        stiffness_symmetry_tolerance = require_symmetric_operator(
            reduced_stiffness,
            name="stiffness",
            relative_tolerance=relative_symmetry_tolerance,
        )
        mass_symmetry_tolerance = require_symmetric_operator(
            reduced_mass,
            name="mass",
            relative_tolerance=relative_symmetry_tolerance,
        )

        eps = SLEPc.EPS().create(comm)
        eps.setOperators(reduced_stiffness, reduced_mass)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        requested = min(free_count - 1, modes + min(8, free_count - modes - 1))
        eps.setDimensions(requested)
        eps.setTolerances(tol=tolerance, max_it=maximum_iterations)
        target_eigenvalue = None
        if target_frequency is None:
            eps.setTarget(0.0)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.getST().setType(SLEPc.ST.Type.SINVERT)
        else:
            target_eigenvalue = (2.0 * np.pi * target_frequency) ** 2
            eps.setTarget(target_eigenvalue)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
            eps.getST().setType(SLEPc.ST.Type.SINVERT)
        eps.setFromOptions()
        eps.solve()

        converged = int(eps.getConverged())
        candidate_records = []
        reduced_vector = reduced_stiffness.createVecRight()
        for index in range(converged):
            raw_eigenvalue = eps.getEigenvalue(index)
            if abs(float(np.imag(raw_eigenvalue))) > tolerance:
                continue
            eigenvalue = float(np.real(raw_eigenvalue))
            eps.getEigenvector(index, reduced_vector)
            local_values = np.asarray(reduced_vector.array_r)
            if local_values.size != free_local.size:
                raise RuntimeError(
                    "Distributed modal subspace layout does not match the "
                    "free-dof map."
                )
            mode = fem.Function(V, name=f"ModalCandidate_{len(candidate_records) + 1}")
            mode.x.array[free_local] = np.real(local_values)
            mode.x.scatter_forward()
            candidate_records.append(
                ModalBackendCandidate(
                    eigenvalue=eigenvalue,
                    residual_norm=float(
                        eps.computeError(index, SLEPc.EPS.ErrorType.RELATIVE)
                    ),
                    mode_shape=mode,
                    orientation_anchor_dof=orient_mode_deterministically(
                        mode,
                        free_local,
                        free_global,
                    ),
                )
            )
        mass_gram, stiffness_gram = _operator_gram_matrices(
            stiffness_matrix,
            mass_matrix,
            tuple(item.mode_shape for item in candidate_records),
        )
        return ModalBackendResult(
            candidates=tuple(candidate_records),
            converged_eigenpairs=converged,
            constrained_dofs=constrained_count,
            free_dofs=free_count,
            eigensolver=str(eps.getType()),
            mass_gram=mass_gram,
            stiffness_gram=stiffness_gram,
            operator_symmetry_relative_tolerance=relative_symmetry_tolerance,
            stiffness_symmetry_absolute_tolerance=stiffness_symmetry_tolerance,
            mass_symmetry_absolute_tolerance=mass_symmetry_tolerance,
        )
    finally:
        for resource in (
            reduced_vector,
            eps,
            reduced_stiffness,
            reduced_mass,
            free_is,
            stiffness_matrix,
            mass_matrix,
        ):
            if resource is not None:
                resource.destroy()


def _operator_gram_matrices(stiffness, mass, modes):
    """Return operator Gram matrices before backend resources are released."""

    count = len(modes)
    if count == 0:
        empty = np.empty((0, 0), dtype=float)
        return empty, empty.copy()
    mass_gram = np.empty((count, count), dtype=float)
    stiffness_gram = np.empty((count, count), dtype=float)
    mass_action = mass.createVecLeft()
    stiffness_action = stiffness.createVecLeft()
    try:
        for column, right in enumerate(modes):
            mass.mult(right.x.petsc_vec, mass_action)
            stiffness.mult(right.x.petsc_vec, stiffness_action)
            for row, left in enumerate(modes):
                mass_gram[row, column] = float(
                    np.real(left.x.petsc_vec.dot(mass_action))
                )
                stiffness_gram[row, column] = float(
                    np.real(left.x.petsc_vec.dot(stiffness_action))
                )
    finally:
        mass_action.destroy()
        stiffness_action.destroy()
    return mass_gram, stiffness_gram


__all__ = [
    "ModalBackendCandidate",
    "ModalBackendResult",
    "solve_modal_eigenproblem",
]
