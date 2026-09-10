"""FEniCSx/SLEPc execution backend for structural modal analysis.

This module owns distributed matrix reduction, eigensolver configuration, and
the lifetime of PETSc/SLEPc resources.  The public mechanics procedure remains
responsible for scientific intent and the result layer remains responsible for
publishing evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Callable, TypeVar

from dolfinx import fem
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

from .._modal_fem import (
    orient_mode_deterministically,
    require_symmetric_operator,
)
from ..dependencies import require


_T = TypeVar("_T")


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

        free_local, free_global, constrained_local = _run_rank_local_phase(
            comm,
            stage="free-dof layout",
            operation=lambda: _free_dof_layout(V, bcs),
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

        converged = _run_rank_local_phase(
            comm,
            stage="converged-eigenpair count",
            operation=lambda: int(eps.getConverged()),
        )
        _require_rank_consensus(
            comm,
            value=converged,
            stage="converged-eigenpair count",
        )
        candidate_records = []
        reduced_vector = reduced_stiffness.createVecRight()
        for index in range(converged):
            eigenvalue, is_real = _run_rank_local_phase(
                comm,
                stage=f"candidate {index} eigenvalue",
                operation=lambda candidate=index: _real_eigenvalue_candidate(
                    eps.getEigenvalue(candidate),
                    tolerance=tolerance,
                ),
            )
            _require_rank_consensus(
                comm,
                value=(eigenvalue, is_real),
                stage=f"candidate {index} eigenvalue decision",
            )
            if not is_real:
                continue
            _run_rank_local_phase(
                comm,
                stage=f"candidate {index} eigenvector extraction",
                operation=lambda candidate=index: eps.getEigenvector(
                    candidate,
                    reduced_vector,
                ),
            )
            local_values = _collect_reduced_mode_values(
                comm,
                reduced_vector,
                free_local=free_local,
                free_global=free_global,
                candidate=index,
            )
            residual_norm = _run_rank_local_phase(
                comm,
                stage=f"candidate {index} residual evaluation",
                operation=lambda candidate=index: float(
                    eps.computeError(candidate, SLEPc.EPS.ErrorType.RELATIVE)
                ),
            )
            mode = _run_rank_local_phase(
                comm,
                stage=f"candidate {index} field materialization",
                operation=lambda values=local_values: _materialize_mode_field(
                    V,
                    free_local=free_local,
                    local_values=values,
                    name=f"ModalCandidate_{len(candidate_records) + 1}",
                ),
            )
            mode.x.scatter_forward()
            candidate_records.append(
                ModalBackendCandidate(
                    eigenvalue=eigenvalue,
                    residual_norm=residual_norm,
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
        active_error = sys.exc_info()[1]
        cleanup_errors = []
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
                try:
                    resource.destroy()
                except BaseException as exc:  # pragma: no cover - PETSc teardown
                    cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        gathered_cleanup_errors = comm.allgather(tuple(cleanup_errors))
        failures = [
            f"rank {rank}: {error}"
            for rank, errors in enumerate(gathered_cleanup_errors)
            for error in errors
        ]
        if failures and active_error is None:
            raise RuntimeError(
                "Modal backend resource teardown failed collectively; "
                + "; ".join(failures)
            )


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


def _materialize_mode_field(V, *, free_local, local_values, name: str):
    """Create one local mode field before the collective ghost update."""

    mode = fem.Function(V, name=name)
    mode.x.array[free_local] = local_values
    return mode


def _run_rank_local_phase(
    comm,
    *,
    stage: str,
    operation: Callable[[], _T],
) -> _T:
    """Run local modal work and make its failure collective under MPI.

    PETSc and SLEPc calls that follow this helper are collective.  A Python or
    local-layout error on only one rank must therefore be reported to every
    rank before any peer enters the next collective operation.
    """

    if comm.size == 1:
        return operation()
    value = None
    local_error = None
    try:
        value = operation()
    except BaseException as exc:  # pragma: no cover - exercised under MPI
        local_error = f"{type(exc).__name__}: {exc}"
    errors = comm.allgather(local_error)
    failures = [
        (rank, error)
        for rank, error in enumerate(errors)
        if error is not None
    ]
    if failures:
        rank, error = failures[0]
        raise RuntimeError(
            f"Modal backend {stage} failed on rank {rank}: {error}"
        )
    return value


def _require_rank_consensus(comm, *, value, stage: str) -> None:
    """Reject rank-dependent control flow before the next collective call."""

    if comm.size == 1:
        return
    values = comm.allgather(value)
    if any(item != values[0] for item in values[1:]):
        raise RuntimeError(
            f"Modal backend {stage} differs across MPI ranks: {values}."
        )


def _free_dof_layout(V, bcs):
    """Build the owned reduced-space map without entering MPI collectives."""

    block_size = int(V.dofmap.index_map_bs)
    owned_blocks = int(V.dofmap.index_map.size_local)
    owned_scalar = owned_blocks * block_size
    constrained = []
    for bc in bcs:
        indices, first_ghost = bc.dof_indices()
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        first_ghost = int(first_ghost)
        if not 0 <= first_ghost <= indices.size:
            raise RuntimeError(
                "Dirichlet dof ownership boundary lies outside its index array."
            )
        constrained.extend(indices[:first_ghost])
    constrained_local = np.unique(np.asarray(constrained, dtype=np.int64))
    if np.any(constrained_local < 0):
        raise RuntimeError("Dirichlet dof indices must be nonnegative.")
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
    if free_global.shape != free_local.shape:
        raise RuntimeError(
            "Distributed modal free-dof map has inconsistent local and global "
            "dimensions."
        )
    return free_local, free_global, constrained_local


def _real_eigenvalue_candidate(raw_eigenvalue, *, tolerance: float):
    """Return a real eigenvalue and the rank-local candidate decision."""

    real = float(np.real(raw_eigenvalue))
    imaginary = float(np.imag(raw_eigenvalue))
    if not np.isfinite(real) or not np.isfinite(imaginary):
        raise RuntimeError("Modal eigensolver returned a nonfinite eigenvalue.")
    return real, abs(imaginary) <= tolerance


def _collect_reduced_mode_values(
    comm,
    vector,
    *,
    free_local,
    free_global,
    candidate: int,
) -> np.ndarray:
    """Copy and collectively validate one rank-local reduced eigenvector."""

    def collect() -> np.ndarray:
        values = np.asarray(vector.array_r)
        if values.ndim != 1 or values.size != free_local.size:
            raise RuntimeError(
                "distributed modal subspace layout does not match the "
                "free-dof map"
            )
        if np.asarray(free_global).shape != np.asarray(free_local).shape:
            raise RuntimeError(
                "local and global modal free-dof maps have different dimensions"
            )
        real_values = np.asarray(np.real(values), dtype=float)
        if not np.all(np.isfinite(real_values)):
            raise RuntimeError("modal eigenvector contains nonfinite values")
        return real_values.copy()

    return _run_rank_local_phase(
        comm,
        stage=f"candidate {candidate} reduced-vector layout",
        operation=collect,
    )


__all__ = [
    "ModalBackendCandidate",
    "ModalBackendResult",
    "solve_modal_eigenproblem",
]
