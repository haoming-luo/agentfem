"""Structural modal solution procedure over assembled stiffness and mass."""

from __future__ import annotations

from dataclasses import dataclass, field
from operator import index as integer_index

import numpy as np

from .._modal import cluster_summaries, selected_clusters_are_complete
from ..backends._modal import ModalBackendResult, solve_modal_eigenproblem
from ..dynamics import ModalSolveInfo


def _positive_integer(value, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        selected = integer_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if selected <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(selected)


def _collect_modal_bcs(*, constraints=(), bcs=()) -> tuple[object, ...]:
    """Lower only strong Dirichlet assets for the current modal backend."""

    selected = list(bcs or ())
    for item in constraints or ():
        if hasattr(item, "bcs"):
            selected.extend(item.bcs)
        elif hasattr(item, "bc"):
            selected.append(item.bc)
        else:
            raise TypeError(
                "AFM-CONSTRAINT-PROCEDURE-001: modal analysis received "
                f"{type(item).__name__}, which is not a strong Dirichlet "
                "constraint. Run model.check() and select a modal-compatible "
                "exact affine/MPC backend for non-Dirichlet kinematic relations."
            )
    return tuple(selected)


@dataclass
class ModalAnalysisStep:
    """Constrained linear modes from ``K phi = lambda M phi``.

    The Step owns modal intent and solved scientific state.  Distributed
    eigensolver resources belong to the backend execution scope, while result
    publication belongs to :mod:`agentfem.results`.
    """

    name: str
    target: object
    stiffness: object
    mass: object
    modes: int
    study: object | None = None
    constraints: tuple[object, ...] = ()
    bcs: tuple[object, ...] = ()
    target_frequency: float | None = None
    tolerance: float = 1.0e-9
    maximum_iterations: int = 1000
    rigid_mode_tolerance: float = 1.0e-10
    procedure: object | None = None
    eigenvalues: np.ndarray | None = field(default=None, init=False)
    mode_shapes: tuple[object, ...] = field(default=(), init=False)
    last_solve_info: ModalSolveInfo | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.modes = _positive_integer(self.modes, name="modes")
        self.maximum_iterations = _positive_integer(
            self.maximum_iterations,
            name="maximum_iterations",
        )
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("Modal tolerance must be finite and positive.")
        if (
            not np.isfinite(self.rigid_mode_tolerance)
            or self.rigid_mode_tolerance < 0.0
        ):
            raise ValueError("rigid_mode_tolerance must be nonnegative.")
        if self.target_frequency is not None and (
            not np.isfinite(self.target_frequency) or self.target_frequency < 0.0
        ):
            raise ValueError("target_frequency must be finite and nonnegative.")

    def solve(self):
        """Execute the distributed eigensolve and retain modal evidence."""

        selected_bcs = _collect_modal_bcs(
            constraints=self.constraints,
            bcs=self.bcs,
        )
        solved = solve_modal_eigenproblem(
            target=self.target,
            stiffness=self.stiffness,
            mass=self.mass,
            modes=int(self.modes),
            bcs=selected_bcs,
            target_frequency=self.target_frequency,
            tolerance=float(self.tolerance),
            maximum_iterations=int(self.maximum_iterations),
        )
        eigenvalues, mode_shapes, solve_info = _select_modal_solution(
            solved,
            modes=int(self.modes),
            target_frequency=self.target_frequency,
            tolerance=float(self.tolerance),
            rigid_mode_tolerance=float(self.rigid_mode_tolerance),
        )
        self.eigenvalues = eigenvalues
        self.mode_shapes = mode_shapes
        self.last_solve_info = solve_info
        if not solve_info.converged:
            raise RuntimeError(
                f"Modal solve accepted {solve_info.accepted_modes} of "
                f"{solve_info.requested_modes} requested modes."
            )
        return self.mode_shapes

    def solve_result(self, *, output=None, strict_output: bool = False):
        """Solve and publish the modal fields and verification evidence."""

        from ..results._modal import from_modal_step

        self.solve()
        return from_modal_step(
            self,
            output=output,
            strict_output=strict_output,
        )

    def summary(self) -> dict[str, object]:
        return {
            "kind": "modal_analysis_step",
            "name": self.name,
            "requested_modes": int(self.modes),
            "target_frequency": self.target_frequency,
            "constraints": len(
                _collect_modal_bcs(constraints=self.constraints, bcs=self.bcs)
            ),
            "stiffness": self.stiffness.summary(),
            "mass": self.mass.summary(),
            "procedure": None if self.procedure is None else self.procedure.summary(),
            "last_solve": (
                None
                if self.last_solve_info is None
                else self.last_solve_info.as_dict()
            ),
        }


def _select_modal_solution(
    backend: ModalBackendResult,
    *,
    modes: int,
    target_frequency: float | None,
    tolerance: float,
    rigid_mode_tolerance: float,
) -> tuple[np.ndarray, tuple[object, ...], ModalSolveInfo]:
    """Apply physical mode filtering and selection to raw backend eigenpairs."""

    indexed = list(enumerate(backend.candidates))
    scale = max(
        1.0,
        max((abs(item.eigenvalue) for _, item in indexed), default=1.0),
    )
    eligible = [
        (index, item)
        for index, item in indexed
        if item.eigenvalue > rigid_mode_tolerance * scale
    ]
    target_eigenvalue = (
        None
        if target_frequency is None
        else (2.0 * np.pi * target_frequency) ** 2
    )
    if target_eigenvalue is None:
        selected = sorted(eligible, key=lambda pair: pair[1].eigenvalue)[:modes]
    else:
        selected = sorted(
            eligible,
            key=lambda pair: abs(pair[1].eigenvalue - target_eigenvalue),
        )[:modes]
        selected.sort(key=lambda pair: pair[1].eigenvalue)

    ordered = sorted(eligible, key=lambda pair: pair[1].eigenvalue)
    selected_backend_indices = {index for index, _ in selected}
    selected_positions = tuple(
        position
        for position, (index, _) in enumerate(ordered)
        if index in selected_backend_indices
    )
    cluster_relative_tolerance = max(1.0e-8, 100.0 * tolerance)
    selected_clusters_complete = bool(ordered) and selected_clusters_are_complete(
        [item.eigenvalue for _, item in ordered],
        selected_positions,
        relative_tolerance=cluster_relative_tolerance,
    )

    eigenvalues = np.asarray(
        [item.eigenvalue for _, item in selected],
        dtype=float,
    )
    mode_shapes = tuple(item.mode_shape for _, item in selected)
    for index, mode_shape in enumerate(mode_shapes, start=1):
        mode_shape.name = f"Mode_{index}"
    selected_indices = [index for index, _ in selected]
    mass_error, stiffness_error = _orthogonality_errors(
        backend.mass_gram,
        backend.stiffness_gram,
        selected_indices,
        eigenvalues,
    )
    orthogonality_tolerance = max(1.0e-7, 100.0 * tolerance)
    solve_info = ModalSolveInfo(
        converged_eigenpairs=backend.converged_eigenpairs,
        requested_modes=modes,
        accepted_modes=len(mode_shapes),
        constrained_dofs=backend.constrained_dofs,
        free_dofs=backend.free_dofs,
        residual_norms=tuple(item.residual_norm for _, item in selected),
        eigensolver=backend.eigensolver,
        target_frequency=target_frequency,
        mass_orthogonality_error=mass_error,
        stiffness_diagonalization_error=stiffness_error,
        orthogonality_tolerance=orthogonality_tolerance,
        orientation_anchor_dofs=tuple(
            item.orientation_anchor_dof for _, item in selected
        ),
        operator_symmetry_relative_tolerance=(
            backend.operator_symmetry_relative_tolerance
        ),
        stiffness_symmetry_absolute_tolerance=(
            backend.stiffness_symmetry_absolute_tolerance
        ),
        mass_symmetry_absolute_tolerance=backend.mass_symmetry_absolute_tolerance,
        stiffness_symmetric=True,
        mass_symmetric=True,
        cluster_relative_tolerance=cluster_relative_tolerance,
        eigenvalue_clusters=(
            cluster_summaries(
                eigenvalues,
                relative_tolerance=cluster_relative_tolerance,
            )
            if eigenvalues.size
            else ()
        ),
        selected_clusters_complete=selected_clusters_complete,
    )
    return eigenvalues, mode_shapes, solve_info


def _orthogonality_errors(
    mass_gram,
    stiffness_gram,
    selected_indices,
    eigenvalues,
) -> tuple[float, float]:
    count = len(selected_indices)
    if count == 0:
        return float("inf"), float("inf")
    selector = np.ix_(selected_indices, selected_indices)
    selected_mass = np.asarray(mass_gram)[selector]
    selected_stiffness = np.asarray(stiffness_gram)[selector]
    mass_error = float(np.max(np.abs(selected_mass - np.eye(count))))
    expected_stiffness = np.diag(np.asarray(eigenvalues, dtype=float))
    stiffness_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    stiffness_error = float(
        np.max(np.abs(selected_stiffness - expected_stiffness)) / stiffness_scale
    )
    return mass_error, stiffness_error


__all__ = ["ModalAnalysisStep"]
