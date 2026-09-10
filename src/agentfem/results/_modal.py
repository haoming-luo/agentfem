"""Private Result assembly for structural modal analysis."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ..provenance import collective_call, collective_scientific_input_manifest
from .core import SimulationResult
from .lifecycle import complete_result


def from_modal_step(step, *, output=None, strict_output: bool = False):
    """Publish modal fields, frequencies, and invariant-subspace evidence."""

    target = getattr(step.target, "value", step.target)
    comm = target.function_space.mesh.comm

    def require_local_solution():
        if (
            step.eigenvalues is None
            or step.last_solve_info is None
            or not step.last_solve_info.converged
            or not step.mode_shapes
        ):
            raise RuntimeError("Modal result assembly requires a completed solve.")
        return tuple(step.mode_shapes)

    modes = collective_call(
        require_local_solution,
        comm=comm,
        label="Modal result readiness",
    )
    mode_counts = tuple(comm.allgather(len(modes)))
    if any(count != mode_counts[0] for count in mode_counts[1:]):
        raise RuntimeError(
            f"Modal result mode count differs across MPI ranks: {mode_counts}."
        )
    manifest = collective_scientific_input_manifest(
        step.scientific_inputs(),
        comm=comm,
        label="modal_result_inputs",
        require_nonempty=True,
    )
    if not manifest["complete"]:
        missing = ", ".join(
            str(item["path"]) for item in manifest["missing"][:5]
        )
        raise ValueError(
            "Modal result publication cannot freeze a complete scientific-"
            f"input identity: {missing or 'unknown'}."
        )
    result = SimulationResult(
        name=step.name,
        scientific_inputs=deepcopy(manifest["record"]),
        metadata={"scientific_input_retention": "frozen_identity_snapshot"},
    )
    result.add_quantities(
        {
            "eigenvalues": step.eigenvalues,
            "angular_frequencies": np.sqrt(step.eigenvalues),
            "frequencies": np.sqrt(step.eigenvalues) / (2.0 * np.pi),
            "residual_norms": np.asarray(step.last_solve_info.residual_norms),
            "mass_orthogonality_error": (
                step.last_solve_info.mass_orthogonality_error
            ),
            "stiffness_diagonalization_error": (
                step.last_solve_info.stiffness_diagonalization_error
            ),
        },
        units={
            "eigenvalues": "rad^2/s^2",
            "angular_frequencies": "rad/s",
            "frequencies": "Hz",
        },
        kind="modal",
    )
    cluster_by_mode = {
        int(mode_index): cluster_index
        for cluster_index, cluster in enumerate(
            step.last_solve_info.eigenvalue_clusters,
            start=1,
        )
        for mode_index in cluster["indices"]
    }
    for index, mode in enumerate(modes, start=1):
        result.add_field(
            f"Mode_{index}",
            mode,
            unit=None,
            description=(
                "Mass-normalized eigenvector; its amplitude is a normalization "
                "coordinate rather than a physical displacement."
            ),
            processing={
                "method": "generalized_hermitian_eigenproblem",
                "normalization": "mass",
                "orientation": "largest_global_component_positive",
                "eigenvalue_cluster": cluster_by_mode[index],
                "comparison_object": next(
                    cluster["comparison_object"]
                    for cluster in step.last_solve_info.eigenvalue_clusters
                    if index in cluster["indices"]
                ),
                "postprocessed": False,
            },
        )
    result.metadata["problem"] = step.summary()
    result.metadata["solve"] = step.last_solve_info.as_dict()
    completed = complete_result(
        step,
        result,
        output=output,
        strict_output=strict_output,
    )
    completed.collective_manifest(comm, include_histories=True)
    return completed


__all__ = ()
