"""Private Result assembly for structural modal analysis."""

from __future__ import annotations

import numpy as np

from .core import SimulationResult
from .lifecycle import complete_result


def from_modal_step(step, *, output=None, strict_output: bool = False):
    """Publish modal fields, frequencies, and invariant-subspace evidence."""

    if step.eigenvalues is None or step.last_solve_info is None:
        raise RuntimeError("Modal result assembly requires a completed solve.")
    modes = step.mode_shapes
    result = SimulationResult(name=step.name)
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
    return complete_result(
        step,
        result,
        output=output,
        strict_output=strict_output,
    )


__all__ = ()
