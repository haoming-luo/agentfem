# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Accepted displacement-controlled hyperelastic path measurements."""

from types import SimpleNamespace
import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from ..constitutive import hyperelasticity
from .._problem_fields import reaction_field
from ..results import reaction_resultant


def displacement_controlled_response(
    model, step, *, on, component=0, monitor=None, monitor_component=1
):
    """Return accepted factor, mean boundary displacement and reaction arrays.

    Restricted to ordinary strong-BC, single-material displacement hyperelastic
    paths without body/traction loads or boundary models. Reconstruct internal
    reactions from each saved state, never from the final state or stress
    extrapolation. Monitor displacement is a boundary average. No mutation of
    the solved state and no claim of limit-point continuation.
    """
    if (
        model.loads
        or model.boundary_models
        or model.eigenstrains
        or len(model.materials) != 1
    ):
        raise ValueError(
            "AFM-PATH-001: requires one hyperelastic material and displacement-only loading."
        )
    material = model.materials[0].item
    if not hyperelasticity.is_finite_strain_hyperelastic(material) or not hasattr(
        step, "value_path"
    ):
        raise ValueError(
            "AFM-PATH-002: requires an ordinary displacement hyperelastic step."
        )
    snapshots = tuple(step.snapshots)
    if not snapshots:
        raise ValueError("AFM-PATH-003: solve with saved accepted increments first.")
    dim = snapshots[0].solution.ufl_shape[0]
    if component not in range(dim) or monitor_component not in range(dim):
        raise ValueError("AFM-PATH-004: invalid displacement component.")
    monitor = on if monitor is None else monitor
    domain = snapshots[0].solution.function_space.mesh
    if on.domain is not domain or monitor.domain is not domain:
        raise ValueError(
            "AFM-PATH-006: measurement boundaries must belong to the solved mesh."
        )
    region = model.materials[0].region
    measure = ufl.dx if region is None else region.measure

    def average(u, boundary, axis):
        comm = u.function_space.mesh.comm
        size = comm.allreduce(
            fem.assemble_scalar(fem.form(1.0 * boundary.measure)), op=MPI.SUM
        )
        if size <= 0:
            raise ValueError("AFM-PATH-005: measurement boundary is empty.")
        return (
            comm.allreduce(
                fem.assemble_scalar(fem.form(u[axis] * boundary.measure)), op=MPI.SUM
            )
            / size
        )

    rows = []
    for snapshot in snapshots:
        u = snapshot.solution
        residual = hyperelasticity.internal_virtual_work(
            u, ufl.TestFunction(u.function_space), material, measure=measure
        )
        proxy = SimpleNamespace(
            reaction_field=lambda name="RF", residual=residual, u=u: reaction_field(
                residual, u, name=name
            )
        )
        rows.append(
            (
                snapshot.load_factor,
                average(u, on, component),
                reaction_resultant(proxy, on=on, component=component),
                average(u, monitor, monitor_component),
            )
        )
    data = np.asarray(rows)
    return {
        name: data[:, i]
        for i, name in enumerate(
            ("load_factor", "displacement", "reaction", "monitor_displacement")
        )
    }
