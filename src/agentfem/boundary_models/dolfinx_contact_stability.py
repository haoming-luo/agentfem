# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""DOLFINx lowering for explicit penalty-contact stability screening."""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from .contact_stability import (
    ContactStabilityEstimate,
    contact_penalty_local_row_sums,
    contact_stability_estimate_from_bound,
)
from .dolfinx_contact_trace import DolfinxContactTraceAdapter


def _maximum_penalty(value, point_count: int, *, name: str) -> float:
    selected = np.asarray(value, dtype=float)
    if selected.ndim == 0:
        maximum = float(selected)
    else:
        selected = selected.reshape(-1)
        if selected.shape != (point_count,):
            raise ValueError(f"{name} must be scalar or contain one value per point.")
        if point_count == 0:
            return -np.inf
        maximum = float(np.max(selected, initial=-np.inf))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError(f"{name} values must be finite and positive.")
    return maximum


def estimate_dolfinx_contact_stability(
    *,
    adapter: DolfinxContactTraceAdapter,
    lumped_mass,
    normal_penalty,
    tangential_penalty=None,
    friction_coefficient: float = 0.0,
    safety_factor: float = 0.8,
) -> ContactStabilityEstimate:
    """Estimate the contact contribution to explicit stability.

    The returned estimate is a conservative screening bound for the contact
    contribution. A complete explicit Procedure must add this spectral bound
    to its body/material spectral bound before selecting the time increment;
    taking the smaller of two independently derived limits is not sufficient.
    """

    if not isinstance(adapter, DolfinxContactTraceAdapter):
        raise TypeError("DOLFINx contact stability requires its trace adapter.")
    function_space = adapter.function_space
    block_size = int(function_space.dofmap.index_map_bs)
    if block_size not in {2, 3}:
        raise ValueError("Contact stability requires a blocked 2D or 3D vector space.")
    raw_mass = getattr(lumped_mass, "mass", lumped_mass)
    mass_space = getattr(lumped_mass, "function_space", None)
    if mass_space is not None and mass_space is not function_space:
        raise ValueError(
            "Lumped mass and contact trace must use the same function-space instance."
        )
    mass_compatibility = (
        "function_space_identity" if mass_space is function_space else "shape_only"
    )
    mass = np.asarray(raw_mass, dtype=float).reshape(-1)
    owned_size = int(function_space.dofmap.index_map.size_local) * block_size
    block_count = int(
        function_space.dofmap.index_map.size_local
        + function_space.dofmap.index_map.num_ghosts
    )
    complete_size = block_count * block_size
    if mass.shape == (owned_size,):
        mass_holder = fem.Function(function_space)
        mass_holder.x.array[:] = 0.0
        mass_holder.x.array[:owned_size] = mass
        mass_holder.x.scatter_forward()
        mass = mass_holder.x.array.copy()
    elif mass.shape != (complete_size,):
        raise ValueError(
            "Lumped mass must contain either every owned displacement dof or "
            "every local-and-ghost displacement dof."
        )
    local_rows = contact_penalty_local_row_sums(
        adapter.trace,
        mass.reshape((block_count, block_size)),
        dimension=block_size,
        normal_penalty=normal_penalty,
        tangential_penalty=tangential_penalty,
        friction_coefficient=friction_coefficient,
    )
    holder = fem.Function(function_space)
    holder.x.array[:] = np.asarray(local_rows).reshape(-1)
    vector = holder.x.petsc_vec
    vector.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )
    local_bound = float(np.max(vector.array[:owned_size], initial=0.0))
    communicator = adapter.communicator
    global_bound = float(communicator.allreduce(local_bound, op=MPI.MAX))
    point_count = int(communicator.allreduce(adapter.trace.point_count, op=MPI.SUM))
    if point_count < 1:
        raise ValueError("Contact stability requires at least one global trace point.")
    local_normal = _maximum_penalty(
        normal_penalty,
        adapter.trace.point_count,
        name="Normal contact penalty",
    )
    normal_maximum = float(communicator.allreduce(local_normal, op=MPI.MAX))
    if tangential_penalty is None:
        tangential_maximum = None
    else:
        local_tangential = _maximum_penalty(
            tangential_penalty,
            adapter.trace.point_count,
            name="Tangential contact penalty",
        )
        tangential_maximum = float(communicator.allreduce(local_tangential, op=MPI.MAX))
    return contact_stability_estimate_from_bound(
        global_bound,
        safety_factor=safety_factor,
        normal_penalty_maximum=normal_maximum,
        tangential_penalty_maximum=tangential_maximum,
        friction_coefficient=friction_coefficient,
        point_count=point_count,
        mass_compatibility=mass_compatibility,
    )


__all__ = [
    "estimate_dolfinx_contact_stability",
]
