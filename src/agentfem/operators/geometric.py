# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Initial-stress geometric stiffness for conservative solid buckling."""

import ufl
from .core import from_ufl


def geometric_stiffness(target, stress, *, measure=ufl.dx, name="KG"):
    """Return KG with tension-positive stress (compression destabilizes).

    The stress belongs to a small-displacement reference equilibrium on the
    same mesh. Follower-load tangents and finite-deformation material tangents
    are not included by this initial-stress operator.
    """
    value = getattr(target, "value", target)
    space = value.function_space
    dimension = space.mesh.geometry.dim
    if value.ufl_shape != (dimension,) or space.mesh.topology.dim != dimension:
        raise ValueError(
            "AFM-BUCKLING-001: geometric stiffness requires a 2D/3D solid vector field."
        )
    if dimension not in (2, 3) or tuple(stress.ufl_shape) != (dimension, dimension):
        raise ValueError("AFM-BUCKLING-002: stress must match the solid dimension.")
    du, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    return from_ufl(
        ufl.inner(ufl.grad(v), ufl.dot(ufl.grad(du), stress)) * measure,
        name=name,
        kind="initial_stress_geometric_stiffness",
        family="elasticity",
    )
