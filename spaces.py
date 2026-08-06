"""Function-space construction helpers."""

from __future__ import annotations

import basix.ufl
import ufl
from dolfinx import fem


def lagrange_space(domain, degree: int = 1):
    """Create a scalar Lagrange function space."""

    return fem.functionspace(domain, ("Lagrange", degree))


def scalar_space(domain, degree: int = 1):
    """Create a scalar Lagrange function space."""

    return lagrange_space(domain, degree=degree)


def vector_lagrange_space(domain, degree: int = 1, dim: int | None = None):
    """Create a vector Lagrange function space."""

    value_dim = domain.geometry.dim if dim is None else dim
    return fem.functionspace(domain, ("Lagrange", degree, (value_dim,)))


def vector_space(domain, degree: int = 1, dim: int | None = None):
    """Create a vector Lagrange function space."""

    return vector_lagrange_space(domain, degree=degree, dim=dim)


def displacement_pressure_space(
    domain,
    *,
    displacement_degree: int = 2,
    pressure_degree: int = 0,
):
    """Create the mixed ``H1`` displacement / discontinuous-pressure space.

    The default ``P2/DG0`` pair provides one constant pressure unknown per
    cell.  This is AgentFEM's explicit mixed-field analogue for constant-
    pressure hybrid solid formulations such as Abaqus ``C3D10H``; the mesh
    topology alone never selects this formulation implicitly.
    """

    if int(displacement_degree) < 1:
        raise ValueError("displacement_degree must be at least one.")
    if int(pressure_degree) < 0:
        raise ValueError("pressure_degree must be non-negative.")
    cell = domain.basix_cell()
    displacement_element = basix.ufl.element(
        "Lagrange",
        cell,
        int(displacement_degree),
        shape=(int(domain.geometry.dim),),
    )
    pressure_element = basix.ufl.element(
        "DG",
        cell,
        int(pressure_degree),
    )
    return fem.functionspace(
        domain,
        basix.ufl.mixed_element((displacement_element, pressure_element)),
    )


def test_function(V):
    """Create a UFL test function for a function space."""

    return ufl.TestFunction(V)


def trial_function(V):
    """Create a UFL trial function for a function space."""

    return ufl.TrialFunction(V)


def named_function(V, name: str, value=0.0):
    """Create a named finite-element function and optionally initialize it."""

    function = fem.Function(V, name=name)
    function.x.array[:] = value
    function.x.scatter_forward()
    return function
