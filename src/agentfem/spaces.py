"""Function-space construction helpers."""

from __future__ import annotations

import basix.ufl
import ufl
from dolfinx import fem


def lagrange_space(domain, degree: int = 1):
    """Create a scalar Lagrange function space."""

    domain = _domain(domain)
    return fem.functionspace(domain, ("Lagrange", degree))


def scalar_space(domain, degree: int = 1):
    """Create a scalar Lagrange function space."""

    return lagrange_space(domain, degree=degree)


def vector_lagrange_space(domain, degree: int = 1, dim: int | None = None):
    """Create a vector Lagrange function space."""

    domain = _domain(domain)
    value_dim = domain.geometry.dim if dim is None else dim
    return fem.functionspace(domain, ("Lagrange", degree, (value_dim,)))


def vector_space(domain, degree: int = 1, dim: int | None = None):
    """Create a vector Lagrange function space."""

    return vector_lagrange_space(domain, degree=degree, dim=dim)


def velocity_pressure_space(
    domain,
    *,
    velocity_degree: int = 2,
    pressure_degree: int = 1,
):
    """Create a Taylor--Hood velocity/pressure mixed space.

    The default ``P2/P1`` pair is the standard conforming choice for
    incompressible Stokes and Navier--Stokes flow. The interpolation policy is
    explicit because a mesh alone never determines a stable mixed method.
    """

    domain = _domain(domain)
    if int(velocity_degree) < 2:
        raise ValueError("velocity_degree must be at least two for Taylor--Hood.")
    if int(pressure_degree) < 1:
        raise ValueError("pressure_degree must be at least one.")
    if int(velocity_degree) <= int(pressure_degree):
        raise ValueError(
            "Taylor--Hood requires velocity_degree greater than pressure_degree."
        )
    cell = domain.basix_cell()
    velocity_element = basix.ufl.element(
        "Lagrange",
        cell,
        int(velocity_degree),
        shape=(int(domain.geometry.dim),),
    )
    pressure_element = basix.ufl.element(
        "Lagrange",
        cell,
        int(pressure_degree),
    )
    return fem.functionspace(
        domain,
        basix.ufl.mixed_element((velocity_element, pressure_element)),
    )


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

    domain = _domain(domain)
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


def _domain(mesh_or_domain):
    """Accept a DOLFINx mesh or an AgentFEM imported-mesh facade."""

    return getattr(mesh_or_domain, "domain", mesh_or_domain)
