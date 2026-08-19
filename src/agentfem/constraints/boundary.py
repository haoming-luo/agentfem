"""Standard boundary-condition construction and application."""

from __future__ import annotations

from dolfinx import fem
from petsc4py import PETSc

import dolfinx.fem.petsc as fem_petsc

from ..kernel import dofs


def scalar_constant(domain, value=0.0):
    """Create a scalar PETSc-backed DOLFINx constant."""

    return fem.Constant(domain, PETSc.ScalarType(value))


def component_dirichlet_bc(V, component: int, marker=None, value=0.0, *, location=None):
    """Create a Dirichlet BC for one component of a vector function space.

    Returns ``(constant, bc)`` so callers can update ``constant.value`` during
    transient solves.
    """

    marker = _marker(marker, location)
    space = _space(V)
    constant = scalar_constant(space.mesh, value)
    component_dofs = dofs.locate_component_dofs(space, component, marker)
    bc = fem.dirichletbc(constant, component_dofs, space.sub(component))
    return constant, bc


def scalar_dirichlet_bc(V, marker=None, value=0.0, *, location=None):
    """Create a Dirichlet BC for a scalar function space.

    Returns ``(constant, bc)`` so callers can update ``constant.value`` during
    transient solves.
    """

    marker = _marker(marker, location)
    space = _space(V)
    constant = scalar_constant(space.mesh, value)
    scalar_dofs = dofs.locate_dofs(space, marker)
    bc = fem.dirichletbc(constant, scalar_dofs, space)
    return constant, bc


def apply_dirichlet_bcs(function, bcs) -> None:
    """Apply strong Dirichlet boundary conditions to a function vector."""

    fem_petsc.set_bc(function.x.petsc_vec, bcs)
    function.x.scatter_forward()


def _space(V):
    if hasattr(V, "function_space"):
        return V.function_space
    return V.space if hasattr(V, "space") else V


def _marker(marker, location):
    selected = location if location is not None else marker
    if selected is None:
        raise ValueError("A marker or location is required for a Dirichlet boundary.")
    # Preserve BoundaryRegion selection semantics so imported physical tags can
    # drive both strong constraints and weak loads.
    if hasattr(selected, "selection"):
        return selected
    return selected.marker if hasattr(selected, "marker") else selected
