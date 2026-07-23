"""Standard boundary-condition construction and application."""

from __future__ import annotations

from dolfinx import fem
from petsc4py import PETSc

import dolfinx.fem.petsc as fem_petsc

from . import dofs


def scalar_constant(domain, value=0.0):
    """Create a scalar PETSc-backed DOLFINx constant."""

    return fem.Constant(domain, PETSc.ScalarType(value))


def component_dirichlet_bc(V, component: int, marker, value=0.0):
    """Create a Dirichlet BC for one component of a vector function space.

    Returns ``(constant, bc)`` so callers can update ``constant.value`` during
    transient solves.
    """

    constant = scalar_constant(V.mesh, value)
    component_dofs = dofs.locate_component_dofs(V, component, marker)
    bc = fem.dirichletbc(constant, component_dofs, V.sub(component))
    return constant, bc


def apply_dirichlet_bcs(function, bcs) -> None:
    """Apply strong Dirichlet boundary conditions to a function vector."""

    fem_petsc.set_bc(function.x.petsc_vec, bcs)
    function.x.scatter_forward()
