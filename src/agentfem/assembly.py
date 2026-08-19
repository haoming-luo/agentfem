"""Assembly helpers for standard finite-element workflows."""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc

import dolfinx.fem.petsc as fem_petsc


def make_form(ufl_form):
    """Compile a UFL form for assembly."""

    return fem.form(ufl_form)


def assemble_vector(form):
    """Assemble a vector and accumulate ghost contributions to owned entries."""

    vector = fem_petsc.assemble_vector(form)
    vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return vector


def assemble_matrix(form, bcs=None):
    """Assemble a matrix and apply optional strong Dirichlet BC structure."""

    matrix = fem_petsc.assemble_matrix(form, bcs=[] if bcs is None else bcs)
    matrix.assemble()
    return matrix


def assemble_lumped_operator(V, coefficient=1.0, measure=ufl.dx) -> np.ndarray:
    """Assemble a diagonal/lumped operator vector on ``V``.

    ``V`` is the DOLFINx function space that owns the degrees of freedom. The
    coefficient may be a Python scalar, ``fem.Constant``, ``fem.Function``, or
    any compatible UFL expression.
    """

    test_function = ufl.TestFunction(V)
    ones = fem.Function(V)
    ones.x.array[:] = 1.0
    lumped_form = fem.form(ufl.inner(coefficient * ones, test_function) * measure)
    lumped_vec = assemble_vector(lumped_form)
    lumped = lumped_vec.array.copy()
    lumped_vec.destroy()
    return lumped


def assemble_lumped_mass(V, density=1.0, measure=ufl.dx) -> np.ndarray:
    """Assemble a lumped mass vector for a scalar or vector space."""

    return assemble_lumped_operator(V, coefficient=density, measure=measure)


def inverse_diagonal(diagonal: np.ndarray) -> np.ndarray:
    """Return a safe inverse for a diagonal vector."""

    safe = diagonal.copy()
    safe[safe <= 0.0] = np.inf
    return 1.0 / safe
