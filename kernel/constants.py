"""Internal constant helpers using AgentFEM scalar-type policy."""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from petsc4py import PETSc


def constant(domain_or_field, value):
    """Create a scalar or vector DOLFINx constant."""

    domain = domain_of(domain_or_field)
    if np.isscalar(value):
        return scalar_constant(domain, value)
    return vector_constant(domain, value)


def scalar_constant(domain_or_field, value=0.0):
    """Create a scalar constant using the active PETSc scalar type."""

    return fem.Constant(domain_of(domain_or_field), PETSc.ScalarType(value))


def scalar_value(value):
    """Convert a scalar to the active PETSc scalar type."""

    return PETSc.ScalarType(value)


def vector_constant(domain_or_field, values):
    """Create a vector constant using the active PETSc scalar type."""

    data = np.asarray(values, dtype=PETSc.ScalarType)
    return fem.Constant(domain_of(domain_or_field), data)


def domain_of(domain_or_field):
    """Return the mesh/domain from a domain, region, function, or unknown field."""

    if hasattr(domain_or_field, "domain"):
        return domain_or_field.domain
    if hasattr(domain_or_field, "value"):
        return domain_or_field.value.function_space.mesh
    if hasattr(domain_or_field, "space"):
        return domain_or_field.space.mesh
    if hasattr(domain_or_field, "function_space"):
        return domain_or_field.function_space.mesh
    if hasattr(domain_or_field, "mesh"):
        return domain_or_field.mesh
    return domain_or_field
