"""Application-level unknown fields.

This layer bundles the finite-element space, solution field, trial function,
and test function so beginner workflows can talk about displacement or
temperature instead of V/u/du/v bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from . import spaces


@dataclass(frozen=True)
class UnknownField:
    """Finite-element unknown bundle for application-level workflows."""

    name: str
    space: object
    value: object
    trial: object
    test: object
    kind: str = "unknown"

    @property
    def function_space(self):
        """Compatibility alias for the underlying function space."""

        return self.space

    def summary(self) -> dict[str, object]:
        """Return an inspectable field summary."""

        element = self.space.ufl_element()
        return {
            "name": self.name,
            "kind": self.kind,
            "element": str(element),
            "value_name": getattr(self.value, "name", self.name),
        }

    def assign_from(self, other) -> None:
        """Copy values from another unknown field or DOLFINx function."""

        source = other.value if hasattr(other, "value") else other
        self.value.x.array[:] = source.x.array
        self.value.x.scatter_forward()

    def max_value(self) -> float:
        """Return the distributed maximum value of the field."""

        local = float(np.max(self.value.x.array))
        return self.value.function_space.mesh.comm.allreduce(local, op=MPI.MAX)

    def max_abs(self) -> float:
        """Return the distributed maximum absolute value of the field."""

        local = float(np.max(np.abs(self.value.x.array)))
        return self.value.function_space.mesh.comm.allreduce(local, op=MPI.MAX)


def scalar_unknown(domain, *, name: str = "Unknown", degree: int = 1, value=0.0) -> UnknownField:
    """Create a scalar finite-element unknown."""

    V = spaces.scalar_space(domain, degree=degree)
    return UnknownField(
        name=name,
        kind="scalar_unknown",
        space=V,
        value=spaces.named_function(V, name, value=value),
        trial=spaces.trial_function(V),
        test=spaces.test_function(V),
    )


def vector_unknown(
    domain,
    *,
    name: str = "Unknown",
    degree: int = 1,
    dim: int | None = None,
    value=0.0,
) -> UnknownField:
    """Create a vector finite-element unknown."""

    V = spaces.vector_space(domain, degree=degree, dim=dim)
    return UnknownField(
        name=name,
        kind="vector_unknown",
        space=V,
        value=spaces.named_function(V, name, value=value),
        trial=spaces.trial_function(V),
        test=spaces.test_function(V),
    )


def displacement(domain, *, degree: int = 1, dim: int | None = None, value=0.0) -> UnknownField:
    """Create a displacement unknown for mechanics workflows."""

    field = vector_unknown(domain, name="Displacement", degree=degree, dim=dim, value=value)
    return UnknownField(
        name=field.name,
        kind="displacement",
        space=field.space,
        value=field.value,
        trial=field.trial,
        test=field.test,
    )


def temperature(domain, *, degree: int = 1, value=0.0) -> UnknownField:
    """Create a temperature unknown for heat-transfer workflows."""

    field = scalar_unknown(domain, name="Temperature", degree=degree, value=value)
    return UnknownField(
        name=field.name,
        kind="temperature",
        space=field.space,
        value=field.value,
        trial=field.trial,
        test=field.test,
    )
