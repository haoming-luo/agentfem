"""Distributed diagnostics for finite-element fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from dolfinx import fem
from mpi4py import MPI

from .kernel import dofs


def kinetic_energy(mass_lumped: np.ndarray, velocity: fem.Function) -> float:
    """Global kinetic energy from a lumped mass vector and velocity field."""

    local = 0.5 * float(np.sum(mass_lumped * dofs.owned_array(velocity) ** 2))
    return velocity.function_space.mesh.comm.allreduce(local, op=MPI.SUM)


def max_abs(function: fem.Function) -> float:
    """Global max absolute value of a finite-element field."""

    local = float(np.max(np.abs(function.x.array)))
    return function.function_space.mesh.comm.allreduce(local, op=MPI.MAX)


@dataclass(frozen=True)
class ScalarDiagnostic:
    """Named scalar diagnostic evaluated on demand."""

    name: str
    value: Callable[[], float]

    def evaluate(self) -> float:
        return float(self.value())


@dataclass(frozen=True)
class DiagnosticSet:
    """Ordered collection of scalar diagnostics."""

    diagnostics: tuple[ScalarDiagnostic, ...]

    @classmethod
    def create(cls, *diagnostics: ScalarDiagnostic):
        return cls(tuple(diagnostics))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(diagnostic.name for diagnostic in self.diagnostics)

    def evaluate(self) -> dict[str, float]:
        return {
            diagnostic.name: diagnostic.evaluate()
            for diagnostic in self.diagnostics
        }
