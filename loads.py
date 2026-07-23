"""Standard load and time-dependent boundary helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import ufl
from petsc4py import PETSc

from . import boundary
from . import forms


@dataclass(frozen=True)
class TimeFunction:
    """Named scalar function of time."""

    name: str
    value: Callable[[float], float]

    def __call__(self, t: float) -> float:
        return float(self.value(t))


@dataclass(frozen=True)
class TimeDependentDirichlet:
    """A Dirichlet boundary condition driven by a scalar time function."""

    constant: object
    bc: object
    time_function: TimeFunction

    def update(self, t: float) -> float:
        value = self.time_function(t)
        self.constant.value = PETSc.ScalarType(value)
        return value


def time_dependent_component_dirichlet(V, component: int, marker, time_function: TimeFunction):
    """Create a time-dependent Dirichlet BC on one vector component."""

    constant, bc = boundary.component_dirichlet_bc(V, component, marker, value=0.0)
    return TimeDependentDirichlet(constant=constant, bc=bc, time_function=time_function)


def apply_dirichlet_bcs(function, bcs) -> None:
    """Apply strong Dirichlet boundary conditions to a function vector."""

    boundary.apply_dirichlet_bcs(function, bcs)


def constant_time_function(value: float, name: str = "constant") -> TimeFunction:
    """Represent a constant value with the same interface as transient data."""

    return TimeFunction(name=name, value=lambda _t: value)


@dataclass(frozen=True)
class BodyLoad:
    """Domain source/body-force term for a weak form."""

    value: object
    measure: object = ufl.dx
    name: str = "body_load"

    def form(self, test_function):
        return forms.body_force_virtual_work(self.value, test_function, self.measure)


@dataclass(frozen=True)
class BoundaryLoad:
    """Boundary flux/traction term for a weak form."""

    value: object
    measure: object
    name: str = "boundary_load"

    def form(self, test_function):
        return forms.boundary_flux_virtual_work(self.value, test_function, self.measure)


@dataclass(frozen=True)
class NeumannLoad:
    """Natural boundary condition applied through the weak-form right hand side."""

    value: object
    measure: object
    name: str = "neumann_load"

    def form(self, test_function):
        return forms.boundary_flux_virtual_work(self.value, test_function, self.measure)


@dataclass(frozen=True)
class LoadSet:
    """Ordered collection of weak-form load terms."""

    loads: tuple[object, ...]

    @classmethod
    def create(cls, *loads):
        return cls(tuple(loads))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(load.name for load in self.loads)

    def form(self, test_function):
        total = None
        for load in self.loads:
            term = load.form(test_function)
            total = term if total is None else total + term
        if total is None:
            raise ValueError("LoadSet.form requires at least one load.")
        return total


def body_force_form(force, test_function):
    """Create a body-force virtual-work form."""

    return forms.body_force_virtual_work(force, test_function, ufl.dx)


def boundary_traction_form(traction, test_function, ds_measure):
    """Create a boundary-traction virtual-work form."""

    return forms.boundary_flux_virtual_work(traction, test_function, ds_measure)
