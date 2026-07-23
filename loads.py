"""Standard load and time-dependent boundary helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import ufl
from petsc4py import PETSc

from .constraints import boundary
from . import forms
from .kernel import constants


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

    def summary(self) -> dict[str, object]:
        """Return a compact description for logs and agent inspection."""

        return {"name": self.name, "kind": "body_load"}


@dataclass(frozen=True)
class BoundaryLoad:
    """Boundary flux/traction term for a weak form."""

    value: object
    measure: object
    name: str = "boundary_load"
    location: object | None = None

    def form(self, test_function):
        return forms.boundary_flux_virtual_work(self.value, test_function, self.measure)

    def summary(self) -> dict[str, object]:
        """Return a compact description for logs and agent inspection."""

        return {
            "name": self.name,
            "kind": "boundary_load",
            "location": getattr(self.location, "name", None),
        }


@dataclass(frozen=True)
class NeumannLoad:
    """Natural boundary condition applied through the weak-form right hand side."""

    value: object
    measure: object
    name: str = "neumann_load"

    def form(self, test_function):
        return forms.boundary_flux_virtual_work(self.value, test_function, self.measure)

    def summary(self) -> dict[str, object]:
        """Return a compact description for logs and agent inspection."""

        return {"name": self.name, "kind": "neumann_load"}


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

    def summary(self) -> tuple[dict[str, object], ...]:
        """Return compact descriptions of all load terms."""

        result = []
        for load in self.loads:
            if hasattr(load, "summary"):
                result.append(load.summary())
            else:
                result.append({"name": getattr(load, "name", repr(load)), "kind": type(load).__name__})
        return tuple(result)


def body_load(value, measure=ufl.dx, *, name: str = "body_load", domain=None, target=None) -> BodyLoad:
    """Create a domain source/body-force load."""

    return BodyLoad(
        value=_as_constant(value, domain=domain, target=target),
        measure=measure,
        name=name,
    )


def body_force(value, *, domain=None, target=None, measure=ufl.dx, name: str = "body_force") -> BodyLoad:
    """Create a mechanical body-force load."""

    return body_load(value, measure=measure, name=name, domain=domain, target=target)


def heat_source(value, *, domain=None, target=None, measure=ufl.dx, name: str = "heat_source") -> BodyLoad:
    """Create a volumetric heat-source load."""

    return body_load(value, measure=measure, name=name, domain=domain, target=target)


def boundary_load(value, measure=None, *, location=None, name: str = "boundary_load") -> BoundaryLoad:
    """Create a generic natural boundary load."""

    selected_measure = measure if measure is not None else _location_measure(location)
    return BoundaryLoad(
        value=_as_constant(value, location=location),
        measure=selected_measure,
        name=name,
        location=location,
    )


def neumann(value, measure, *, name: str = "neumann_load") -> NeumannLoad:
    """Create a Neumann force/flux/traction term for the weak RHS."""

    return NeumannLoad(value=value, measure=measure, name=name)


def traction(value, *, location, name: str = "traction") -> BoundaryLoad:
    """Create a mechanical traction applied on a boundary region."""

    return boundary_load(value, location=location, name=name)


def heat_flux(value, *, location, name: str = "heat_flux") -> BoundaryLoad:
    """Create a prescribed heat flux applied on a boundary region."""

    return boundary_load(value, location=location, name=name)


def body_force_form(force, test_function):
    """Create a body-force virtual-work form."""

    return forms.body_force_virtual_work(force, test_function, ufl.dx)


def boundary_traction_form(traction, test_function, ds_measure):
    """Create a boundary-traction virtual-work form."""

    return forms.boundary_flux_virtual_work(traction, test_function, ds_measure)


def _location_measure(location):
    if location is None:
        raise ValueError("A boundary load requires measure or location.")
    if not hasattr(location, "measure"):
        raise ValueError("location must provide a boundary integration measure.")
    return location.measure


def _as_constant(value, *, location=None, domain=None, target=None):
    if _is_ufl_like(value):
        return value
    owner = domain or location or target
    if owner is None:
        return value
    return constants.constant(owner, value)


def _is_ufl_like(value) -> bool:
    return hasattr(value, "ufl_shape") or hasattr(value, "ufl_domain")
