"""Standard natural-load helpers."""

from __future__ import annotations

from dataclasses import dataclass

import ufl

from . import amplitudes
from .constraints import boundary
from .constraints import TimeDependentDirichlet
from .constraints import time_dependent_component_dirichlet as _constraint_time_dirichlet
from . import forms
from .kernel import constants


TimeFunction = amplitudes.Amplitude


def time_dependent_component_dirichlet(V, component: int, marker, time_function):
    """Compatibility wrapper for time-dependent component Dirichlet constraints."""

    return _constraint_time_dirichlet(V, component, marker, value=time_function)


def apply_dirichlet_bcs(function, bcs) -> None:
    """Apply strong Dirichlet boundary conditions to a function vector."""

    boundary.apply_dirichlet_bcs(function, bcs)


def constant_time_function(value: float, name: str = "constant") -> amplitudes.Amplitude:
    """Represent a constant value with the same interface as transient data."""

    return amplitudes.constant(value, name=name)


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


def boundary_load(
    value,
    measure=None,
    *,
    location=None,
    on=None,
    name: str = "boundary_load",
) -> BoundaryLoad:
    """Create a generic natural boundary load."""

    selected_location = _select_location(location=location, on=on)
    selected_measure = measure if measure is not None else _location_measure(selected_location)
    return BoundaryLoad(
        value=_as_constant(value, location=selected_location),
        measure=selected_measure,
        name=name,
        location=selected_location,
    )


def neumann(value, measure, *, name: str = "neumann_load") -> NeumannLoad:
    """Create a Neumann force/flux/traction term for the weak RHS."""

    return NeumannLoad(value=value, measure=measure, name=name)


def traction(value, *, location=None, on=None, name: str = "traction") -> BoundaryLoad:
    """Create a mechanical traction applied on a boundary region."""

    return boundary_load(value, location=location, on=on, name=name)


def heat_flux(value, *, location=None, on=None, name: str = "heat_flux") -> BoundaryLoad:
    """Create a prescribed heat flux applied on a boundary region."""

    return boundary_load(value, location=location, on=on, name=name)


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


def _select_location(*, location=None, on=None):
    if location is not None and on is not None:
        raise ValueError("Pass either on=... or location=..., not both.")
    return location if location is not None else on


def _as_constant(value, *, location=None, domain=None, target=None):
    if _is_ufl_like(value):
        return value
    owner = domain or location or target
    if owner is None:
        return value
    return constants.constant(owner, value)


def _is_ufl_like(value) -> bool:
    return hasattr(value, "ufl_shape") or hasattr(value, "ufl_domain")
