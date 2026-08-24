"""Standard natural-load helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from . import amplitudes
from . import _axisymmetric
from .constraints import boundary
from .constraints import TimeDependentDirichlet
from .constraints import time_dependent_component_dirichlet as _constraint_time_dirichlet
from . import forms
from .ir.values import describe_value
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

        return {
            "name": self.name,
            "kind": "body_load",
            "value": describe_value(self.value),
        }


@dataclass(frozen=True)
class GravityLoad:
    """Gravity body force ``rho g`` over a material domain."""

    acceleration: object
    density: object
    value: object
    measure: object = ufl.dx
    name: str = "gravity"
    location: object | None = None

    def form(self, test_function):
        return forms.body_force_virtual_work(self.value, test_function, self.measure)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "gravity_load",
            "location": getattr(self.location, "name", None),
            "acceleration": describe_value(self.acceleration),
            "density": describe_value(self.density),
        }


@dataclass(frozen=True)
class CentrifugalLoad:
    """Rotating-frame body force ``rho omega x (omega x r)`` outward."""

    angular_velocity: object
    center: tuple[float, ...]
    density: object
    value: object
    measure: object = ufl.dx
    name: str = "centrifugal"
    location: object | None = None

    def form(self, test_function):
        return forms.body_force_virtual_work(self.value, test_function, self.measure)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "centrifugal_load",
            "location": getattr(self.location, "name", None),
            "center": self.center,
            "angular_velocity": describe_value(self.angular_velocity),
            "density": describe_value(self.density),
        }


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
            "value": describe_value(self.value),
        }


@dataclass(frozen=True)
class PressureLoad:
    """Pressure load pulled back to a reference boundary measure."""

    pressure: object
    traction: object
    measure: object
    configuration: str = "reference"
    name: str = "pressure"
    location: object | None = None

    @property
    def value(self):
        return self.traction

    def form(self, test_function):
        return forms.boundary_flux_virtual_work(
            self.traction,
            test_function,
            self.measure,
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "pressure_load",
            "location": getattr(self.location, "name", None),
            "pressure": describe_value(self.pressure),
            "configuration": self.configuration,
            "sign_convention": "positive pressure acts inward",
        }


@dataclass(frozen=True)
class HydrostaticPressureLoad(PressureLoad):
    """Pressure varying with elevation from a reference free surface."""

    density: object = None
    gravity: object = None
    reference_point: tuple[float, ...] = ()

    def summary(self) -> dict[str, object]:
        result = super().summary()
        result.update(
            {
                "kind": "hydrostatic_pressure_load",
                "density": describe_value(self.density),
                "gravity": describe_value(self.gravity),
                "reference_point": self.reference_point,
            }
        )
        return result


@dataclass(frozen=True)
class SurfaceResultantLoad:
    """A requested total force uniformly distributed over a reference boundary.

    This is useful for continuum-solid models where a physical end load is
    known but a singular nodal force is neither intended nor numerically
    desirable.  In two-dimensional studies the resultant is per unit
    out-of-plane thickness.
    """

    resultant: tuple[float, ...]
    traction: object
    reference_measure: float
    measure: object
    location: object
    name: str = "surface_force"

    @property
    def value(self):
        return self.traction

    def form(self, test_function):
        return forms.boundary_flux_virtual_work(
            self.traction,
            test_function,
            self.measure,
        )

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "surface_resultant_load",
            "location": getattr(self.location, "name", None),
            "resultant": self.resultant,
            "reference_measure": self.reference_measure,
            "traction": describe_value(self.traction),
            "configuration": "reference",
            "distribution": "uniform",
        }


@dataclass(frozen=True)
class DistributedCouplingLoad:
    """Force and moment distributed over a continuum surface."""

    force: tuple[float, ...]
    moment: object
    reference_point: tuple[float, ...]
    centroid: tuple[float, ...]
    traction: object
    reference_measure: float
    measure: object
    location: object
    name: str = "distributing_coupling"
    reference_name: str | None = None
    coordinate_system: str | None = None

    @property
    def value(self):
        return self.traction

    def form(self, test_function):
        return forms.boundary_flux_virtual_work(self.traction, test_function, self.measure)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "distributed_coupling_load",
            "location": getattr(self.location, "name", None),
            "force": self.force,
            "moment": self.moment,
            "reference_point": self.reference_point,
            "reference_name": self.reference_name,
            "coordinate_system": self.coordinate_system,
            "surface_centroid": self.centroid,
            "reference_measure": self.reference_measure,
            "weighting": "continuum_tributary_measure",
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

        return {
            "name": self.name,
            "kind": "neumann_load",
            "value": describe_value(self.value),
        }


@dataclass(frozen=True)
class AmplitudeLoad:
    """A spatial load multiplied by one reusable scalar amplitude."""

    load: object
    scale: object
    amplitude: amplitudes.Amplitude
    name: str = "amplitude_load"

    @property
    def location(self):
        return getattr(self.load, "location", None)

    @property
    def value(self):
        return self.scale * getattr(self.load, "value", self.load)

    def update(self, time: float) -> float:
        value = self.amplitude(time)
        self.scale.value = PETSc.ScalarType(value)
        return value

    def form(self, test_function):
        return self.scale * self.load.form(test_function)

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": "amplitude_load",
            "load": (
                self.load.summary()
                if hasattr(self.load, "summary")
                else type(self.load).__name__
            ),
            "amplitude": self.amplitude.summary(),
        }


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


def body_force(
    value, *, domain=None, target=None, measure=ufl.dx,
    system=None, name: str = "body_force",
) -> BodyLoad:
    """Create a mechanical body-force load in global or local components."""

    return body_load(
        _global_vector(value, system),
        measure=measure,
        name=name,
        domain=domain,
        target=target,
    )


def gravity(
    acceleration,
    *,
    density,
    domain=None,
    target=None,
    region=None,
    measure=None,
    system=None,
    name: str = "gravity",
) -> GravityLoad:
    """Create a gravity load from acceleration and material density.

    Acceleration follows the global coordinate components, for example
    ``(0, -9.81)`` in 2D or ``(0, 0, -9.81)`` in 3D.  The resulting body
    force density is ``rho * acceleration``.
    """

    owner = domain or target or region
    if owner is None:
        raise ValueError("gravity requires domain=, target=, or region=.")
    selected_measure = measure
    if selected_measure is None:
        selected_measure = getattr(region, "measure", ufl.dx)
    selected_acceleration = _as_constant(
        _global_vector(acceleration, system), domain=owner
    )
    selected_density = _as_constant(density, domain=owner)
    return GravityLoad(
        acceleration=selected_acceleration,
        density=selected_density,
        value=selected_density * selected_acceleration,
        measure=selected_measure,
        name=name,
        location=region,
    )


def centrifugal(
    angular_velocity,
    *,
    density,
    center=None,
    domain=None,
    target=None,
    region=None,
    measure=None,
    name: str = "centrifugal",
) -> CentrifugalLoad:
    """Create the outward body force caused by constant angular velocity."""

    owner = domain or target or region
    if owner is None:
        raise ValueError("centrifugal requires domain=, target=, or region=.")
    selected_domain = getattr(owner, "domain", owner)
    if hasattr(selected_domain, "function_space"):
        selected_domain = selected_domain.function_space.mesh
    dimension = int(selected_domain.geometry.dim)
    origin = np.zeros(dimension) if center is None else np.asarray(center, dtype=float)
    if origin.shape != (dimension,) or not np.all(np.isfinite(origin)):
        raise ValueError(f"centrifugal center must have {dimension} finite components.")
    x = ufl.SpatialCoordinate(selected_domain)
    radial = x - ufl.as_vector(tuple(float(value) for value in origin))
    if dimension == 2:
        omega = float(np.asarray(angular_velocity).reshape(-1)[0])
        acceleration = omega**2 * radial
        selected_omega = omega
    elif dimension == 3:
        values = np.asarray(angular_velocity, dtype=float).reshape(-1)
        if values.size == 1:
            values = np.asarray((0.0, 0.0, float(values[0])))
        if values.size != 3 or not np.all(np.isfinite(values)):
            raise ValueError("3D centrifugal angular_velocity needs 3 components.")
        selected_omega = constants.constant(selected_domain, values)
        acceleration = -ufl.cross(selected_omega, ufl.cross(selected_omega, radial))
    else:
        raise NotImplementedError("centrifugal currently supports 2D and 3D solids.")
    selected_density = constants.constant(selected_domain, density)
    return CentrifugalLoad(
        angular_velocity=selected_omega,
        center=tuple(float(value) for value in origin),
        density=selected_density,
        value=selected_density * acceleration,
        measure=measure or getattr(region, "measure", ufl.dx),
        name=name,
        location=region,
    )


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


def with_amplitude(load, amplitude, *, domain=None, name: str | None = None) -> AmplitudeLoad:
    """Drive an existing load by a scalar amplitude multiplier."""

    history = amplitudes.as_amplitude(
        amplitude,
        name=name or f"{getattr(load, 'name', 'load')}_amplitude",
    )
    selected_domain = domain
    if selected_domain is None:
        location = getattr(load, "location", None)
        selected_domain = getattr(location, "domain", None)
    if selected_domain is None:
        value = getattr(load, "value", None)
        selected_domain = getattr(value, "ufl_domain", lambda: None)()
    if selected_domain is None:
        raise ValueError("with_amplitude requires a load domain.")
    scale = fem.Constant(selected_domain, PETSc.ScalarType(history(0.0)))
    return AmplitudeLoad(
        load=load,
        scale=scale,
        amplitude=history,
        name=name or getattr(load, "name", "amplitude_load"),
    )


def traction(
    value, *, location=None, on=None, system=None, name: str = "traction"
) -> BoundaryLoad:
    """Create a traction in global or an explicit local coordinate system."""

    return boundary_load(
        _global_vector(value, system), location=location, on=on, name=name
    )


def surface_force(
    resultant,
    *,
    location=None,
    on=None,
    reference_measure: float | None = None,
    study=None,
    system=None,
    name: str = "surface_force",
) -> SurfaceResultantLoad:
    """Distribute a total reference-configuration force over a boundary.

    ``resultant`` is the desired MPI-global force vector.  AgentFEM computes
    the selected edge length in planar 2D, revolved area in axisymmetric 2D,
    or surface area in 3D and applies the uniform traction
    ``resultant / reference_measure``.  Pass an explicit
    ``reference_measure`` only when the geometric measure is intentionally
    supplied by an external model.
    """

    selected = _select_location(location=location, on=on)
    if selected is None or not hasattr(selected, "domain") or not hasattr(selected, "measure"):
        raise ValueError("surface_force requires a named boundary region.")
    values = np.asarray(_global_vector(resultant, system), dtype=float).reshape(-1)
    dimension = int(selected.domain.geometry.dim)
    if values.size != dimension:
        raise ValueError(
            f"surface_force requires {dimension} force components, got {values.size}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("surface_force resultant components must be finite.")
    area = reference_measure
    if area is None:
        weight = _axisymmetric.integration_weight(selected.domain, study)
        local = fem.assemble_scalar(fem.form(weight * selected.measure))
        area = selected.domain.comm.allreduce(float(local), op=MPI.SUM)
    area = float(area)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("surface_force reference_measure must be finite and positive.")
    traction_value = constants.constant(selected.domain, values / area)
    return SurfaceResultantLoad(
        resultant=tuple(float(value) for value in values),
        traction=traction_value,
        reference_measure=area,
        measure=selected.measure,
        location=selected,
        name=name,
    )


def distributing_coupling(
    force, *, moment=None, reference_point=None, location=None, on=None,
    system=None,
    name: str = "distributing_coupling",
) -> DistributedCouplingLoad:
    """Distribute force/moment over a surface with tributary-area weighting."""

    selected = _select_location(location=location, on=on)
    if selected is None or not hasattr(selected, "domain"):
        raise ValueError("distributing_coupling requires a boundary region.")
    domain, measure = selected.domain, selected.measure
    dimension = int(domain.geometry.dim)
    values = np.asarray(_global_vector(force, system), dtype=float).reshape(-1)
    if values.size != dimension or not np.all(np.isfinite(values)):
        raise ValueError(f"force must have {dimension} finite components.")
    x = ufl.SpatialCoordinate(domain)
    def assembled(expression):
        local = fem.assemble_scalar(fem.form(expression * measure))
        return float(domain.comm.allreduce(float(local), op=MPI.SUM))
    area = assembled(ufl.as_ufl(1.0))
    if area <= 0.0:
        raise ValueError("distributing_coupling requires positive surface measure.")
    centroid = np.asarray([assembled(x[i]) / area for i in range(dimension)])
    reference = centroid if reference_point is None else np.asarray(reference_point, dtype=float).reshape(-1)
    if reference.size != dimension:
        raise ValueError(f"reference_point must have {dimension} components.")
    arm = x - ufl.as_vector(tuple(float(value) for value in centroid))
    base = ufl.as_vector(tuple(float(value) for value in values / area))
    if dimension == 2:
        selected_moment = float(0.0 if moment is None else np.asarray(moment).reshape(-1)[0])
        offset = centroid - reference
        target = selected_moment - (offset[0] * values[1] - offset[1] * values[0])
        polar = assembled(ufl.dot(arm, arm)) / area
        if abs(target) > 0.0 and polar <= np.finfo(float).eps:
            raise ValueError("Surface cannot transmit the requested moment.")
        alpha = 0.0 if polar <= np.finfo(float).eps else float(target / polar)
        correction = (alpha / area) * ufl.as_vector((-arm[1], arm[0]))
    elif dimension == 3:
        requested = (
            np.zeros(3)
            if moment is None
            else np.asarray(_global_vector(moment, system), dtype=float).reshape(-1)
        )
        if requested.size != 3:
            raise ValueError("3D distributing-coupling moment needs 3 components.")
        target = requested - np.cross(centroid - reference, values)
        tensor = np.asarray([[assembled((ufl.dot(arm, arm) if i == j else 0.0) - arm[i]*arm[j])/area for j in range(3)] for i in range(3)])
        if np.linalg.matrix_rank(tensor) < 3 and np.linalg.norm(target) > 1.0e-12:
            raise ValueError("Surface geometry cannot transmit the requested 3D moment.")
        alpha = np.linalg.pinv(tensor) @ target
        correction = ufl.cross(ufl.as_vector(tuple(alpha / area)), arm)
        selected_moment = tuple(float(value) for value in requested)
    else:
        raise NotImplementedError("distributing_coupling supports 2D and 3D solids.")
    return DistributedCouplingLoad(
        force=tuple(float(value) for value in values),
        moment=selected_moment,
        reference_point=tuple(float(value) for value in reference),
        centroid=tuple(float(value) for value in centroid),
        traction=base + correction,
        reference_measure=area,
        measure=measure,
        location=selected,
        name=name,
        reference_name=getattr(reference_point, "name", None),
        coordinate_system=getattr(system, "name", None),
    )


def remote_force(
    force,
    *,
    reference_point,
    moment=None,
    location=None,
    on=None,
    system=None,
    name: str = "remote_force",
) -> DistributedCouplingLoad:
    """Apply a reference-point force/moment through a continuum surface."""

    return distributing_coupling(
        force,
        moment=moment,
        reference_point=reference_point,
        location=location,
        on=on,
        system=system,
        name=name,
    )


def pressure(
    value,
    *,
    location=None,
    on=None,
    normal=None,
    configuration: str = "reference",
    displacement=None,
    name: str = "pressure",
) -> PressureLoad:
    """Create inward pressure on a reference or current boundary.

    ``configuration="reference"`` is a dead nominal pressure ``-p N``.
    ``configuration="current"`` is a follower pressure pulled back with
    Nanson's relation, ``-p J F^{-T} N``; it therefore requires the current
    displacement field and contributes to the nonlinear tangent automatically.
    """

    selected_location = _select_location(location=location, on=on)
    if selected_location is None or not hasattr(selected_location, "domain"):
        raise ValueError("pressure requires a boundary region with a domain.")
    domain = selected_location.domain
    selected_pressure = _as_constant(value, location=selected_location)
    reference_normal = normal if normal is not None else ufl.FacetNormal(domain)
    normalized = str(configuration).lower().replace("-", "_")
    if normalized in {"reference", "dead", "nominal"}:
        traction_value = -selected_pressure * reference_normal
        normalized = "reference"
    elif normalized in {"current", "follower"}:
        if displacement is None:
            raise ValueError(
                "Current-configuration follower pressure requires displacement=...."
            )
        current = getattr(displacement, "value", displacement)
        dimension = len(current)
        F = ufl.Identity(dimension) + ufl.grad(current)
        traction_value = (
            -selected_pressure
            * ufl.det(F)
            * ufl.inv(F).T
            * reference_normal
        )
        normalized = "current"
    else:
        raise ValueError("pressure configuration must be reference or current.")
    return PressureLoad(
        pressure=selected_pressure,
        traction=traction_value,
        measure=selected_location.measure,
        configuration=normalized,
        name=name,
        location=selected_location,
    )


def hydrostatic_pressure(
    *,
    density,
    gravity,
    reference_point,
    reference_pressure=0.0,
    on=None,
    location=None,
    clip_at_zero: bool = True,
    configuration: str = "reference",
    displacement=None,
    name: str = "hydrostatic_pressure",
) -> HydrostaticPressureLoad:
    """Create ``p = p_ref + rho g dot (x - x_ref)`` on a boundary."""

    selected = _select_location(location=location, on=on)
    if selected is None or not hasattr(selected, "domain"):
        raise ValueError("hydrostatic_pressure requires a boundary region.")
    dimension = int(selected.domain.geometry.dim)
    point = np.asarray(reference_point, dtype=float).reshape(-1)
    gravity_values = np.asarray(gravity, dtype=float).reshape(-1)
    if point.size != dimension or gravity_values.size != dimension:
        raise ValueError(
            f"reference_point and gravity must each have {dimension} components."
        )
    rho = constants.constant(selected.domain, density)
    gravity_field = constants.constant(selected.domain, gravity_values)
    x = ufl.SpatialCoordinate(selected.domain)
    pressure_value = (
        reference_pressure
        + rho * ufl.dot(gravity_field, x - ufl.as_vector(tuple(point)))
    )
    if clip_at_zero:
        pressure_value = ufl.max_value(pressure_value, 0.0)
    base = pressure(
        pressure_value,
        location=selected,
        configuration=configuration,
        displacement=displacement,
        name=name,
    )
    return HydrostaticPressureLoad(
        pressure=base.pressure,
        traction=base.traction,
        measure=base.measure,
        configuration=base.configuration,
        name=base.name,
        location=base.location,
        density=rho,
        gravity=gravity_field,
        reference_point=tuple(float(value) for value in point),
    )


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


def _global_vector(value, system):
    """Return global components while keeping coordinate conventions explicit."""

    if system is None:
        return value
    if not hasattr(system, "vector_to_global"):
        raise TypeError("system must provide vector_to_global(...).")
    return system.vector_to_global(value)


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
