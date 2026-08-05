"""Constraint containers for standard finite-element workflows.

Strong constraints such as Dirichlet data and periodic/MPC relations belong
here. Natural boundary data such as Neumann fluxes and tractions are weak-form
loads, so they belong in ``agentfem.loads``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
from dolfinx import fem
from mpi4py import MPI

from .. import amplitudes
from ..ir.values import describe_value
from ..kernel import constants
from . import boundary
from .affine import (
    AbaqusPeriodicConstraint,
    AffineReduction,
    DistributedAffineReduction,
    abaqus_periodic_cell,
)


@dataclass(frozen=True)
class DirichletConstraint:
    """Strong Dirichlet constraint and its optional mutable value object."""

    bc: object
    value: object | None = None
    name: str = "dirichlet"
    location: object | None = None

    @classmethod
    def component(
        cls,
        V,
        component: int,
        marker=None,
        value=0.0,
        *,
        location=None,
        name: str = "dirichlet",
    ):
        """Create a component-wise Dirichlet constraint on a vector space."""

        constant, bc = boundary.component_dirichlet_bc(
            V,
            component,
            marker,
            value=value,
            location=location,
        )
        selected_location = location
        return cls(bc=bc, value=constant, name=name, location=selected_location)

    @classmethod
    def scalar(cls, V, marker=None, value=0.0, *, location=None, name: str = "dirichlet"):
        """Create a scalar Dirichlet constraint."""

        constant, bc = boundary.scalar_dirichlet_bc(V, marker, value=value, location=location)
        return cls(bc=bc, value=constant, name=name, location=location)

    def summary(self) -> dict[str, object]:
        """Return a compact description for logs and agent inspection."""

        return {
            "name": self.name,
            "kind": "dirichlet_constraint",
            "location": getattr(self.location, "name", None),
            "value": describe_value(self.value),
        }


@dataclass(frozen=True)
class TimeDependentDirichlet:
    """Dirichlet constraint driven by an amplitude."""

    constant: object
    bc: object
    amplitude: amplitudes.Amplitude
    name: str = "time_dependent_dirichlet"
    location: object | None = None

    def update(self, time: float) -> float:
        """Evaluate the amplitude and update the backing constant."""

        value = self.amplitude(time)
        self.constant.value = constants.scalar_value(value)
        return value

    def summary(self) -> dict[str, object]:
        """Return a compact description for logs and agent inspection."""

        return {
            "name": self.name,
            "kind": "time_dependent_dirichlet",
            "location": getattr(self.location, "name", None),
            "amplitude": self.amplitude.summary(),
        }


@dataclass
class PrescribedValuePath:
    """Update ordinary strong boundary values along a normalized step path.

    Constant Dirichlet data are interpreted as end-of-step values and scaled
    by ``0 <= factor <= 1``.  A :class:`TimeDependentDirichlet` instead
    evaluates its own amplitude at the normalized factor.  The object keeps
    this policy visible and reusable by nonlinear procedures.
    """

    constants: tuple[tuple[object, object], ...] = ()
    amplitudes: tuple[TimeDependentDirichlet, ...] = ()

    def update(self, factor: float) -> None:
        selected = float(factor)
        if not 0.0 <= selected <= 1.0 + 1.0e-12:
            raise ValueError("PrescribedValuePath factor must lie in [0, 1].")
        for constant, reference in self.constants:
            constant.value = selected * reference
        for constraint in self.amplitudes:
            constraint.update(selected)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "prescribed_value_path",
            "constant_values": len(self.constants),
            "amplitude_values": len(self.amplitudes),
        }


def prescribed_value_path(constraints) -> PrescribedValuePath:
    """Create a normalized load-factor driver from registered constraints."""

    constants = []
    histories = []
    for item in _flatten_dirichlet(constraints):
        if isinstance(item, TimeDependentDirichlet):
            histories.append(item)
            continue
        value = getattr(item, "value", None)
        if value is None or not hasattr(value, "value"):
            continue
        reference = np.asarray(value.value).copy()
        constants.append((value, reference))
    return PrescribedValuePath(tuple(constants), tuple(histories))


def dirichlet_constraints(constraints) -> tuple[object, ...]:
    """Return concrete Dirichlet assets from nested model constraint sets."""

    return _flatten_dirichlet(constraints)


def _flatten_dirichlet(items) -> tuple[object, ...]:
    if items is None:
        return ()
    if isinstance(items, (list, tuple)):
        result = []
        for item in items:
            result.extend(_flatten_dirichlet(item))
        return tuple(result)
    if hasattr(items, "dirichlet"):
        return _flatten_dirichlet(items.dirichlet)
    return (items,)


def scalar_dirichlet(
    V,
    marker=None,
    value=0.0,
    *,
    location=None,
    on=None,
    name: str = "dirichlet",
) -> DirichletConstraint:
    """Semantic wrapper for scalar essential boundary data."""

    selected_location = _select_location(location=location, on=on)
    return DirichletConstraint.scalar(V, marker, value=value, location=selected_location, name=name)


def component_dirichlet(
    V,
    component: int,
    marker=None,
    value=0.0,
    *,
    location=None,
    on=None,
    name: str = "dirichlet",
) -> DirichletConstraint:
    """Semantic wrapper for vector-component essential boundary data."""

    selected_location = _select_location(location=location, on=on)
    return DirichletConstraint.component(
        V,
        component,
        marker,
        value=value,
        location=selected_location,
        name=name,
    )


def dirichlet(
    V,
    marker=None,
    value=0.0,
    *,
    component: int | None = None,
    location=None,
    on=None,
    name: str = "dirichlet",
) -> DirichletConstraint:
    """Create scalar or component-wise Dirichlet data from one entry point."""

    selected_location = _select_location(location=location, on=on)
    if component is None:
        return scalar_dirichlet(V, marker, value=value, location=selected_location, name=name)
    return component_dirichlet(
        V,
        component,
        marker,
        value=value,
        location=selected_location,
        name=name,
    )


def time_dependent_component_dirichlet(
    target,
    component: int,
    marker=None,
    value=None,
    *,
    amplitude=None,
    location=None,
    on=None,
    name: str = "time_dependent_dirichlet",
) -> TimeDependentDirichlet:
    """Create a component-wise Dirichlet constraint driven by an amplitude."""

    selected_location = _select_location(location=location, on=on)
    selected_amplitude = amplitude if amplitude is not None else value
    if selected_amplitude is None:
        raise ValueError("time_dependent_component_dirichlet requires value= or amplitude=.")
    history = amplitudes.as_amplitude(selected_amplitude, name=name)
    constant, bc = boundary.component_dirichlet_bc(
        target,
        component,
        marker,
        value=0.0,
        location=selected_location,
    )
    return TimeDependentDirichlet(
        constant=constant,
        bc=bc,
        amplitude=history,
        name=name,
        location=selected_location,
    )


def time_dependent_scalar_dirichlet(
    target,
    marker=None,
    value=None,
    *,
    amplitude=None,
    location=None,
    on=None,
    name: str = "time_dependent_dirichlet",
) -> TimeDependentDirichlet:
    """Create a scalar Dirichlet constraint driven by an amplitude."""

    selected_location = _select_location(location=location, on=on)
    selected_amplitude = amplitude if amplitude is not None else value
    if selected_amplitude is None:
        raise ValueError("time_dependent_scalar_dirichlet requires value= or amplitude=.")
    history = amplitudes.as_amplitude(selected_amplitude, name=name)
    constant, bc = boundary.scalar_dirichlet_bc(
        target,
        marker,
        value=0.0,
        location=selected_location,
    )
    return TimeDependentDirichlet(
        constant=constant,
        bc=bc,
        amplitude=history,
        name=name,
        location=selected_location,
    )


def apply_dirichlet_bcs(function, bcs) -> None:
    """Apply strong Dirichlet boundary conditions to a function vector."""

    boundary.apply_dirichlet_bcs(function, bcs)


def fixed(
    target,
    *,
    location=None,
    on=None,
    value=0.0,
    components: int | tuple[int, ...] | list[int] | None = None,
    name: str | None = None,
) -> "ConstraintSet":
    """Create fixed-value Dirichlet constraints for an application field.

    For scalar fields, ``components=None`` creates one scalar Dirichlet
    condition. For vector fields, ``components=None`` fixes every component.
    Provide one or more component indices when only selected dofs should be
    constrained.
    """

    selected_location = _select_location(location=location, on=on)
    if selected_location is None:
        raise ValueError("fixed requires a geometric location. Pass on=... or location=....")
    label = name or f"fixed_{getattr(selected_location, 'name', 'location')}"
    if components is None:
        components = _all_components_or_none(target)

    if components is None:
        return ConstraintSet(
            dirichlet=[
                scalar_dirichlet(target, location=selected_location, value=value, name=label),
            ]
        )

    component_ids = (int(components),) if isinstance(components, Integral) else tuple(components)
    component_values = _component_values(value, len(component_ids))
    return ConstraintSet(
        dirichlet=[
            component_dirichlet(
                target,
                component,
                location=selected_location,
                value=component_value,
                name=f"{label}_component_{component}",
            )
            for component, component_value in zip(component_ids, component_values)
        ]
    )


def fixed_component(
    target,
    component: int,
    *,
    location=None,
    on=None,
    value=0.0,
    name: str | None = None,
):
    """Create a fixed-value constraint for one vector component."""

    return fixed(target, location=location, on=on, value=value, components=component, name=name)


def symmetry(
    target,
    *,
    on=None,
    location=None,
    normal_axis: int | str,
    value=0.0,
    name: str | None = None,
) -> "ConstraintSet":
    """Apply an axis-aligned solid-mechanics symmetry condition.

    For displacement-only solid elements, symmetry fixes the displacement
    component normal to the plane. Arbitrary inclined symmetry planes require
    a linear multi-point constraint and are deliberately not approximated here.
    """

    component = _axis_component(normal_axis)
    available = _all_components_or_none(target)
    if available is None or component not in available:
        raise ValueError(
            f"normal_axis={normal_axis!r} selects component {component}, "
            f"but the target provides components={available!r}."
        )
    selected_location = _select_location(location=location, on=on)
    label = name or f"symmetry_{'xyz'[component]}"
    return fixed(
        target,
        location=selected_location,
        value=value,
        components=component,
        name=label,
    )


def roller(
    target,
    *,
    on=None,
    location=None,
    normal_axis: int | str,
    value=0.0,
    name: str | None = None,
) -> "ConstraintSet":
    """Alias for an axis-aligned frictionless roller/support condition."""

    return symmetry(
        target,
        on=on,
        location=location,
        normal_axis=normal_axis,
        value=value,
        name=name or f"roller_{normal_axis}",
    )


def fixed_all(target, *, location=None, on=None, value=0.0, name: str | None = None):
    """Create a scalar/all-dof fixed-value constraint."""

    return fixed(target, location=location, on=on, value=value, components=None, name=name)


def prescribed(
    target,
    *,
    on=None,
    location=None,
    value=0.0,
    component=None,
    components=None,
    name: str | None = None,
):
    """Create prescribed scalar or vector-component values.

    This engineering spelling is equivalent to ``fixed`` but reads naturally
    for non-zero displacement and temperature boundary data.
    """

    if component is not None:
        if components is not None:
            raise ValueError("Pass either component=... or components=..., not both.")
        components = component
    return fixed(
        target,
        on=on,
        location=location,
        value=value,
        components=components,
        name=name or "prescribed",
    )


def clamped(target, *, on=None, location=None, value=0.0, name: str | None = None):
    """Fix every displacement component on a support boundary."""

    available = _all_components_or_none(target)
    if available is None:
        raise ValueError("clamped requires a vector displacement-like field.")
    return fixed(
        target,
        on=on,
        location=location,
        value=value,
        components=available,
        name=name or "clamped",
    )


def prescribed_temperature(
    target,
    value,
    *,
    on=None,
    location=None,
    name: str | None = None,
):
    """Prescribe temperature on a named boundary."""

    if _all_components_or_none(target) is not None:
        raise ValueError("prescribed_temperature requires a scalar field.")
    return fixed(
        target,
        on=on,
        location=location,
        value=value,
        components=None,
        name=name or "prescribed_temperature",
    )


def _all_components_or_none(target) -> tuple[int, ...] | None:
    """Return vector component ids, or ``None`` for a scalar target."""

    value = getattr(target, "value", target)
    shape = getattr(value, "ufl_shape", ())
    if len(shape) == 0:
        return None
    if len(shape) != 1:
        raise ValueError(
            "Automatic fixed constraints only support scalar or vector fields. "
            "Pass components explicitly for tensor-valued targets."
        )
    return tuple(range(int(shape[0])))


def _component_values(value, count: int) -> tuple:
    """Expand a scalar or component-wise value into one value per component."""

    if _is_scalar_value(value):
        return tuple(value for _ in range(count))
    try:
        values = tuple(value)
    except TypeError:
        return tuple(value for _ in range(count))
    if len(values) != count:
        raise ValueError(
            f"Expected {count} component values for fixed constraint, got {len(values)}."
        )
    return values


def _is_scalar_value(value) -> bool:
    return isinstance(value, (str, bytes, Real)) or not hasattr(value, "__len__")


def _select_location(*, location=None, on=None):
    if location is not None and on is not None:
        raise ValueError("Pass either on=... or location=..., not both.")
    return location if location is not None else on


def _axis_component(axis: int | str) -> int:
    if isinstance(axis, str):
        normalized = axis.lower().strip()
        names = {"x": 0, "y": 1, "z": 2}
        if normalized not in names:
            raise ValueError("normal_axis must be x, y, z, 0, 1, or 2.")
        return names[normalized]
    selected = int(axis)
    if selected not in {0, 1, 2}:
        raise ValueError("normal_axis must be x, y, z, 0, 1, or 2.")
    return selected


def _space(target):
    if hasattr(target, "space"):
        return target.space
    if hasattr(target, "function_space"):
        return target.function_space
    if hasattr(target, "value") and hasattr(target.value, "function_space"):
        return target.value.function_space
    return target


def _region_marker(location):
    marker = getattr(location, "marker", location)
    if marker is None:
        raise ValueError("Periodic constraints require master/slave markers or regions.")
    return marker


def _axis_id(axis: str | int, gdim: int) -> int:
    if isinstance(axis, str):
        names = {"x": 0, "y": 1, "z": 2}
        key = axis.lower()
        if key not in names:
            raise ValueError("match_axis must be 'x', 'y', 'z', or an integer.")
        axis_id = names[key]
    else:
        axis_id = int(axis)
    if axis_id < 0 or axis_id >= int(gdim):
        raise ValueError(f"match_axis {axis!r} is outside geometric dimension {gdim}.")
    return axis_id


@dataclass(frozen=True)
class PeriodicProjectionConstraint:
    """Projection-style periodic constraint for explicit field updates.

    This method enforces equality by averaging paired dof values. It is useful
    for serial explicit dynamics workflows, but it is not a strict MPC
    constraint and does not currently support distributed meshes.
    """

    pairs: tuple[tuple[np.ndarray, np.ndarray], ...]
    pair_count: int
    name: str = "periodic_projection"
    master: object | None = None
    slave: object | None = None
    match_axis: str | int = 0
    supports_parallel: bool = False

    def apply(self, function) -> None:
        """Apply periodic equality by averaging paired dof values."""

        from .. import fields

        function = fields.unwrap(function)
        values = function.x.array
        for slave_dofs, master_dofs in self.pairs:
            averaged = 0.5 * (values[slave_dofs] + values[master_dofs])
            values[slave_dofs] = averaged
            values[master_dofs] = averaged
        function.x.scatter_forward()

    def __call__(self, function) -> None:
        """Callable alias for use by time integrators."""

        self.apply(function)

    def mismatch(self, function) -> float:
        """Return the max absolute paired-dof mismatch."""

        from .. import fields

        function = fields.unwrap(function)
        values = function.x.array
        local = 0.0
        for slave_dofs, master_dofs in self.pairs:
            if len(slave_dofs) > 0:
                local = max(local, float(np.max(np.abs(values[slave_dofs] - values[master_dofs]))))
        return function.function_space.mesh.comm.allreduce(local, op=MPI.MAX)

    def summary(self) -> dict[str, object]:
        """Return method and limitation details for logs or agents."""

        return {
            "name": self.name,
            "kind": "periodic_constraint",
            "method": "projection",
            "enforcement": "nodal_pair_averaging",
            "pair_count": self.pair_count,
            "master": getattr(self.master, "name", None),
            "slave": getattr(self.slave, "name", None),
            "match_axis": self.match_axis,
            "supports_parallel": self.supports_parallel,
        }


def periodic(
    target,
    *,
    master,
    slave,
    match_axis: str | int = 0,
    method: str = "projection",
    tolerance: float = 1.0e-12,
    name: str = "periodic",
):
    """Create a periodic constraint with an explicit method choice."""

    normalized = method.lower().replace("-", "_")
    if normalized in {"projection", "nodal_projection"}:
        return periodic_projection(
            target,
            master=master,
            slave=slave,
            match_axis=match_axis,
            tolerance=tolerance,
            name=name,
        )
    if normalized in {"mpc", "multi_point_constraint"}:
        raise NotImplementedError(
            "constraints.periodic(..., method='mpc') is planned but not implemented. "
            "Use method='projection' for serial explicit projection workflows."
        )
    raise ValueError(f"Unknown periodic constraint method: {method!r}.")


def periodic_projection(
    target,
    *,
    master,
    slave,
    match_axis: str | int = 0,
    tolerance: float = 1.0e-12,
    name: str = "periodic_projection",
) -> PeriodicProjectionConstraint:
    """Create component-wise dof pairs for projection-style periodicity."""

    V = _space(target)
    domain = V.mesh
    if domain.comm.size > 1:
        raise RuntimeError(
            "Projection-style periodic constraints are serial-only in this release. "
            "Use method='mpc' when a parallel implementation is added."
        )
    master_marker = _region_marker(master)
    slave_marker = _region_marker(slave)
    axis_id = _axis_id(match_axis, domain.geometry.dim)
    pairs = []
    pair_count = 0
    components = range(V.num_sub_spaces) if getattr(V, "num_sub_spaces", 0) else (None,)
    for component in components:
        if component is None:
            coords = V.tabulate_dof_coordinates()
            slave_parent = fem.locate_dofs_geometrical(V, slave_marker)
            master_parent = fem.locate_dofs_geometrical(V, master_marker)
            slave_child = slave_parent
            master_child = master_parent
        else:
            Vc, _ = V.sub(component).collapse()
            coords = Vc.tabulate_dof_coordinates()
            slave_parent, slave_child = fem.locate_dofs_geometrical(
                (V.sub(component), Vc), slave_marker
            )
            master_parent, master_child = fem.locate_dofs_geometrical(
                (V.sub(component), Vc), master_marker
            )
        if len(slave_child) != len(master_child):
            raise RuntimeError(
                "Periodic projection requires matching slave/master dofs "
                f"for component {component}."
            )

        slave_order = np.argsort(coords[slave_child, axis_id])
        master_order = np.argsort(coords[master_child, axis_id])
        slave_parent = np.asarray(slave_parent[slave_order], dtype=np.int32)
        master_parent = np.asarray(master_parent[master_order], dtype=np.int32)
        slave_coords = coords[slave_child[slave_order], axis_id]
        master_coords = coords[master_child[master_order], axis_id]
        if not np.allclose(slave_coords, master_coords, atol=tolerance, rtol=0.0):
            mismatch = float(np.max(np.abs(slave_coords - master_coords)))
            raise RuntimeError(
                "Periodic projection dofs are not aligned on match_axis "
                f"{match_axis!r}: max mismatch={mismatch:.3e}."
            )
        pairs.append((slave_parent, master_parent))
        pair_count += len(slave_parent)
    return PeriodicProjectionConstraint(
        pairs=tuple(pairs),
        pair_count=pair_count,
        name=name,
        master=master,
        slave=slave,
        match_axis=match_axis,
    )


@dataclass(frozen=True)
class PeriodicConstraintSpec:
    """Geometric description of a periodic constraint.

    This is a lightweight specification. Concrete implementations may use
    explicit nodal projection, MPC, or another backend.
    """

    slave_marker: object
    master_marker: object
    map_slave_to_master: object
    name: str = "periodic"

    def summary(self) -> dict[str, object]:
        """Return a compact description for logs and agent inspection."""

        return {"name": self.name, "kind": "periodic_constraint"}


@dataclass
class ConstraintSet:
    """Collection of constraints used by assembly or field updates."""

    dirichlet: list[DirichletConstraint] = field(default_factory=list)
    periodic: list[PeriodicConstraintSpec] = field(default_factory=list)

    @property
    def bcs(self) -> list:
        """Return DOLFINx Dirichlet BC objects."""

        return [constraint.bc for constraint in self.dirichlet]

    def add_dirichlet(self, constraint: DirichletConstraint) -> None:
        self.dirichlet.append(constraint)

    def add_periodic(self, constraint: PeriodicConstraintSpec) -> None:
        self.periodic.append(constraint)

    def extend(self, other: "ConstraintSet") -> None:
        """Append constraints from another set."""

        self.dirichlet.extend(other.dirichlet)
        self.periodic.extend(other.periodic)

    def summary(self) -> dict[str, object]:
        """Return compact descriptions of all constraints."""

        return {
            "dirichlet": tuple(constraint.summary() for constraint in self.dirichlet),
            "periodic": tuple(constraint.summary() for constraint in self.periodic),
        }
