"""Constraint containers for standard finite-element workflows.

Strong constraints such as Dirichlet data and periodic/MPC relations belong
here. Natural boundary data such as Neumann fluxes and tractions are weak-form
loads, so they belong in ``agentfem.loads``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real

from . import boundary


@dataclass(frozen=True)
class DirichletConstraint:
    """Strong Dirichlet constraint and its optional mutable value object."""

    bc: object
    value: object | None = None
    name: str = "dirichlet"

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
        return cls(bc=bc, value=constant, name=name)

    @classmethod
    def scalar(cls, V, marker=None, value=0.0, *, location=None, name: str = "dirichlet"):
        """Create a scalar Dirichlet constraint."""

        constant, bc = boundary.scalar_dirichlet_bc(V, marker, value=value, location=location)
        return cls(bc=bc, value=constant, name=name)

    def summary(self) -> dict[str, object]:
        """Return a compact description for logs and agent inspection."""

        return {"name": self.name, "kind": "dirichlet_constraint"}


def scalar_dirichlet(
    V,
    marker=None,
    value=0.0,
    *,
    location=None,
    name: str = "dirichlet",
) -> DirichletConstraint:
    """Semantic wrapper for scalar essential boundary data."""

    return DirichletConstraint.scalar(V, marker, value=value, location=location, name=name)


def component_dirichlet(
    V,
    component: int,
    marker=None,
    value=0.0,
    *,
    location=None,
    name: str = "dirichlet",
) -> DirichletConstraint:
    """Semantic wrapper for vector-component essential boundary data."""

    return DirichletConstraint.component(
        V,
        component,
        marker,
        value=value,
        location=location,
        name=name,
    )


def dirichlet(
    V,
    marker=None,
    value=0.0,
    *,
    component: int | None = None,
    location=None,
    name: str = "dirichlet",
) -> DirichletConstraint:
    """Create scalar or component-wise Dirichlet data from one entry point."""

    if component is None:
        return scalar_dirichlet(V, marker, value=value, location=location, name=name)
    return component_dirichlet(
        V,
        component,
        marker,
        value=value,
        location=location,
        name=name,
    )


def apply_dirichlet_bcs(function, bcs) -> None:
    """Apply strong Dirichlet boundary conditions to a function vector."""

    boundary.apply_dirichlet_bcs(function, bcs)


def fixed(
    target,
    *,
    location,
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

    label = name or f"fixed_{getattr(location, 'name', 'location')}"
    if components is None:
        components = _all_components_or_none(target)

    if components is None:
        return ConstraintSet(
            dirichlet=[
                scalar_dirichlet(target, location=location, value=value, name=label),
            ]
        )

    component_ids = (int(components),) if isinstance(components, Integral) else tuple(components)
    component_values = _component_values(value, len(component_ids))
    return ConstraintSet(
        dirichlet=[
            component_dirichlet(
                target,
                component,
                location=location,
                value=component_value,
                name=f"{label}_component_{component}",
            )
            for component, component_value in zip(component_ids, component_values)
        ]
    )


def fixed_component(target, component: int, *, location, value=0.0, name: str | None = None):
    """Create a fixed-value constraint for one vector component."""

    return fixed(target, location=location, value=value, components=component, name=name)


def fixed_all(target, *, location, value=0.0, name: str | None = None):
    """Create a scalar/all-dof fixed-value constraint."""

    return fixed(target, location=location, value=value, components=None, name=name)


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
