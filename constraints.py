"""Constraint containers for standard finite-element workflows.

Strong constraints such as Dirichlet data and periodic/MPC relations belong
here. Natural boundary data such as Neumann fluxes and tractions are weak-form
loads, so they belong in ``agentfem.loads``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import boundary


@dataclass(frozen=True)
class DirichletConstraint:
    """Strong Dirichlet constraint and its optional mutable value object."""

    bc: object
    value: object | None = None
    name: str = "dirichlet"

    @classmethod
    def component(cls, V, component: int, marker, value=0.0, *, name: str = "dirichlet"):
        """Create a component-wise Dirichlet constraint on a vector space."""

        constant, bc = boundary.component_dirichlet_bc(V, component, marker, value=value)
        return cls(bc=bc, value=constant, name=name)


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
