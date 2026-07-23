"""Problem/state containers for standard finite-element workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import assembly
from . import dofs
from . import spaces
from . import time
from .solvers import LinearSolverOptions, solve_linear_problem


@dataclass
class LinearVariationalProblem:
    """A standard linear variational problem, ``a(u, v) = L(v)``."""

    bilinear_form: object
    linear_form: object
    solution: object
    bcs: list = field(default_factory=list)
    solver_options: LinearSolverOptions | None = None

    def solve(self):
        """Assemble and solve the problem into ``solution``."""

        return solve_linear_problem(
            self.bilinear_form,
            self.linear_form,
            self.solution,
            bcs=self.bcs,
            options=self.solver_options,
        )


@dataclass
class TransientState:
    """Current/next fields for a first-order transient unknown."""

    current: object
    next: object

    @classmethod
    def create(cls, V, *, name: str = "Field"):
        """Create a zero-initialized transient state on a function space."""

        return cls(
            current=spaces.named_function(V, name),
            next=spaces.named_function(V, name),
        )

    def accept_step(self) -> None:
        """Copy ``next`` into ``current``."""

        dofs.copy_function(self.current, self.next)


@dataclass
class SecondOrderDynamicsState:
    """Displacement/velocity/acceleration fields for second-order dynamics."""

    u: object
    v: object
    a: object
    u_next: object
    v_next: object
    a_next: object

    @classmethod
    def create(
        cls,
        V,
        *,
        displacement_name: str = "Displacement",
        velocity_name: str = "Velocity",
        acceleration_name: str = "Acceleration",
    ):
        """Create zero-initialized explicit dynamics fields for a space."""

        return cls(
            u=spaces.named_function(V, displacement_name),
            v=spaces.named_function(V, velocity_name),
            a=spaces.named_function(V, acceleration_name),
            u_next=spaces.named_function(V, displacement_name),
            v_next=spaces.named_function(V, velocity_name),
            a_next=spaces.named_function(V, acceleration_name),
        )

    def predict_displacement(self, dt: float) -> None:
        """Predict ``u_next`` from the current state."""

        time.central_difference_predict_displacement(self.u_next, self.u, self.v, self.a, dt)

    def set_acceleration_from_residual(self, residual, inv_mass: np.ndarray) -> None:
        """Set ``a_next`` from a residual vector and inverse lumped mass."""

        time.acceleration_from_residual(self.a_next, residual, inv_mass)

    def correct_velocity(self, dt: float) -> None:
        """Correct ``v_next`` using ``a`` and ``a_next``."""

        time.central_difference_correct_velocity(self.v_next, self.v, self.a, self.a_next, dt)

    def accept_step(self) -> None:
        """Copy next-step fields into current fields."""

        dofs.copy_function(self.u, self.u_next)
        dofs.copy_function(self.v, self.v_next)
        dofs.copy_function(self.a, self.a_next)

    def accept_displacement(self) -> None:
        """Copy ``u_next`` into ``u``."""

        dofs.copy_function(self.u, self.u_next)

    def accept_velocity_acceleration(self) -> None:
        """Copy ``v_next`` and ``a_next`` into ``v`` and ``a``."""

        dofs.copy_function(self.v, self.v_next)
        dofs.copy_function(self.a, self.a_next)


@dataclass
class LumpedMassOperator:
    """Diagonal mass operator for explicit dynamics."""

    mass: np.ndarray
    inv_mass: np.ndarray

    @classmethod
    def assemble(cls, V, density=1.0, measure=None):
        """Assemble a lumped mass operator for a function space."""

        if measure is None:
            mass = assembly.assemble_lumped_mass(V, density)
        else:
            mass = assembly.assemble_lumped_mass(V, density, measure=measure)
        return cls(mass=mass, inv_mass=assembly.inverse_diagonal(mass))


ExplicitDynamicsState = SecondOrderDynamicsState
