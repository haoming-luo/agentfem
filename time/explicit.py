"""Explicit dynamics time-integration workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .. import constraints
from .. import fields
from .. import operators
from ..kernel import dofs


@dataclass(frozen=True)
class NewmarkParameters:
    """Newmark family parameters."""

    beta: float
    gamma: float

    @classmethod
    def central_difference(cls) -> "NewmarkParameters":
        """Return the explicit central-difference Newmark parameters."""

        return cls(beta=0.0, gamma=0.5)

    def summary(self) -> dict[str, float]:
        """Return an agent-readable parameter summary."""

        return {"beta": self.beta, "gamma": self.gamma}


@dataclass
class ExplicitDynamicsIntegrator:
    """Formula-level explicit second-order dynamics integrator.

    The default method is central difference, i.e. Newmark with
    ``beta = 0`` and ``gamma = 1/2``.
    """

    state: object
    mass: object
    parameters: NewmarkParameters
    method: str = "central_difference"
    name: str = "explicit_dynamics"

    @property
    def inv_mass(self) -> np.ndarray:
        """Return the inverse lumped mass diagonal."""

        if hasattr(self.mass, "inv_mass"):
            return self.mass.inv_mass
        return self.mass

    def predict_displacement(self, dt: float) -> None:
        """Predict displacement using the explicit Newmark formula."""

        beta = self.parameters.beta
        if beta != 0.0:
            raise NotImplementedError(
                "ExplicitDynamicsIntegrator currently supports beta=0 central difference."
            )
        self.state.u_next.assign(
            self.state.u
            + dt * self.state.v
            + 0.5 * dt**2 * self.state.a
        )

    def update_displacement(self) -> None:
        """Use the constrained/projected predicted displacement as current."""

        fields.assign(self.state.u, self.state.u_next)

    def update_midstep_velocity(self, dt: float) -> None:
        """Update the mid-step velocity ``v_mid = v_n + dt/2 a_n``."""

        gamma = self.parameters.gamma
        if gamma != 0.5:
            raise NotImplementedError(
                "ExplicitDynamicsIntegrator currently supports gamma=1/2 central difference."
            )
        self.state.v_mid.assign(self.state.v + 0.5 * dt * self.state.a)

    def solve_acceleration(self, residual) -> None:
        """Solve the explicit acceleration update ``a_next = -M^{-1} r``."""

        dofs.assign_owned(self.state.a_next, -residual.array * self.inv_mass)

    def update_velocity(self, dt: float) -> None:
        """Update whole-step velocity ``v_next = v_mid + dt/2 a_next``."""

        self.state.v_next.assign(self.state.v_mid + 0.5 * dt * self.state.a_next)

    def advance_state(self) -> None:
        """Advance ``u``, ``v``, and ``a`` to the next time level."""

        fields.assign(self.state.u, self.state.u_next)
        fields.assign(self.state.v, self.state.v_next)
        fields.assign(self.state.a, self.state.a_next)

    def advance_velocity_acceleration(self) -> None:
        """Advance velocity and acceleration while displacement is already active."""

        fields.assign(self.state.v, self.state.v_next)
        fields.assign(self.state.a, self.state.a_next)

    def step(
        self,
        dt: float,
        *,
        time: float | None = None,
        residual_operator=None,
        displacement_bcs=None,
        projections: Iterable[Callable[[object], None]] = (),
        update_prescribed_values: Iterable[Callable[[float], object]] = (),
    ):
        """Advance one explicit central-difference step.

        This high-level helper is intentionally thin. Formula-level examples can
        call the individual methods directly when the algorithm should remain
        visible line by line.
        """

        if time is not None:
            for update in update_prescribed_values:
                update(time)
        self.predict_displacement(dt)
        if displacement_bcs:
            constraints.apply_dirichlet_bcs(self.state.u_next, displacement_bcs)
        for projection in projections:
            projection(self.state.u_next)
        self.update_displacement()
        self.update_midstep_velocity(dt)
        residual = operators.assemble_vector(residual_operator)
        try:
            self.solve_acceleration(residual)
        finally:
            residual.destroy()
        for projection in projections:
            projection(self.state.a_next)
        self.update_velocity(dt)
        for projection in projections:
            projection(self.state.v_next)
        self.advance_velocity_acceleration()

    def summary(self) -> dict[str, object]:
        """Return an agent-readable integration summary."""

        return {
            "name": self.name,
            "family": "explicit_dynamics",
            "method": self.method,
            "newmark_beta": self.parameters.beta,
            "newmark_gamma": self.parameters.gamma,
            "mass": "lumped" if hasattr(self.mass, "inv_mass") else type(self.mass).__name__,
        }


def central_difference(
    *,
    state,
    mass,
    name: str = "central_difference",
) -> ExplicitDynamicsIntegrator:
    """Create a central-difference explicit dynamics integrator."""

    return ExplicitDynamicsIntegrator(
        state=state,
        mass=mass,
        parameters=NewmarkParameters.central_difference(),
        method="central_difference",
        name=name,
    )
