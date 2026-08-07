"""Explicit dynamics time-integration workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .. import constraints as constraint_api
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
        prescribed: Iterable[object] = (),
        constraints: Iterable[object] = (),
        displacement_bcs=None,
        projections: Iterable[Callable[[object], None]] = (),
        update_prescribed_values: Iterable[Callable[[float], object]] = (),
    ):
        """Advance one explicit central-difference step.

        ``prescribed`` contains prescribed values such as time-dependent
        Dirichlet data. ``constraints`` contains model constraints such as
        periodic relations. Older low-level arguments such as
        ``displacement_bcs`` and ``projections`` remain supported for explicit
        formula-level scripts.
        """

        prescribed_values = tuple(prescribed)
        active_constraints = tuple(constraints) + tuple(projections)
        if time is not None:
            for item in prescribed_values:
                if hasattr(item, "update"):
                    item.update(time)
            for update in update_prescribed_values:
                update(time)
        self.predict_displacement(dt)
        displacement_bcs = _collect_bcs(prescribed_values, displacement_bcs)
        prescribed_kinematics = _prescribed_kinematics(
            prescribed_values,
            time=time,
            dt=dt,
        )
        for dof in _owned_dirichlet_dofs(displacement_bcs):
            prescribed_kinematics.setdefault(int(dof), (0.0, 0.0, 0.0))
        if displacement_bcs:
            constraint_api.apply_dirichlet_bcs(self.state.u_next, displacement_bcs)
        _apply_constraints(active_constraints, self.state.u_next)
        self.update_displacement()
        self.update_midstep_velocity(dt)
        _assign_prescribed_component(
            self.state.v_mid,
            prescribed_kinematics,
            component=0,
        )
        _apply_constraints(active_constraints, self.state.v_mid)
        residual = operators.assemble_vector(residual_operator)
        try:
            self.solve_acceleration(residual)
        finally:
            residual.destroy()
        _assign_prescribed_component(
            self.state.a_next,
            prescribed_kinematics,
            component=1,
        )
        _apply_constraints(active_constraints, self.state.a_next)
        self.update_velocity(dt)
        _assign_prescribed_component(
            self.state.v_next,
            prescribed_kinematics,
            component=2,
        )
        _apply_constraints(active_constraints, self.state.v_next)
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


def _collect_bcs(prescribed, displacement_bcs) -> list:
    result = []
    if displacement_bcs:
        result.extend(displacement_bcs)
    for item in prescribed:
        if hasattr(item, "bcs"):
            result.extend(item.bcs)
        elif hasattr(item, "bc"):
            result.append(item.bc)
    return result


def _owned_dirichlet_dofs(bcs) -> np.ndarray:
    """Return unique owned scalar dofs constrained by backend Dirichlet BCs."""

    selected = []
    for bc in bcs:
        indices, first_ghost = bc.dof_indices()
        selected.extend(np.asarray(indices[:first_ghost], dtype=np.int64).tolist())
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.asarray(selected, dtype=np.int64))


def project_homogeneous_kinematics(
    field,
    *,
    prescribed: Iterable[object] = (),
    constraints: Iterable[object] = (),
) -> None:
    """Project a velocity or acceleration onto active kinematic constraints.

    Strong prescribed displacement degrees of freedom receive zero velocity
    or acceleration at a held step transition.  Other reusable constraint
    objects, such as an explicit periodic projection, are then applied using
    the same path as the central-difference integrator.
    """

    function = fields.unwrap(field)
    bcs = _collect_bcs(tuple(prescribed), None)
    selected = _owned_dirichlet_dofs(bcs)
    if selected.size:
        function.x.array[selected] = 0.0
        function.x.scatter_forward()
    _apply_constraints(tuple(constraints), field)


def _prescribed_kinematics(prescribed, *, time, dt: float):
    """Resolve midpoint velocity, acceleration, and whole-step velocity.

    Ordinary Dirichlet data are stationary.  Amplitude-driven data are
    differentiated from their declared scalar history with centered finite
    differences.  This keeps velocity and acceleration compatible with a
    moving support instead of applying displacement while leaving inertial
    state unconstrained.
    """

    resolved: dict[int, tuple[float, float, float]] = {}
    selected_time = 0.0 if time is None else float(time)
    h = 0.5 * float(dt)
    for item in prescribed:
        bcs = []
        if hasattr(item, "bcs"):
            bcs.extend(item.bcs)
        elif hasattr(item, "bc"):
            bcs.append(item.bc)
        amplitude = getattr(item, "amplitude", None)
        if amplitude is None:
            values = (0.0, 0.0, 0.0)
        else:
            def derivative(at):
                return (amplitude(at + h) - amplitude(at - h)) / (2.0 * h)

            midpoint_velocity = derivative(selected_time - 0.5 * dt)
            whole_velocity = derivative(selected_time)
            acceleration = (
                amplitude(selected_time + h)
                - 2.0 * amplitude(selected_time)
                + amplitude(selected_time - h)
            ) / h**2
            values = (
                float(midpoint_velocity),
                float(acceleration),
                float(whole_velocity),
            )
        for dof in _owned_dirichlet_dofs(bcs):
            previous = resolved.get(int(dof))
            if previous is not None and not np.allclose(previous, values):
                raise ValueError(
                    f"Conflicting prescribed kinematics at scalar dof {int(dof)}."
                )
            resolved[int(dof)] = values
    return resolved


def _assign_prescribed_component(field, kinematics, *, component: int) -> None:
    if not kinematics:
        return
    function = fields.unwrap(field)
    for dof, values in kinematics.items():
        function.x.array[dof] = values[component]
    function.x.scatter_forward()


def _apply_constraints(constraints, field) -> None:
    for item in constraints:
        if hasattr(item, "apply"):
            item.apply(field)
        elif callable(item):
            item(field)
        elif hasattr(item, "periodic"):
            _apply_constraints(item.periodic, field)
