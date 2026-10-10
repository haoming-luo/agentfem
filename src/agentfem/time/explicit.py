# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit dynamics time-integration workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from time import perf_counter

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
    last_residual_owned: np.ndarray | None = field(
        default=None,
        init=False,
        repr=False,
    )

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

        increment_started = perf_counter()
        ledger = getattr(self, "performance", None)
        prescribed_values = tuple(prescribed)
        active_constraints = tuple(constraints) + tuple(projections)
        kinematics_started = perf_counter()
        from ..provenance import collective_call

        def prepare_kinematics():
            # Time histories are rank-local inputs. Deliver a failed amplitude
            # or conflicting boundary history before any field ghost exchange.
            if time is not None:
                for item in prescribed_values:
                    if hasattr(item, "update"):
                        item.update(time)
                for update in update_prescribed_values:
                    update(time)
            bcs = _collect_bcs(prescribed_values, displacement_bcs)
            kinematics = _prescribed_kinematics(
                prescribed_values, time=time, dt=dt,
                stationary_dofs=_owned_dirichlet_dofs(bcs),
            )
            return bcs, kinematics

        displacement_bcs, prescribed_kinematics = collective_call(
            prepare_kinematics,
            comm=fields.unwrap(self.state.u).function_space.mesh.comm,
            label="Explicit prescribed kinematics",
        )
        self.predict_displacement(dt)
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
        if ledger is not None:
            ledger.add(
                "kinematic_prediction_and_constraints",
                perf_counter() - kinematics_started,
            )
        self.last_residual_owned = None
        residual_started = perf_counter()
        residual = operators.assemble_vector(residual_operator)
        if ledger is not None:
            ledger.add("residual_assembly", perf_counter() - residual_started)
        owned = dofs.owned_size(self.state.a_next)
        self.last_residual_owned = np.asarray(
            residual.array[:owned],
            dtype=float,
        ).copy()
        state_started = perf_counter()
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
        if ledger is not None:
            ledger.add("acceleration_and_state_update", perf_counter() - state_started)
            ledger.add("explicit_increment", perf_counter() - increment_started)

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
        selected.append(np.asarray(indices[:first_ghost], dtype=np.int64))
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(selected))


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


@dataclass(frozen=True)
class _PrescribedKinematics:
    dofs: np.ndarray
    values: np.ndarray


def _prescribed_kinematics(prescribed, *, time, dt: float, stationary_dofs=()):
    """Resolve midpoint velocity, acceleration, and whole-step velocity.

    Ordinary Dirichlet data are stationary.  Amplitude-driven data are
    differentiated from their declared scalar history with centered finite
    differences.  This keeps velocity and acceleration compatible with a
    moving support instead of applying displacement while leaving inertial
    state unconstrained.
    """

    indices, components = [], []
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
            def derivative(at, selected_amplitude=amplitude):
                return (
                    selected_amplitude(at + h) - selected_amplitude(at - h)
                ) / (2.0 * h)

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
        if not np.isfinite(values).all():
            raise ValueError("Prescribed kinematics must be finite.")
        selected = _owned_dirichlet_dofs(bcs)
        indices.append(selected)
        components.append(np.broadcast_to(values, (len(selected), 3)))
    index = np.concatenate(indices) if indices else np.empty(0, dtype=np.int64)
    values = np.concatenate(components) if components else np.empty((0, 3))
    if index.size:
        # Stable order preserves the previous last-declaration-wins convention
        # for compatible duplicate histories, without Python work per DOF.
        order = np.argsort(index, kind="stable")
        index, values = index[order], values[order]
        duplicate = index[1:] == index[:-1]
        shared = np.flatnonzero(duplicate)
        compatible = np.all(np.isclose(values[shared], values[shared + 1]), axis=1)
        conflicts = shared[~compatible]
        if conflicts.size:
            raise ValueError(
                f"Conflicting prescribed kinematics at scalar dof {int(index[conflicts[0]])}."
            )
        keep = np.r_[~duplicate, True]
        index, values = index[keep], values[keep]
    stationary = np.setdiff1d(np.asarray(stationary_dofs, dtype=np.int64), index)
    if stationary.size:
        index = np.concatenate((index, stationary))
        values = np.concatenate((values, np.zeros((len(stationary), 3))))
    return _PrescribedKinematics(index, values)


def _assign_prescribed_component(field, kinematics, *, component: int) -> None:
    function = fields.unwrap(field)
    function.x.array[kinematics.dofs] = kinematics.values[:, component]
    # Every partition participates, including ranks without prescribed DOFs.
    function.x.scatter_forward()


def _apply_constraints(constraints, field) -> None:
    for item in constraints:
        if hasattr(item, "apply"):
            item.apply(field)
        elif callable(item):
            item(field)
        elif hasattr(item, "periodic"):
            _apply_constraints(item.periodic, field)
