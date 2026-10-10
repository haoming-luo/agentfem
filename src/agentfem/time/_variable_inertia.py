# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private serial numerical-inertia State, with an explicit external policy.

The policy must be deterministic and stateless; history belongs here, not in a
hidden callback closure. The Procedure owns when a proposal is evaluated; this object owns inertia and
its artificial momentum/energy inputs. No stable-step estimator is implied.
Prescribed nodal motion and MPC are not yet admitted with variable inertia.
"""

import numpy as np

from ._scaled_inertia import FixedScaledInertia, _checked_nodal_factors
from ..provenance import content_fingerprint


class VariableScaledInertia:
    def __init__(
        self,
        physical_mass_diagonal,
        *,
        policy,
        policy_identity,
        maximum_added_mass_ratio,
        block_size=3,
    ):
        if (
            not callable(policy)
            or not isinstance(policy_identity, dict)
            or not policy_identity
        ):
            raise ValueError(
                "Variable inertia requires a callable policy and explicit identity."
            )
        limit = float(maximum_added_mass_ratio)
        if not np.isfinite(limit) or limit < 0:
            raise ValueError("Added-mass budget must be finite and nonnegative.")
        if block_size not in (2, 3):
            raise ValueError(
                "Variable mechanical inertia requires two or three components."
            )
        self._initial = FixedScaledInertia(
            physical_mass_diagonal, factor=1.0, block_size=block_size
        )
        self._current = self._initial
        self.policy = policy
        self.maximum_added_mass_ratio = limit
        self.identity = content_fingerprint(
            {
                "schema": "agentfem.variable-scaled-inertia.v1",
                "physical": self._initial.identity,
                "policy": policy_identity,
                "maximum_added_mass_ratio": limit,
            }
        )
        self.energy_input = 0.0
        self.momentum_input = np.zeros(block_size)
        self.angular_momentum_input = np.zeros(3)
        self.last_increment = 0
        self.update_count = 0

    @property
    def physical_mass_diagonal(self):
        return self._initial.physical_mass_diagonal

    @property
    def mass_diagonal(self):
        return self._current.mass_diagonal

    @property
    def mass(self):
        return self.mass_diagonal

    @property
    def inv_mass(self):
        return self._current.inv_mass

    @property
    def factors(self):
        return self._current.factors

    def checkpoint_identity(self):
        return {
            "schema": "agentfem.variable-scaled-inertia.v1",
            "identity": self.identity,
            "portability": "same_serial_nodal_order",
            "fixed": False,
        }

    def validate_state(self, state):
        self._initial.validate_state(state)

    def validate_procedure(self, step):
        self.validate_state(step.state)
        self.validate_accepted_station(step.completed_steps)
        if step.prescribed or step.constraints:
            raise NotImplementedError(
                "Variable inertia with prescribed nodal motion or MPC is not admitted."
            )

    def prepare_increment(self, state, *, increment, time, dt):
        """Update before prediction, preserving velocity and accepted force.

        At unchanged configuration, a_new = M_old/M_new * a_old. The velocity
        is not rescaled to hide the added momentum or kinetic energy: both are
        recorded as numerical inputs, never physical external work.
        """
        from ..fields import unwrap

        self.validate_state(state)
        if (
            increment != self.last_increment + 1
            or not np.isfinite(time)
            or not np.isfinite(dt)
            or dt <= 0
        ):
            raise ValueError(
                "Variable inertia requires consecutive accepted increments and valid time."
            )
        displacement = unwrap(state.u).x.array.copy()
        velocity = unwrap(state.v).x.array.copy()
        displacement.flags.writeable = velocity.flags.writeable = False
        proposal = self.policy(
            displacement=displacement,
            velocity=velocity,
            current_factors=self.factors.copy(),
            increment=increment,
            time=float(time),
            dt=float(dt),
        )
        factors = _checked_nodal_factors(proposal, len(self.factors))
        if np.array_equal(factors, self.factors):
            # The common no-change proposal must not copy/hash several large
            # mass diagonals every increment. Validation still precedes reuse.
            self.last_increment = int(increment)
            return
        candidate = FixedScaledInertia(
            self.physical_mass_diagonal,
            factor=factors,
            block_size=self._initial.block_size,
        )
        if np.any(candidate.factors < self.factors):
            raise ValueError(
                "Variable numerical inertia cannot decrease accepted mass."
            )
        if candidate.summary()["added_mass_ratio"] > self.maximum_added_mass_ratio:
            raise ValueError(
                "Variable numerical inertia exceeds its added-mass budget."
            )
        delta = candidate.mass_diagonal - self.mass_diagonal
        if not np.any(delta != 0):
            self.last_increment = int(increment)
            return
        energy = self.energy_input + float(0.5 * np.dot(delta, velocity * velocity))
        momentum = self.momentum_input + (delta * velocity).reshape(
            -1, self._initial.block_size
        ).sum(axis=0)
        block = self._initial.block_size
        positions = unwrap(state.u).function_space.tabulate_dof_coordinates().copy()
        positions[:, :block] += displacement.reshape(-1, block)
        added_momentum = np.zeros_like(positions)
        added_momentum[:, :block] = (delta * velocity).reshape(-1, block)
        angular = self.angular_momentum_input + np.cross(positions, added_momentum).sum(
            axis=0
        )
        acceleration = unwrap(state.a).x.array * (
            self.mass_diagonal / candidate.mass_diagonal
        )
        if (
            not np.isfinite(energy)
            or not np.isfinite(momentum).all()
            or not np.isfinite(angular).all()
            or not np.isfinite(acceleration).all()
        ):
            raise ValueError("Variable inertia update produced a nonfinite state.")
        # Candidate validation is complete before either mass or fields change.
        unwrap(state.a).x.array[:] = acceleration
        self._current = candidate
        self.energy_input, self.momentum_input = energy, momentum
        self.angular_momentum_input = angular
        self.last_increment = int(increment)
        self.update_count += int(np.any(delta != 0))

    def snapshot(self):
        return {
            "schema": "agentfem.variable-scaled-inertia-state.v1",
            "identity": self.identity,
            "factors": self.factors.copy(),
            "energy_input": self.energy_input,
            "momentum_input": self.momentum_input.copy(),
            "angular_momentum_input": self.angular_momentum_input.copy(),
            "last_increment": self.last_increment,
            "update_count": self.update_count,
        }

    def validate_accepted_station(self, increment):
        if self.last_increment != increment:
            raise ValueError(
                "Numerical inertia and mechanical accepted increments differ."
            )

    def restore(self, record):
        keys = {
            "schema",
            "identity",
            "factors",
            "energy_input",
            "momentum_input",
            "angular_momentum_input",
            "last_increment",
            "update_count",
        }
        if (
            not isinstance(record, dict)
            or set(record) != keys
            or record["schema"] != "agentfem.variable-scaled-inertia-state.v1"
            or record["identity"] != self.identity
        ):
            raise ValueError("Variable inertia checkpoint identity or schema differs.")
        candidate = FixedScaledInertia(
            self.physical_mass_diagonal,
            factor=record["factors"],
            block_size=self._initial.block_size,
        )
        energy = record["energy_input"]
        momentum = np.asarray(record["momentum_input"])
        angular = np.asarray(record["angular_momentum_input"])
        last, count = record["last_increment"], record["update_count"]
        if (
            not np.isscalar(energy)
            or np.iscomplexobj(energy)
            or not np.isfinite(energy)
            or energy < 0
            or momentum.shape != (self._initial.block_size,)
            or np.iscomplexobj(momentum)
            or not np.isfinite(momentum).all()
            or angular.shape != (3,)
            or np.iscomplexobj(angular)
            or not np.isfinite(angular).all()
            or isinstance(last, bool)
            or not isinstance(last, int)
            or last < 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= last
            or candidate.summary()["added_mass_ratio"] > self.maximum_added_mass_ratio
        ):
            raise ValueError("Invalid variable inertia checkpoint state.")
        if count == 0 and (
            np.any(candidate.factors != 1)
            or energy != 0
            or np.any(momentum != 0)
            or np.any(angular != 0)
        ):
            raise ValueError(
                "Unchanged inertia cannot carry added mass, momentum or energy."
            )
        self._current = candidate
        self.energy_input = float(energy)
        self.momentum_input = momentum.astype(float, copy=True)
        self.angular_momentum_input = angular.astype(float, copy=True)
        self.last_increment, self.update_count = last, count

    def summary(self):
        return {
            **self._current.summary(),
            **self.checkpoint_identity(),
            "update_count": self.update_count,
            "last_increment": self.last_increment,
            "numerical_inertia_energy_input": self.energy_input,
            "numerical_inertia_momentum_input": self.momentum_input.tolist(),
            "numerical_inertia_angular_momentum_input": self.angular_momentum_input.tolist(),
            "automatic_stability_certificate": False,
        }
