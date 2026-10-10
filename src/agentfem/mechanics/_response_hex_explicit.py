# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private response-only material adapter to the existing explicit Procedure.

Caller increment ceilings are resolution choices, NOT proven stability bounds.
Internal work remains work; no recoverable energy or tangent is manufactured.
"""

import numpy as np
from hashlib import sha256

from ..provenance import content_fingerprint
from ..state import field_transaction


class ResponseHexWorkMonitor:
    """Accepted work balance ingredients, not a thermodynamic decomposition."""

    def __init__(self, residual):
        self.residual = residual

    def evaluate(self, *, displacement, velocity):
        from ..fields import unwrap
        from ..diagnostics import kinetic_energy

        r = self.residual
        if r._trial is not None or r.time != r.accepted_time:
            raise RuntimeError("Work monitor requires an accepted material state.")
        if unwrap(displacement) is not r.internal.displacement:
            raise ValueError("Work monitor displacement identity differs.")
        if not np.array_equal(r.internal.displacement.x.array, r.accepted_displacement):
            raise RuntimeError("Work monitor requires the accepted configuration.")
        kinetic = kinetic_energy(r.internal.mass_diagonal, velocity)
        return {
            "kinetic_energy": kinetic,
            "material_internal_work": r.internal_work,
            "hourglass_energy": r.hourglass_energy,
            "interface_stored_energy": r.interface_energy,
            "artificial_bulk_viscosity_work": r.viscosity_work,
            "accounted_internal_kinetic_energy": kinetic
            + r.internal_work
            + r.hourglass_energy
            + r.interface_energy
            + r.viscosity_work,
        }


class ResponseHexExplicitResidual:
    def __init__(
        self,
        internal,
        material,
        *,
        time_increment_ceiling,
        material_identity,
        stress_free_initial_state,
        cohesive=None,
        bulk_viscosity=None,
    ):
        from ..constitutive._explicit_material_driver import ExplicitQuadratureResponse

        if internal.comm.size != 1 or not isinstance(
            internal.response, ExplicitQuadratureResponse
        ):
            raise NotImplementedError("Response-only explicit lifecycle is serial.")
        if stress_free_initial_state is not True:
            raise ValueError(
                "An explicit stress-free initial-state declaration is required."
            )
        ceiling = float(time_increment_ceiling)
        if not np.isfinite(ceiling) or ceiling <= 0:
            raise ValueError("A finite positive caller increment ceiling is required.")
        if not isinstance(material_identity, dict) or not material_identity:
            raise ValueError("Explicit material identity is required for recovery.")
        if np.any(internal.displacement.x.array != 0):
            raise ValueError("Response-only initialization requires zero displacement.")
        initial = material.state_schema.initial_state()
        committed = internal.response.state.committed_state_vectors()
        if not np.array_equal(committed, np.broadcast_to(initial, committed.shape)):
            raise ValueError("Response-only initialization requires virgin state.")
        self.internal, self.material = internal, material
        self.bulk_viscosity = bulk_viscosity
        self._secant_velocity = None
        if bulk_viscosity is not None:
            from ..elements._bulk_viscosity_policy import BulkViscosityPolicy
            from dolfinx import fem

            if not isinstance(bulk_viscosity, BulkViscosityPolicy):
                raise TypeError("Expected a BulkViscosityPolicy.")
            self._secant_velocity = fem.Function(internal.displacement.function_space)
        self.cohesive = cohesive
        if cohesive is not None:
            from .._elastic_cohesive import ElasticCohesiveLaw
            from .._nonmatching_force import NonmatchingCohesiveForce

            if not isinstance(cohesive, NonmatchingCohesiveForce):
                raise TypeError("Expected existing nonmatching interface force.")
            law = cohesive.assembler.law
            if cohesive.displacement is not internal.displacement:
                raise ValueError("Body and interface displacement differ.")
            if not isinstance(law, ElasticCohesiveLaw) or not (
                law.normal_stiffness
                == law.tangential_stiffness
                == law.second_tangential_stiffness
            ):
                raise NotImplementedError(
                    "Response-only finite bond requires isotropic elasticity."
                )
            method = cohesive.assembler.pairing.method
            if not (
                method.startswith("coplanar-")
                or method == "original-node-q1-isotropic-elastic"
            ):
                raise NotImplementedError("Unreviewed finite reference pairing method.")
        self.checkpoint_state_required = True
        self.comm = internal.comm
        self.ceiling = ceiling
        digest = sha256(
            content_fingerprint(
                {
                    "material": material_identity,
                    "state": repr(material.state_schema),
                    "ceiling": ceiling,
                    "stress_free_initial_state": True,
                    "bulk_viscosity": None
                    if bulk_viscosity is None
                    else bulk_viscosity.summary(),
                }
            ).encode()
        )
        for array in (
            internal.cells.coordinates,
            internal.cell_nodes,
            internal.mass_diagonal,
            internal.cells.hourglass_coefficient,
        ):
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        self.identity = digest.hexdigest()
        if cohesive is not None:
            digest.update(content_fingerprint(cohesive.snapshot()).encode())
            digest.update(cohesive.negative_dofs.tobytes())
            digest.update(cohesive.positive_dofs.tobytes())
            self.identity = digest.hexdigest()
        self.accepted_gradient = np.tile(np.eye(3), (len(internal.cell_nodes), 1, 1))
        self.accepted_time = self.time = 0.0
        self.internal_work = 0.0
        self.hourglass_energy = 0.0
        self.interface_energy = 0.0
        self.viscosity_work = 0.0
        self.damping_rate_bound = 0.0
        self._trial = None
        self.accepted_force = np.zeros_like(internal.mass_diagonal)
        self.accepted_displacement = internal.displacement.x.array.copy()

    def validate_time_increment(self, dt):
        if not np.isfinite(dt) or dt <= 0 or dt > self.ceiling * (1 + 1e-12):
            raise ValueError("Increment exceeds the declared resolution ceiling.")

    def update_time(self, time):
        if not np.isfinite(time) or time < self.accepted_time:
            raise ValueError("Invalid response-only time.")
        self.time = float(time)

    def assemble_vector(self):
        try:
            return self._assemble_vector()
        except BaseException:
            self.rollback()
            raise

    def _assemble_vector(self):
        dt = self.time - self.accepted_time
        self.validate_time_increment(dt)
        self._trial = None
        with self.internal.explicit_trial_evaluation(
            self.material,
            deformation_gradient_old=self.accepted_gradient,
            time=self.accepted_time,
            time_increment=dt,
        ) as (vector, trial):
            viscosity_work = self.viscosity_work
            damping_bound = 0.0
            if self.bulk_viscosity is not None:
                increment = (
                    self.internal.displacement.x.array - self.accepted_displacement
                )
                self._secant_velocity.x.array[:] = increment / dt
                damping, viscous = self.internal.assemble_bulk_viscosity(
                    self._secant_velocity,
                    **self.bulk_viscosity.options(
                        np.linalg.det(trial.deformation_gradient)
                    ),
                )
                try:
                    contribution = float(damping.array @ increment)
                    expected = dt * float(viscous.artificial_power.sum())
                    roundoff = (
                        128
                        * np.finfo(float).eps
                        * float(np.sum(np.abs(damping.array * increment)))
                    )
                    if (
                        expected < 0
                        or abs(contribution - expected) > roundoff + 1e-10 * expected
                    ):
                        raise ValueError(
                            "Artificial damping work is not conjugate to the increment."
                        )
                    damping_bound = float(
                        np.max(viscous.mass_scaled_damping_bound, initial=0.0)
                    )
                    # Conditional fixed-coefficient estimate: ceiling is still
                    # supplied, NOT a proven material/contact spectral bound.
                    omega = 2 / self.ceiling
                    limit = 4 / (np.hypot(damping_bound, 2 * omega) + damping_bound)
                    if dt > limit * (1 + 1e-12):
                        raise ValueError(
                            "Increment exceeds the damping-adjusted resolution ceiling."
                        )
                    vector.axpy(1.0, damping)
                    # Equivalent positive local-power contraction avoids
                    # cancellation of opposite nodal forces in rigid motion.
                    viscosity_work += expected
                finally:
                    damping.destroy()
            interface_energy = 0.0
            if self.cohesive is not None:
                try:
                    interface_energy = self.cohesive.add_to_vector(vector).stored_energy
                except BaseException:
                    self.cohesive.rollback()
                    raise
            if not np.isfinite(vector.array).all():
                raise ValueError("Nonfinite response-only force.")
            work = self.internal_work + float(
                self.internal.cells.volume
                @ trial.material_response.internal_work_density_increment
            )
            hg = float(np.sum(trial.element_response.hourglass_energy))
            if not np.isfinite(
                [work, hg, interface_energy, viscosity_work, damping_bound]
            ).all():
                raise ValueError("Nonfinite response-only work.")
            self._trial = (
                trial,
                work,
                hg,
                self.time,
                self.internal.displacement.x.array.copy(),
                vector.array.copy(),
                interface_energy,
                viscosity_work,
                damping_bound,
            )
        return vector

    def commit(self):
        if self._trial is None:
            raise RuntimeError("No response-only trial to accept.")
        (
            trial,
            work,
            hg,
            time,
            displacement,
            force,
            interface_energy,
            viscosity_work,
            damping_bound,
        ) = self._trial
        if time != self.time or not np.array_equal(
            displacement, self.internal.displacement.x.array
        ):
            raise RuntimeError("Configuration changed after response-only trial.")
        self.internal.response.commit()
        if self.cohesive is not None:
            self.cohesive.commit()
        self.accepted_gradient = trial.deformation_gradient.copy()
        self.accepted_time = time
        self.internal_work, self.hourglass_energy = work, hg
        self.accepted_force = force
        self.interface_energy = interface_energy
        self.viscosity_work = viscosity_work
        self.damping_rate_bound = damping_bound
        self.accepted_displacement = displacement
        self._trial = None

    def assemble_accepted_vector(self):
        if (
            self._trial is not None
            or self.time != self.accepted_time
            or not np.array_equal(
                self.internal.displacement.x.array, self.accepted_displacement
            )
        ):
            raise RuntimeError("Accepted force requires the accepted configuration.")
        vector = self.internal.displacement.x.petsc_vec.duplicate()
        vector.array[:] = self.accepted_force
        return vector

    def rollback(self):
        self.internal.response.rollback()
        if self.cohesive is not None:
            self.cohesive.rollback()
        self.time = self.accepted_time
        self._trial = None

    def summary(self):
        return {
            "kind": "private_response_hex_explicit",
            "identity": self.identity,
            "increment_ceiling": self.ceiling,
            "stability_certified": False,
            "material_internal_work": self.internal_work,
            "artificial_bulk_viscosity_work": self.viscosity_work,
            "bulk_viscosity": None
            if self.bulk_viscosity is None
            else self.bulk_viscosity.summary(),
            "stored_energy_decomposition_available": False,
            "execution_scope": "serial_development",
        }

    def _fields(self):
        response = self.internal.response
        result = dict(response.trial_fields())
        result.update(
            {
                "committed_state_" + k: v.function
                for k, v in response.state.committed.items()
            }
        )
        result.update(
            {
                "committed_response_" + k: v.function
                for k, v in response.response_transaction.committed.items()
            }
        )
        return result

    def transaction_snapshot(self):
        if self._trial is not None:
            raise RuntimeError("Cannot save an unaccepted response-only trial.")
        return {
            "schema": "agentfem.private-response-hex.v4",
            "identity": self.identity,
            "time": self.accepted_time,
            "gradient": self.accepted_gradient.copy(),
            "internal_work": self.internal_work,
            "hourglass_energy": self.hourglass_energy,
            "interface_energy": self.interface_energy,
            "viscosity_work": self.viscosity_work,
            "damping_rate_bound": self.damping_rate_bound,
            "cohesive": None if self.cohesive is None else self.cohesive.snapshot(),
            "accepted_force": self.accepted_force.copy(),
            "accepted_displacement": self.accepted_displacement.copy(),
            "fields": {k: v.x.array.copy() for k, v in self._fields().items()},
        }

    checkpoint_snapshot = transaction_snapshot

    def checkpoint_capabilities(self):
        from ..checkpointing import CheckpointCapabilities

        return CheckpointCapabilities(
            schemas=("agentfem.transient-checkpoint.v7",),
            boundary="accepted_step",
            payload_scope="full_restart_state",
            atomic_publication=True,
            rank_count_portability="unsupported",
            state_components=(
                "material history and accepted stress/work",
                "accepted deformation gradient",
                "accumulated internal work",
            ),
            identity_scope=(
                "material and schema",
                "geometry and mass",
                "hourglass coefficients",
                "caller increment ceiling",
            ),
            limitations=(
                "serial same-partition response-only development route",
                "caller ceiling is not a certified stability bound",
            ),
            evidence=(
                "serial interrupted/continuous regression",
                "invalid state atomic rejection",
            ),
        )

    def snapshot(self):
        record = self.transaction_snapshot()
        record["gradient"] = record["gradient"].tolist()
        for name in ("accepted_force", "accepted_displacement"):
            record[name] = record[name].tolist()
        record["fields"] = {k: v.tolist() for k, v in record["fields"].items()}
        return record

    def restore(self, record):
        if (
            record.get("schema") != "agentfem.private-response-hex.v4"
            or record.get("identity") != self.identity
        ):
            raise ValueError("Response-only checkpoint identity mismatch.")
        fields = self._fields()
        if set(record["fields"]) != set(fields):
            raise ValueError("Response-only checkpoint fields mismatch.")
        gradient = np.asarray(record["gradient"])
        if (
            np.iscomplexobj(gradient)
            or gradient.shape != self.accepted_gradient.shape
            or not np.isfinite(gradient).all()
            or np.any(np.linalg.det(gradient) <= 0)
        ):
            raise ValueError("Invalid response-only checkpoint gradient.")
        values = [
            record[k]
            for k in (
                "time",
                "internal_work",
                "hourglass_energy",
                "interface_energy",
                "viscosity_work",
                "damping_rate_bound",
            )
        ]
        if (
            any(
                not np.isscalar(v) or not np.isreal(v) or not np.isfinite(v)
                for v in values
            )
            or values[0] < 0
            or values[2] < 0
            or values[3] < 0
            or values[4] < 0
            or values[5] < 0
        ):
            raise ValueError("Invalid response-only checkpoint scalars.")
        arrays = {k: np.asarray(record["fields"][k]) for k in fields}
        force = np.asarray(record["accepted_force"])
        displacement = np.asarray(record["accepted_displacement"])
        if any(
            np.iscomplexobj(a) or a.shape != ref.shape or not np.isfinite(a).all()
            for a, ref in (
                (force, self.accepted_force),
                (displacement, self.accepted_displacement),
            )
        ):
            raise ValueError("Invalid accepted force/displacement checkpoint.")
        if any(
            np.iscomplexobj(v)
            or v.shape != fields[k].x.array.shape
            or not np.isfinite(v).all()
            for k, v in arrays.items()
        ):
            raise ValueError("Invalid response-only checkpoint field values.")
        for name in self.internal.response.state.committed:
            if not np.array_equal(
                arrays["state_" + name], arrays["committed_state_" + name]
            ):
                raise ValueError("Checkpoint contains unaccepted material history.")
        for name in self.internal.response.response_transaction.committed:
            if not np.array_equal(arrays[name], arrays["committed_response_" + name]):
                raise ValueError("Checkpoint contains unaccepted material response.")
        if (record.get("cohesive") is None) != (self.cohesive is None):
            raise ValueError("Response-only checkpoint interface differs.")
        previous_interface = None if self.cohesive is None else self.cohesive.snapshot()
        try:
            if self.cohesive is not None:
                self.cohesive.rollback()
                self.cohesive.restore(record["cohesive"])
            with field_transaction(**fields):
                for k, v in fields.items():
                    v.x.array[:] = arrays[k]
                    v.x.scatter_forward()
        except BaseException:
            if self.cohesive is not None:
                self.cohesive.restore(previous_interface)
            raise
        self.accepted_gradient = gradient.copy()
        self.accepted_time = self.time = float(values[0])
        (
            self.internal_work,
            self.hourglass_energy,
            self.interface_energy,
            self.viscosity_work,
            self.damping_rate_bound,
        ) = map(float, values[1:])
        self.accepted_force = force.copy()
        self.accepted_displacement = displacement.copy()
        self._trial = None
