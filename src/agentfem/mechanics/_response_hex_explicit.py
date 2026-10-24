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
from .._resident_field_records import _freeze


class ResponseHexWorkMonitor:
    """Accepted work balance ingredients, not a thermodynamic decomposition."""

    def __init__(self, residual, *, numerical_inertia=None):
        self.residual = residual
        self.numerical_inertia = numerical_inertia
        if numerical_inertia is not None:
            from ..time._scaled_inertia import FixedScaledInertia
            from ..time._variable_inertia import VariableScaledInertia

            if not isinstance(
                numerical_inertia, (FixedScaledInertia, VariableScaledInertia)
            ) or not np.array_equal(
                numerical_inertia.physical_mass_diagonal,
                residual.internal.mass_diagonal,
            ):
                raise ValueError(
                    "Numerical inertia must retain this body's physical mass."
                )

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
        physical_kinetic = kinetic_energy(r.internal.mass_diagonal, velocity)
        kinetic = (
            physical_kinetic
            if self.numerical_inertia is None
            else kinetic_energy(self.numerical_inertia.mass_diagonal, velocity)
        )
        return {
            "kinetic_energy": kinetic,
            **(
                {
                    "numerical_inertia_energy_input": self.numerical_inertia.energy_input,
                    "numerical_inertia_momentum_input_norm": float(
                        np.linalg.norm(self.numerical_inertia.momentum_input)
                    ),
                    "numerical_inertia_angular_momentum_input_norm": float(
                        np.linalg.norm(self.numerical_inertia.angular_momentum_input)
                    ),
                }
                if hasattr(self.numerical_inertia, "energy_input")
                else {}
            ),
            **(
                {
                    "physical_mass_kinetic_energy": physical_kinetic,
                    "added_inertia_kinetic_energy": kinetic - physical_kinetic,
                }
                if self.numerical_inertia is not None
                else {}
            ),
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
    # Experimental execution policy, not a public solver option. Real-model
    # screens have not established a throughput benefit over ordinary copies.
    _resident_records_enabled = False

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
        require_material_increment_limit=False,
        numerical_inertia=None,
        material_restart_identity=None,
    ):
        from ..constitutive._explicit_material_driver import ExplicitQuadratureResponse
        from ..constitutive._explicit_material import explicit_increment_policy_identity

        if not isinstance(require_material_increment_limit, bool):
            raise TypeError("require_material_increment_limit must be a bool.")
        self.material_increment_policy = explicit_increment_policy_identity(material)
        if require_material_increment_limit and self.material_increment_policy is None:
            raise ValueError(
                "This response route requires a material increment policy."
            )

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
        if material_restart_identity is not None and (
            not isinstance(material_restart_identity, dict) or not material_restart_identity
        ):
            raise ValueError("Explicit material restart identity must be a nonempty dictionary.")
        if np.any(internal.displacement.x.array != 0):
            raise ValueError("Response-only initialization requires zero displacement.")
        initial = material.state_schema.initial_state()
        committed = internal.response.state.committed_state_vectors()
        if not np.array_equal(committed, np.broadcast_to(initial, committed.shape)):
            raise ValueError("Response-only initialization requires virgin state.")
        self.internal, self.material = internal, material
        self.numerical_inertia = numerical_inertia
        if numerical_inertia is not None:
            from ..time._scaled_inertia import FixedScaledInertia
            from ..time._variable_inertia import VariableScaledInertia

            if not isinstance(
                numerical_inertia, (FixedScaledInertia, VariableScaledInertia)
            ):
                raise TypeError(
                    "Response damping requires a reviewed numerical inertia owner."
                )
            if not np.array_equal(
                numerical_inertia.physical_mass_diagonal, internal.mass_diagonal
            ):
                raise ValueError(
                    "Response numerical inertia has different physical mass."
                )
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
        identity_record = {
            "material": material_identity,
            "state": repr(material.state_schema),
            "ceiling": ceiling,
            "stress_free_initial_state": True,
            **({"numerical_inertia": numerical_inertia.checkpoint_identity()}
               if numerical_inertia is not None else {}),
            **({"material_increment_policy": self.material_increment_policy}
               if self.material_increment_policy is not None else {}),
            "bulk_viscosity": None if bulk_viscosity is None else bulk_viscosity.summary(),
        }
        digest = sha256(content_fingerprint(identity_record).encode())
        restart_digest = (None if material_restart_identity is None else sha256(
            content_fingerprint({**identity_record, "material": material_restart_identity}).encode()))
        digests = (digest,) if restart_digest is None else (digest, restart_digest)
        for array in (
            internal.cells.coordinates,
            internal.cell_nodes,
            internal.mass_diagonal,
            internal.cells.hourglass_coefficient,
        ):
            layout = str((array.shape, array.dtype.str)).encode()
            payload = np.ascontiguousarray(array).tobytes()
            for item in digests:
                item.update(layout)
                item.update(payload)
        if cohesive is not None:
            interface_payloads = (content_fingerprint(cohesive.snapshot()).encode(),
                                  cohesive.negative_dofs.tobytes(), cohesive.positive_dofs.tobytes())
            for item in digests:
                for payload in interface_payloads:
                    item.update(payload)
        self.identity = digest.hexdigest()
        self.restart_identity = None if restart_digest is None else restart_digest.hexdigest()
        self.accepted_gradient = _freeze(np.tile(np.eye(3), (len(internal.cell_nodes), 1, 1)))
        self.accepted_time = self.time = 0.0
        self.internal_work = 0.0
        self.hourglass_energy = 0.0
        self.interface_energy = 0.0
        self.viscosity_work = 0.0
        self.damping_rate_bound = 0.0
        self._trial = None
        self.accepted_force = _freeze(np.zeros_like(internal.mass_diagonal))
        self.accepted_displacement = _freeze(internal.displacement.x.array)

    def validate_time_increment(self, dt):
        if not np.isfinite(dt) or dt <= 0 or dt > self.ceiling * (1 + 1e-12):
            raise ValueError("Increment exceeds the declared resolution ceiling.")

    def validate_explicit_inertia(self, mass):
        if self.numerical_inertia is not None and mass is not self.numerical_inertia:
            raise ValueError(
                "Response damping and explicit integrator must share numerical inertia."
            )

    def update_time(self, time):
        if not np.isfinite(time) or time < self.accepted_time:
            raise ValueError("Invalid response-only time.")
        self.time = float(time)

    def assemble_vector(self):
        from contextlib import nullcontext
        ledger = getattr(self, 'performance', None)
        scope = nullcontext() if ledger is None else ledger.measure('body_residual_inclusive')
        with scope:
            try:
                return self._assemble_vector()
            except BaseException:
                self.rollback()
                raise

    def _assemble_vector(self):
        from ..constitutive._explicit_material import explicit_increment_policy_identity

        if (
            explicit_increment_policy_identity(self.material)
            != self.material_increment_policy
        ):
            raise ValueError(
                "Material increment policy identity changed after construction."
            )
        dt = self.time - self.accepted_time
        self.validate_time_increment(dt)
        self._trial = None
        increment = self.internal.displacement.x.array - self.accepted_displacement
        if self.bulk_viscosity is not None:
            self._secant_velocity.x.array[:] = increment / dt
        with self.internal.explicit_trial_evaluation(
            self.material,
            deformation_gradient_old=self.accepted_gradient,
            time=self.accepted_time,
            time_increment=dt,
            velocity=self._secant_velocity,
            bulk_viscosity=self.bulk_viscosity,
        ) as (vector, trial):
            viscosity_work = self.viscosity_work
            damping_bound = 0.0
            if self.bulk_viscosity is not None:
                viscous = trial.bulk_viscosity_response
                from ..elements._bulk_viscosity import _cell_increment_work
                contribution, absolute_products = _cell_increment_work(
                    viscous.internal_force, self.internal.cell_nodes,
                    increment.reshape(-1, 3),
                )
                expected = dt * float(viscous.artificial_power.sum())
                roundoff = 128 * np.finfo(float).eps * absolute_products
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
                if self.numerical_inertia is not None:
                    # Physical density, damping force and work are unchanged.
                    damping_bound /= float(self.numerical_inertia.factors.min())
                # Conditional fixed-coefficient estimate: ceiling is still
                # supplied, NOT a proven material/contact spectral bound.
                omega = 2 / self.ceiling
                limit = 4 / (np.hypot(damping_bound, 2 * omega) + damping_bound)
                if dt > limit * (1 + 1e-12):
                    raise ValueError(
                        "Increment exceeds the damping-adjusted resolution ceiling."
                    )
                viscosity_work += expected
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
                _freeze(self.internal.displacement.x.array),
                _freeze(vector.array),
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
        # Allocate immutable accepted data before advancing material state.
        # Already owned bytes need only fresh ndarray metadata, not another copy.
        accepted_gradient = _freeze(trial.deformation_gradient)
        self.internal.response.commit()
        if self.cohesive is not None:
            self.cohesive.commit()
        self.accepted_gradient = accepted_gradient
        self.accepted_time = time
        self.internal_work, self.hourglass_energy = work, hg
        self.accepted_force = force
        self.interface_energy = interface_energy
        self.viscosity_work = viscosity_work
        self.damping_rate_bound = damping_bound
        self.accepted_displacement = displacement
        if self._resident_records_enabled:
            self._remember_accepted_fields(trial)
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
            "material_increment_policy": self.material_increment_policy,
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
        response.publish_accepted()
        result = dict(response._storage_trial_fields())
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
            "schema": ("agentfem.private-response-hex.v4" if self.restart_identity is None
                       else "agentfem.private-response-hex.v5"),
            "identity": self.identity,
            **({"restart_identity": self.restart_identity} if self.restart_identity is not None else {}),
            "time": self.accepted_time,
            "gradient": _freeze(self.accepted_gradient),
            "internal_work": self.internal_work,
            "hourglass_energy": self.hourglass_energy,
            "interface_energy": self.interface_energy,
            "viscosity_work": self.viscosity_work,
            "damping_rate_bound": self.damping_rate_bound,
            "cohesive": None if self.cohesive is None else self.cohesive.snapshot(),
            "accepted_force": _freeze(self.accepted_force),
            "accepted_displacement": _freeze(self.accepted_displacement),
            "fields": (self.internal.response._owned.snapshot()
                       if self.internal.response._owned is not None else
                       self._field_records().capture(self._fields())
                       if self._resident_records_enabled else
                       {k: v.x.array.copy() for k, v in self._fields().items()}),
        }

    def _field_records(self):
        from .._resident_field_records import ResidentFieldRecords

        if not hasattr(self, "_resident_fields"):
            self._resident_fields = ResidentFieldRecords()
        return self._resident_fields

    def _remember_accepted_fields(self, trial):
        """Seed disposable cache from owned response transport, never live views.

        Capture still compares live storage: a custom commit or external write
        cannot be hidden by this optimization. Other layouts use normal capture.
        """
        from ..constitutive.quadrature import QuadratureField
        from .._resident_field_records import _freeze

        response = self.internal.response
        values = {"S": trial.material_response.cauchy_stress,
                  "WORK": trial.material_response.internal_work_density_increment}
        records = self._field_records()
        # Piola is produced by the operator rather than the provider transport.
        # Freeze once for both mirrored fields, not once per rollback entry.
        piola = response.response_transaction.committed["P"]
        if type(piola) is QuadratureField:
            payload = _freeze(piola.function.x.array)
            records.remember("P", payload)
            records.remember("committed_response_P", payload)
        for name, value in values.items():
            field = response.response_transaction.committed[name]
            if type(field) is not QuadratureField:
                continue
            field._array_indices()
            if field._identity_layout_cache:
                records.remember(name, value.reshape(-1))
                records.remember("committed_response_" + name, value.reshape(-1))
        if len(response.state.state_schema.variables) == 1:
            name = response.state.state_schema.variables[0].name
            field = response.state.committed[name]
            if type(field) is QuadratureField:
                field._array_indices()
            if type(field) is QuadratureField and field._identity_layout_cache:
                value = trial.material_response.state_new.reshape(-1)
                records.remember("state_" + name, value)
                records.remember("committed_state_" + name, value)

    checkpoint_snapshot = transaction_snapshot

    def _compatible_record(self, record):
        # Legacy archives contain only an execution digest. Preserve their
        # exact check: no guessing thread counts or dropping unknown identity.
        if record.get("schema") == "agentfem.private-response-hex.v4":
            return record.get("identity") == self.identity
        return (record.get("schema") == "agentfem.private-response-hex.v5"
                and self.restart_identity is not None
                and record.get("restart_identity") == self.restart_identity
                and (record.get("identity") == self.identity or (
                    isinstance(record.get("identity"), str)
                    and len(record["identity"]) == 64
                    and all(c in '0123456789abcdef' for c in record["identity"]))))

    def _increment_field_scope(self, record):
        from .._field_rollback_scope import _accepted_field_scope

        if not self._compatible_record(record):
            raise ValueError("Increment field owner identity mismatch.")
        if self.internal.response._owned is not None:
            from contextlib import nullcontext
            return nullcontext()
        return _accepted_field_scope(self._fields(), record["fields"])

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
        if not self._compatible_record(record):
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
        # Archive arrays may be writable; freeze before changing live state.
        accepted_gradient = _freeze(gradient)
        accepted_force = _freeze(force)
        accepted_displacement = _freeze(displacement)
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
        self.accepted_gradient = accepted_gradient
        self.accepted_time = self.time = float(values[0])
        (
            self.internal_work,
            self.hourglass_energy,
            self.interface_energy,
            self.viscosity_work,
            self.damping_rate_bound,
        ) = map(float, values[1:])
        self.accepted_force = accepted_force
        self.accepted_displacement = accepted_displacement
        self._trial = None
        if self.internal.response._owned is not None:
            self.internal.response._owned.capture()
