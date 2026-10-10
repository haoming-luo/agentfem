# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private finite-Hex residual lifecycle for the existing explicit Procedure.

Fixed-reference only. No public provider registration is made here.
The caller declares a complete-path spectral ceiling; each endpoint additionally
checks the signed symmetric tangent enclosure before acceptance. Optional
negative-curvature resolution does not establish physical stability.
"""

from hashlib import sha256

import numpy as np

from ..elements._finite_uniform_hex_material import response_fields
from ..provenance import (
    content_fingerprint,
    collective_call,
    collective_canonical_record,
)
from ..state import field_transaction
from ..time.stability import ExplicitStabilityContribution, combine_explicit_stability


class FiniteHexExplicitResidual:
    def __init__(
        self,
        internal,
        material,
        *,
        omega_squared_bound,
        safety=0.8,
        cohesive=None,
        maximum_negative_growth_per_increment=None,
    ):
        self.comm = internal.comm
        collective_call(
            lambda: self._initialize(
                internal,
                material,
                omega_squared_bound=omega_squared_bound,
                safety=safety,
                cohesive=cohesive,
                maximum_negative_growth_per_increment=maximum_negative_growth_per_increment,
            ),
            comm=self.comm,
            label="Finite Hex8 trajectory preparation",
        )
        collective_canonical_record(
            {
                "material": material.summary(),
                "bound": self.bound,
                "safety": self.safety,
                "negative_growth_resolution": self.maximum_negative_growth_per_increment,
            },
            comm=self.comm,
            label="Finite Hex8 trajectory contract",
        )
        self.partition_identity = self.identity
        self.identity = content_fingerprint(
            self.comm.allgather(self.partition_identity)
        )

    def _initialize(
        self,
        internal,
        material,
        *,
        omega_squared_bound,
        safety,
        cohesive,
        maximum_negative_growth_per_increment,
    ):
        self.internal, self.material = internal, material
        self.cohesive = cohesive
        self.bound, self.safety = float(omega_squared_bound), float(safety)
        self.maximum_negative_growth_per_increment = (
            maximum_negative_growth_per_increment
        )
        if maximum_negative_growth_per_increment is not None:
            selected = float(maximum_negative_growth_per_increment)
            if not np.isfinite(selected) or not 0 < selected <= 0.25:
                raise ValueError("Negative-curvature resolution must lie in (0, 0.25].")
            self.maximum_negative_growth_per_increment = selected
        if not np.isfinite(self.bound) or self.bound <= 0:
            raise ValueError(
                "A positive finite complete-path spectral ceiling is required."
            )
        if not np.isfinite(self.safety) or not 0 < self.safety < 1:
            raise ValueError("Stability safety must be between zero and one.")
        contributions = [
            ExplicitStabilityContribution.from_spectral_bound(
                "finite_hex_bulk_and_hourglass",
                self.bound,
                method="caller_complete_path_ceiling_with_endpoint_screen",
            )
        ]
        if cohesive is not None:
            from .._elastic_cohesive import ElasticCohesiveLaw
            from .._nonmatching_force import NonmatchingCohesiveForce

            if not isinstance(cohesive, NonmatchingCohesiveForce):
                raise TypeError("Expected an existing nonmatching cohesive force.")
            law = cohesive.assembler.law
            if not isinstance(law, ElasticCohesiveLaw) or not (
                law.normal_stiffness
                == law.tangential_stiffness
                == law.second_tangential_stiffness
            ):
                raise NotImplementedError(
                    "Finite reference interface requires isotropic elastic separation stiffness."
                )
            if cohesive.displacement is not internal.displacement:
                raise ValueError(
                    "Bulk and interface must share the displacement field."
                )
            if not cohesive.assembler.pairing.method.startswith("coplanar-"):
                raise NotImplementedError(
                    "Finite reference interface requires checked common-refinement coverage."
                )
            contributions.append(
                ExplicitStabilityContribution.from_spectral_bound(
                    "isotropic_reference_interface",
                    cohesive.elastic_stability_bound(internal.mass_diagonal),
                    method="fixed_reference_isotropic_interface_row_bound",
                )
            )
        self.stability = combine_explicit_stability(
            contributions,
            safety_factor=self.safety,
        )
        if np.any(internal.displacement.x.array != 0):
            raise ValueError(
                "Private finite Hex8 trajectory starts from undeformed state."
            )
        initial = material.state_schema.initial_state()
        committed = internal.response.state.committed_state_vectors()
        if not np.array_equal(committed, np.broadcast_to(initial, committed.shape)):
            raise ValueError(
                "Private finite Hex8 requires virgin material history at construction."
            )
        description = getattr(material, "summary", None)
        if not callable(description):
            raise ValueError("Material must declare a restart identity summary.")
        digest = sha256(content_fingerprint(description()).encode())
        if cohesive is not None:
            digest.update(content_fingerprint(cohesive.snapshot()).encode())
            for mapping in (cohesive.negative_dofs, cohesive.positive_dofs):
                digest.update(mapping.tobytes())
        for array in (
            internal.cell_nodes,
            internal.cells.coordinates,
            internal.mass_diagonal,
            internal.cells.hourglass_coefficient,
        ):
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        digest.update(
            content_fingerprint(
                {
                    "bound": self.bound,
                    "safety": self.safety,
                    "negative_growth_resolution": self.maximum_negative_growth_per_increment,
                }
            ).encode()
        )
        self.identity = digest.hexdigest()
        self.accepted_gradient = np.tile(np.eye(3), (len(internal.cell_nodes), 1, 1))
        self.accepted_time = self.time = 0.0
        self._trial = None
        self._trial_displacement = None
        self._trial_time = None
        self.last_bound = None
        self.last_spectrum = None
        self._accepted_energy = None
        self._accepted_force = None
        self._accepted_displacement = None
        self._trial_force = None
        self._trial_interface = None

    def _screen_trial(self, trial, dt):
        if trial.material_response.minimum_suggested_time_scale < 1:
            raise ValueError(
                "Material requested increment reduction; fixed explicit step rejected."
            )
        spectrum = self.internal.cells.tangent_spectral_report(
            first_piola_tangent=self.internal.response.tangent.owned_values
        )
        if spectrum.negative_material_curvature_cells:
            if self.maximum_negative_growth_per_increment is None:
                raise ValueError(
                    "Indefinite material tangent requires an explicit curvature policy."
                )
            if (
                dt * np.sqrt(spectrum.negative_eigenvalue_magnitude_bound)
                > self.maximum_negative_growth_per_increment
            ):
                raise ValueError(
                    "Increment under-resolves the negative-curvature growth bound."
                )
        if spectrum.positive_eigenvalue_upper_bound > self.bound * (1 + 1e-12):
            raise ValueError("Current tangent exceeds the declared spectral ceiling.")
        return spectrum

    def _initial_energy_response(self, initial_response):
        from dataclasses import replace
        from ..constitutive.material_array_batch import MaterialPointArrayBatchOutput

        if (
            self.accepted_time != 0
            or self._trial is not None
            or self._accepted_energy is not None
        ):
            raise RuntimeError(
                "Initial energy can only be declared once before advancement."
            )
        if not isinstance(initial_response, MaterialPointArrayBatchOutput):
            raise TypeError("Expected a declared initial material array response.")
        initial = replace(initial_response)
        count = len(self.internal.response.state.committed_state_vectors())
        if (
            initial.point_count != count
            or initial.state_schema != self.material.state_schema
            or initial.tangent_convention != self.material.tangent_convention
            or not np.array_equal(
                initial.state_new,
                self.internal.response.state.committed_state_vectors(),
            )
        ):
            raise ValueError(
                "Initial response identity/state differs from the virgin material."
            )
        if (
            initial.strain_energy_density is None
            or initial.dissipation_density_increment is None
        ):
            raise ValueError(
                "Initial energy requires explicit stored energy and zero dissipation increment."
            )
        if np.any(initial.dissipation_density_increment != 0):
            raise ValueError("Initial declaration must not dissipate energy.")
        if set(initial.stored_energy_density_components) != set(
            self.internal.response.stored_energy_density_components
        ):
            raise ValueError(
                "Initial stored-energy components differ from the quadrature contract."
            )
        u = self.internal.displacement.x.array
        if np.any(u != 0):
            raise ValueError("Initial response requires the undeformed configuration.")
        element = self.internal.cells.response(
            u.reshape(-1, 3)[self.internal.cell_nodes],
            first_piola=initial.cauchy_stress[: len(self.internal.cell_nodes)],
            stored_energy_density=initial.strain_energy_density[
                : len(self.internal.cell_nodes)
            ],
        )
        return initial, element

    def enable_energy(self, initial_response):
        """Declare the virgin response explicitly; never perform a fictitious step."""
        initial, element = collective_call(
            lambda: self._initial_energy_response(initial_response),
            comm=self.comm,
            label="Finite Hex8 initial energy declaration",
        )
        u = self.internal.displacement.x.array
        vector = self.internal._scatter(element.internal_force)
        try:
            initial_interface = None
            if self.cohesive is not None:
                initial_interface = self.cohesive.evaluate()
                values = vector.array.reshape(-1, 3)
                values[self.cohesive.negative_dofs] += (
                    initial_interface.negative_residual
                )
                values[self.cohesive.positive_dofs] += (
                    initial_interface.positive_residual
                )
            force = vector.array.copy()
        finally:
            vector.destroy()
        # F=I: Cauchy and first Piola stresses coincide.
        energy = {
            "bulk_stored_energy": float(np.sum(element.physical_energy)),
            "hourglass_energy": float(np.sum(element.hourglass_energy)),
            "material_dissipation": 0.0,
            "interface_stored_energy": 0.0
            if initial_interface is None
            else initial_interface.stored_energy,
        }
        if (
            not all(np.isfinite(value) for value in energy.values())
            or not np.isfinite(force).all()
        ):
            raise ValueError("Non-finite initial energy or force.")
        digest = sha256(
            (self.partition_identity + content_fingerprint(energy)).encode()
        )
        digest.update(force.tobytes())
        response = self.internal.response
        if set(initial.stored_energy_density_components) != set(
            response.stored_energy_density_components
        ):
            raise ValueError(
                "Initial stored-energy components differ from the quadrature contract."
            )
        assignments = (
            (response.first_piola_stress, initial.cauchy_stress),
            (response.cauchy_stress, initial.cauchy_stress),
            (response.tangent, initial.consistent_tangent.reshape(-1, 3, 3, 3, 3)),
            (response.strain_energy_density, initial.strain_energy_density),
            *(
                (field, initial.stored_energy_density_components[name])
                for name, field in response.stored_energy_density_components.items()
            ),
        )
        with field_transaction(**response_fields(response)):
            for field, values in assignments:
                field.assign(values)
        self._accepted_force = force
        self._accepted_displacement = u.copy()
        self._accepted_energy = energy
        self.partition_identity = digest.hexdigest()
        self.identity = content_fingerprint(
            self.comm.allgather(self.partition_identity)
        )

    def assemble_accepted_vector(self):
        """Owned force sample for work/restart; does not update material history."""
        collective_call(
            self.require_accepted_configuration,
            comm=self.comm,
            label="Finite Hex8 accepted force",
        )
        vector = self.internal.displacement.x.petsc_vec.duplicate()
        vector.array[:] = self._accepted_force
        return vector

    def require_accepted_configuration(self):
        """Validate cached samples without allocating or assembling a vector."""
        if self._accepted_force is None or self._trial is not None:
            raise RuntimeError("Accepted energy/force response is unavailable.")
        if self.time != self.accepted_time or not np.array_equal(
            self.internal.displacement.x.array, self._accepted_displacement
        ):
            raise RuntimeError(
                "Force sampling requires the accepted configuration and time."
            )

    def validate_time_increment(self, dt):
        if not np.isfinite(dt) or dt <= 0 or dt > self.stability.selected:
            raise ValueError("Time increment exceeds the declared spectral ceiling.")

    def update_time(self, time):
        selected = float(time)
        if not np.isfinite(selected) or selected < self.accepted_time:
            raise ValueError("Finite Hex8 time must not precede accepted time.")
        self.time = selected

    def assemble_vector(self):
        dt = self.time - self.accepted_time
        collective_call(
            lambda: self.validate_time_increment(dt),
            comm=self.comm,
            label="Finite Hex8 increment",
        )
        vector = None
        self._trial = None
        self._trial_interface = None
        try:
            with self.internal.trial_evaluation(
                self.material,
                deformation_gradient_old=self.accepted_gradient,
                time=self.accepted_time,
                time_increment=dt,
            ) as (vector, trial):
                from mpi4py import MPI

                spectrum = collective_call(
                    lambda: self._screen_trial(trial, dt),
                    comm=self.comm,
                    label="Finite Hex8 spectrum",
                )
                bound = self.comm.allreduce(
                    spectrum.positive_eigenvalue_upper_bound, op=MPI.MAX
                )
                negative_bound = self.comm.allreduce(
                    spectrum.negative_eigenvalue_magnitude_bound, op=MPI.MAX
                )
                negative_cells = self.comm.allreduce(
                    spectrum.negative_material_curvature_cells, op=MPI.SUM
                )
                if self.cohesive is not None:
                    self._trial_interface = self.cohesive.add_to_vector(vector)
                if not np.isfinite(vector.array).all():
                    raise ValueError("Non-finite combined finite Hex8 residual.")
            self._trial = trial
            self._trial_displacement = self.internal.displacement.x.array.copy()
            self._trial_time = self.time
            self._trial_force = (
                vector.array.copy() if self._accepted_energy is not None else None
            )
            self.last_bound = bound
            self.last_spectrum = {
                "positive_eigenvalue_upper_bound": bound,
                "negative_eigenvalue_magnitude_bound": negative_bound,
                "negative_material_curvature_cells": negative_cells,
            }
            return vector
        except BaseException:
            self.internal.response.rollback()
            if self.cohesive is not None:
                self.cohesive.rollback()
            self.internal._tangent_available = False
            raise

    def _commit_energy(self):
        if self._trial is None:
            raise RuntimeError("No finite Hex8 material trial to accept.")
        if self.time != self._trial_time:
            raise RuntimeError("Time changed after the material trial.")
        if not np.array_equal(
            self.internal.displacement.x.array, self._trial_displacement
        ):
            raise RuntimeError("Displacement changed after the material trial.")
        energy = None
        if self._accepted_energy is not None:
            response = self._trial.material_response
            if not np.all(response.dissipation_density_increment_defined):
                raise ValueError(
                    "Energy accounting requires explicit material dissipation increments."
                )
            energy = {
                "bulk_stored_energy": float(
                    np.sum(self._trial.element_response.physical_energy)
                ),
                "hourglass_energy": float(
                    np.sum(self._trial.element_response.hourglass_energy)
                ),
                "material_dissipation": self._accepted_energy["material_dissipation"]
                + float(
                    self.internal.cells.volume
                    @ response.dissipation_density_increment[
                        : len(self.internal.cell_nodes)
                    ]
                ),
                "interface_stored_energy": 0.0
                if self._trial_interface is None
                else self._trial_interface.stored_energy,
            }
            if not all(np.isfinite(value) for value in energy.values()):
                raise ValueError("Non-finite accepted energy increment.")
        return energy

    def commit(self):
        energy = collective_call(
            self._commit_energy, comm=self.comm, label="Finite Hex8 acceptance"
        )
        self.internal.response.commit()
        if self.cohesive is not None:
            self.cohesive.commit()
        self.accepted_gradient = self._trial.deformation_gradient.copy()
        self.accepted_time = self.time
        if energy is not None:
            self._accepted_energy = energy
            self._accepted_force = self._trial_force
            self._accepted_displacement = self._trial_displacement.copy()
        self._trial = None

    def _fields(self):
        response = self.internal.response
        return {
            **response_fields(response),
            **{
                f"committed_{name}": value.function
                for name, value in response.state.committed.items()
            },
            **{
                f"trial_{name}": value.function
                for name, value in response.state.trial.items()
            },
        }

    def snapshot(self):
        record = self.transaction_snapshot()
        record["gradient"] = record["gradient"].tolist()
        record["fields"] = {
            name: value.tolist() for name, value in record["fields"].items()
        }
        for name in ("accepted_force", "accepted_displacement"):
            if record[name] is not None:
                record[name] = record[name].tolist()
        return record

    def checkpoint_snapshot(self):
        """Numeric arrays for the shared partition-bound auxiliary envelope."""
        return self.transaction_snapshot()

    def checkpoint_capabilities(self):
        from ..checkpointing import CheckpointCapabilities

        return CheckpointCapabilities(
            schemas=(
                "agentfem.transient-checkpoint.v5",
                "agentfem.transient-checkpoint.v6",
                "agentfem.transient-checkpoint.v7",
            ),
            boundary="accepted_step",
            payload_scope="full_restart_state",
            state_components=(
                "accepted deformation gradient",
                "material committed/trial fields",
                "material response fields",
                "stability state",
                "interface identity",
            ),
            atomic_publication=True,
            rank_count_portability="unsupported",
            identity_scope=(
                "material schema and parameters",
                "reference cells and mass",
                "hourglass coefficients",
                "interface mapping",
                "stability ceiling",
            ),
            limitations=(
                "same-partition finite Hex8; no cross-partition material restore",
            ),
            evidence=(
                "serial interrupted/continuous path",
                "corrupt auxiliary atomic rejection",
            ),
        )

    def transaction_snapshot(self):
        """Bound rollback and binary-checkpoint storage without JSON expansion."""
        if self._trial is not None:
            raise RuntimeError("Cannot archive an unaccepted finite Hex8 trial.")
        return {
            "schema": "agentfem.private-finite-hex.v1",
            "identity": self.identity,
            "partition_identity": self.partition_identity,
            "time": self.accepted_time,
            "gradient": self.accepted_gradient.copy(),
            "last_bound": self.last_bound,
            "last_spectrum": None
            if self.last_spectrum is None
            else dict(self.last_spectrum),
            "accepted_energy": None
            if self._accepted_energy is None
            else dict(self._accepted_energy),
            "accepted_force": None
            if self._accepted_force is None
            else self._accepted_force.copy(),
            "accepted_displacement": None
            if self._accepted_displacement is None
            else self._accepted_displacement.copy(),
            "cohesive": None if self.cohesive is None else self.cohesive.snapshot(),
            "fields": {
                name: value.x.array.copy() for name, value in self._fields().items()
            },
        }

    def _restore_payload(self, record):
        if (
            record.get("schema") != "agentfem.private-finite-hex.v1"
            or record.get("identity") != self.identity
            or record.get("partition_identity") != self.partition_identity
        ):
            raise ValueError("Finite Hex8 checkpoint identity mismatch.")
        selected_time = float(record["time"])
        if np.iscomplexobj(record["gradient"]):
            raise ValueError("Finite Hex8 checkpoint gradient must be real.")
        gradient = np.asarray(record["gradient"], dtype=float)
        if gradient.size == 0 and self.accepted_gradient.size == 0:
            # JSON loses trailing dimensions of an empty owned-cell array.
            gradient = gradient.reshape(self.accepted_gradient.shape)
        bound = record["last_bound"]
        spectrum = record.get("last_spectrum")
        if spectrum is not None:
            if (
                set(spectrum)
                != {
                    "positive_eigenvalue_upper_bound",
                    "negative_eigenvalue_magnitude_bound",
                    "negative_material_curvature_cells",
                }
                or spectrum["positive_eigenvalue_upper_bound"] != bound
                or not np.isfinite(spectrum["negative_eigenvalue_magnitude_bound"])
                or spectrum["negative_eigenvalue_magnitude_bound"] < 0
                or not isinstance(spectrum["negative_material_curvature_cells"], int)
                or not 0
                <= spectrum["negative_material_curvature_cells"]
                <= self.internal.displacement.function_space.mesh.topology.index_map(
                    3
                ).size_global
            ):
                raise ValueError("Invalid finite Hex8 signed spectrum evidence.")
        if (
            not np.isfinite(selected_time)
            or selected_time < 0
            or gradient.shape != self.accepted_gradient.shape
            or not np.isfinite(gradient).all()
            or np.any(np.linalg.det(gradient) <= 0)
            or (
                bound is not None
                and (
                    not np.isfinite(bound) or not 0 < bound <= self.bound * (1 + 1e-12)
                )
            )
        ):
            raise ValueError("Invalid finite Hex8 checkpoint kinematics/stability.")
        fields = self._fields()
        energy = record.get("accepted_energy")
        accepted_arrays = []
        if (energy is None) != (self._accepted_energy is None):
            raise ValueError("Finite Hex8 checkpoint energy contract differs.")
        if energy is not None:
            if set(energy) != set(self._accepted_energy) or not all(
                np.isreal(value) and np.isscalar(value) and np.isfinite(value)
                for value in energy.values()
            ):
                raise ValueError("Invalid finite Hex8 accepted energy.")
            for name in ("accepted_force", "accepted_displacement"):
                value = np.asarray(record[name])
                if (
                    np.iscomplexobj(value)
                    or value.shape
                    != (
                        self.internal.mass_diagonal.shape
                        if name == "accepted_force"
                        else self.internal.displacement.x.array.shape
                    )
                    or not np.isfinite(value).all()
                ):
                    raise ValueError("Invalid finite Hex8 accepted force/displacement.")
                accepted_arrays.append(value.copy())
        elif (
            record.get("accepted_force") is not None
            or record.get("accepted_displacement") is not None
        ):
            raise ValueError(
                "Unexpected finite Hex8 accepted force without energy contract."
            )
        expected_cohesive = None if self.cohesive is None else self.cohesive.snapshot()
        if record.get("cohesive") != expected_cohesive:
            raise ValueError("Finite Hex8 checkpoint interface identity mismatch.")
        if set(record["fields"]) != set(fields):
            raise ValueError("Finite Hex8 checkpoint fields differ.")
        if any(np.iscomplexobj(value) for value in record["fields"].values()):
            raise ValueError("Finite Hex8 checkpoint fields must be real.")
        arrays = {
            name: np.asarray(record["fields"][name], dtype=float) for name in fields
        }
        for name, field in fields.items():
            if (
                arrays[name].shape != field.x.array.shape
                or not np.isfinite(arrays[name]).all()
            ):
                raise ValueError(f"Invalid finite Hex8 checkpoint field {name}.")
        return (
            selected_time,
            gradient,
            bound,
            spectrum,
            energy,
            accepted_arrays,
            fields,
            arrays,
        )

    def restore(self, record):
        (
            selected_time,
            gradient,
            bound,
            spectrum,
            energy,
            accepted_arrays,
            fields,
            arrays,
        ) = collective_call(
            lambda: self._restore_payload(record),
            comm=self.comm,
            label="Finite Hex8 restore identity",
        )
        with field_transaction(**fields):
            for name, field in fields.items():
                field.x.array[:] = arrays[name]
                field.x.scatter_forward()
        self.accepted_time = self.time = selected_time
        self.accepted_gradient = gradient.copy()
        self.last_bound = bound
        self.last_spectrum = None if spectrum is None else dict(spectrum)
        if energy is not None:
            self._accepted_energy = dict(energy)
            self._accepted_force, self._accepted_displacement = accepted_arrays
        self._trial = None
        self._trial_displacement = None
        self._trial_time = None
        self._trial_force = None
        self._trial_interface = None
        self.internal._tangent_available = False
        if self.cohesive is not None:
            self.cohesive.rollback()

    def summary(self):
        return {
            "kind": "private_finite_hex_explicit_residual",
            "identity": self.identity,
            "accepted_time": self.accepted_time,
            "stability_scope": "caller_path_ceiling_and_endpoint_tangent_screen",
            "omega_squared_bound": self.bound,
            "last_endpoint_bound": self.last_bound,
            "signed_spectrum": self.last_spectrum,
            "negative_growth_resolution": self.maximum_negative_growth_per_increment,
            "stability": self.stability.summary(),
            "interface_scope": None
            if self.cohesive is None
            else "isotropic_elastic_reference_area",
        }
