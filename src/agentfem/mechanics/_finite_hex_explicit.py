# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private finite-Hex residual lifecycle for the existing explicit Procedure.

Serial and fixed-reference only. No public provider registration is made here.
The caller declares a complete-path spectral ceiling; each endpoint additionally
checks the bounded symmetric/nonnegative tangent screen before acceptance.
"""

from hashlib import sha256

import numpy as np

from ..elements._finite_uniform_hex_material import response_fields
from ..provenance import content_fingerprint
from ..state import field_transaction
from ..time.stability import ExplicitStabilityContribution, combine_explicit_stability


class FiniteHexExplicitResidual:
    def __init__(
        self, internal, material, *, omega_squared_bound, safety=0.8, cohesive=None
    ):
        self.internal, self.material = internal, material
        self.cohesive = cohesive
        self.bound, self.safety = float(omega_squared_bound), float(safety)
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
            content_fingerprint({"bound": self.bound, "safety": self.safety}).encode()
        )
        self.identity = digest.hexdigest()
        self.accepted_gradient = np.tile(np.eye(3), (len(internal.cell_nodes), 1, 1))
        self.accepted_time = self.time = 0.0
        self._trial = None
        self._trial_displacement = None
        self._trial_time = None
        self.last_bound = None

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
        self.validate_time_increment(dt)
        vector = None
        self._trial = None
        try:
            with self.internal.trial_evaluation(
                self.material,
                deformation_gradient_old=self.accepted_gradient,
                time=self.accepted_time,
                time_increment=dt,
            ) as (vector, trial):
                if trial.material_response.minimum_suggested_time_scale < 1:
                    raise ValueError(
                        "Material requested increment reduction; fixed explicit step rejected."
                    )
                bound = self.internal.cells.tangent_spectral_bound(
                    first_piola_tangent=self.internal.response.tangent.values
                )
                if bound > self.bound * (1 + 1e-12):
                    raise ValueError(
                        "Current tangent exceeds the declared spectral ceiling."
                    )
                if self.cohesive is not None:
                    self.cohesive.add_to_vector(vector)
                if not np.isfinite(vector.array).all():
                    raise ValueError("Non-finite combined finite Hex8 residual.")
            self._trial = trial
            self._trial_displacement = self.internal.displacement.x.array.copy()
            self._trial_time = self.time
            self.last_bound = bound
            return vector
        except BaseException:
            self.internal.response.rollback()
            if self.cohesive is not None:
                self.cohesive.rollback()
            self.internal._tangent_available = False
            raise

    def commit(self):
        if self._trial is None:
            raise RuntimeError("No finite Hex8 material trial to accept.")
        if self.time != self._trial_time:
            raise RuntimeError("Time changed after the material trial.")
        if not np.array_equal(
            self.internal.displacement.x.array, self._trial_displacement
        ):
            raise RuntimeError("Displacement changed after the material trial.")
        self.internal.response.commit()
        if self.cohesive is not None:
            self.cohesive.commit()
        self.accepted_gradient = self._trial.deformation_gradient.copy()
        self.accepted_time = self.time
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
        return record

    def checkpoint_snapshot(self):
        """Numeric arrays for the shared serial binary auxiliary envelope."""
        return self.transaction_snapshot()

    def checkpoint_capabilities(self):
        from ..checkpointing import CheckpointCapabilities

        return CheckpointCapabilities(
            schemas=("agentfem.transient-checkpoint.v5", "agentfem.transient-checkpoint.v6"),
            boundary="accepted_step", payload_scope="full_restart_state",
            state_components=("accepted deformation gradient", "material committed/trial fields",
                              "material response fields", "stability state", "interface identity"),
            atomic_publication=True, rank_count_portability="unsupported",
            identity_scope=("material schema and parameters", "reference cells and mass",
                            "hourglass coefficients", "interface mapping", "stability ceiling"),
            limitations=("private serial finite Hex8; no cross-partition material restore",),
            evidence=("serial interrupted/continuous path", "corrupt auxiliary atomic rejection"),
        )

    def transaction_snapshot(self):
        """Bound rollback and binary-checkpoint storage without JSON expansion."""
        if self._trial is not None:
            raise RuntimeError("Cannot archive an unaccepted finite Hex8 trial.")
        return {
            "schema": "agentfem.private-finite-hex.v1",
            "identity": self.identity,
            "time": self.accepted_time,
            "gradient": self.accepted_gradient.copy(),
            "last_bound": self.last_bound,
            "cohesive": None if self.cohesive is None else self.cohesive.snapshot(),
            "fields": {
                name: value.x.array.copy() for name, value in self._fields().items()
            },
        }

    def restore(self, record):
        if (
            record.get("schema") != "agentfem.private-finite-hex.v1"
            or record.get("identity") != self.identity
        ):
            raise ValueError("Finite Hex8 checkpoint identity mismatch.")
        selected_time = float(record["time"])
        if np.iscomplexobj(record["gradient"]):
            raise ValueError("Finite Hex8 checkpoint gradient must be real.")
        gradient = np.asarray(record["gradient"], dtype=float)
        bound = record["last_bound"]
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
        with field_transaction(**fields):
            for name, field in fields.items():
                field.x.array[:] = arrays[name]
                field.x.scatter_forward()
        self.accepted_time = self.time = selected_time
        self.accepted_gradient = gradient.copy()
        self.last_bound = bound
        self._trial = None
        self._trial_displacement = None
        self._trial_time = None
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
            "stability": self.stability.summary(),
            "interface_scope": None
            if self.cohesive is None
            else "isotropic_elastic_reference_area",
        }
