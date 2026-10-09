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
    def __init__(self, internal, material, *, omega_squared_bound, safety=0.8):
        self.internal, self.material = internal, material
        self.bound, self.safety = float(omega_squared_bound), float(safety)
        if not np.isfinite(self.bound) or self.bound <= 0:
            raise ValueError(
                "A positive finite complete-path spectral ceiling is required."
            )
        if not np.isfinite(self.safety) or not 0 < self.safety < 1:
            raise ValueError("Stability safety must be between zero and one.")
        self.stability = combine_explicit_stability(
            (
                ExplicitStabilityContribution.from_spectral_bound(
                    "finite_hex_bulk_and_hourglass",
                    self.bound,
                    method="caller_complete_path_ceiling_with_endpoint_screen",
                ),
            ),
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
            with field_transaction(**response_fields(self.internal.response)):
                vector, trial = self.internal.evaluate(
                    self.material,
                    deformation_gradient_old=self.accepted_gradient,
                    time=self.accepted_time,
                    time_increment=dt,
                )
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
            self._trial = trial
            self._trial_displacement = self.internal.displacement.x.array.copy()
            self._trial_time = self.time
            self.last_bound = bound
            return vector
        except Exception:
            if vector is not None:
                vector.destroy()
            self.internal.response.rollback()
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

    def transaction_snapshot(self):
        """Bound in-memory rollback storage; JSON expansion is checkpoint-only."""
        if self._trial is not None:
            raise RuntimeError("Cannot archive an unaccepted finite Hex8 trial.")
        return {
            "schema": "agentfem.private-finite-hex.v1",
            "identity": self.identity,
            "time": self.accepted_time,
            "gradient": self.accepted_gradient.copy(),
            "last_bound": self.last_bound,
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
        if set(record["fields"]) != set(fields):
            raise ValueError("Finite Hex8 checkpoint fields differ.")
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

    def summary(self):
        return {
            "kind": "private_finite_hex_explicit_residual",
            "identity": self.identity,
            "accepted_time": self.accepted_time,
            "stability_scope": "caller_path_ceiling_and_endpoint_tangent_screen",
            "omega_squared_bound": self.bound,
            "last_endpoint_bound": self.last_bound,
        }
