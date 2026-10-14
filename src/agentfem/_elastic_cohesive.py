# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Undamaged elastic traction--separation law using the cohesive protocol."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElasticCohesiveLaw:
    """Local normal/tangent stiffnesses, in traction per separation units.

    Compression and tension share the normal stiffness. There is no damage,
    friction, failure threshold or irreversible history. Tangential axes are
    the caller's declared interface frame, not material axes inferred here.
    """

    normal_stiffness: float
    tangential_stiffness: float
    second_tangential_stiffness: float | None = None
    name: str = "elastic traction-separation"

    def __post_init__(self):
        for key in (
            "normal_stiffness",
            "tangential_stiffness",
            "second_tangential_stiffness",
        ):
            value = getattr(self, key)
            if value is None and key == "second_tangential_stiffness":
                value = self.tangential_stiffness
            value = float(value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive.")
            object.__setattr__(self, key, value)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Elastic interface requires a name.")

    def summary(self):
        return {
            "kind": "elastic_cohesive",
            "mode": "mixed",
            "name": self.name,
            "normal_stiffness": self.normal_stiffness,
            "tangential_stiffness": self.tangential_stiffness,
            "second_tangential_stiffness": self.second_tangential_stiffness,
            "damage": False,
            "dissipation": False,
            "traction_measure": "force_per_reference_area",
            "stiffness_units": "traction_per_length",
        }

    def update(self, jump):
        from .interfaces import VectorCohesiveResponse

        values = np.asarray(jump, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] not in (2, 3)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "Elastic interface jump must be finite with shape (points, 2 or 3)."
            )
        count, dimension = values.shape
        stiffness = np.array(
            [
                self.normal_stiffness,
                self.tangential_stiffness,
                self.second_tangential_stiffness,
            ]
        )[:dimension]
        traction = values * stiffness
        energy = 0.5 * np.sum(values * traction, axis=1)
        if not np.all(np.isfinite(traction)) or not np.all(np.isfinite(energy)):
            raise ValueError("Elastic interface response overflowed.")
        tangent = np.broadcast_to(
            np.diag(stiffness), (count, dimension, dimension)
        ).copy()
        shear_energy = 0.5 * np.sum(values[:, 1:] * traction[:, 1:], axis=1)
        mixity = np.divide(shear_energy, energy, out=np.zeros(count), where=energy > 0)
        return VectorCohesiveResponse(
            jump=values.copy(),
            traction=traction,
            tangent=tangent,
            maximum_effective_separation=np.zeros(count),
            damage=np.zeros(count),
            stored_energy=energy,
            dissipated_energy=np.zeros(count),
            mode_mixity=mixity,
        )

    def transaction(self, size):
        return _ElasticTransaction(self, size)


class _ElasticTransaction:
    """Stateless material with explicit trial lifecycle, no artificial damage history."""

    def __init__(self, law, size):
        if (
            isinstance(size, bool)
            or not isinstance(size, (int, np.integer))
            or size < 0
        ):
            raise ValueError("Elastic transaction size must be a nonnegative integer.")
        self.law, self.size, self.trial = law, int(size), None

    def evaluate(self, jump):
        response = self.law.update(jump)
        if len(response.jump) != self.size:
            raise ValueError("Elastic interface point count differs from transaction.")
        return response

    def begin(self, jump):
        # A failed replacement trial must never leave an older trial committable.
        self.trial = None
        self.trial = self.evaluate(jump)
        return self.trial

    def commit(self):
        if self.trial is None:
            raise RuntimeError("No elastic interface trial to commit.")
        self.trial = None

    def rollback(self):
        self.trial = None

    def initialize(self, values):
        if self.trial is not None:
            raise RuntimeError("Rollback the trial before initialization.")
        values = np.broadcast_to(np.asarray(values, dtype=float), (self.size,))
        if not np.all(np.isfinite(values)) or np.any(values != 0):
            raise ValueError(
                "An elastic interface cannot initialize damage or precracking."
            )

    def snapshot(self):
        return {
            "schema": "agentfem.elastic-interface-state.v1",
            "law": self.law.summary(),
            "size": self.size,
        }

    def restore(self, snapshot):
        if snapshot != self.snapshot():
            raise ValueError("Elastic interface state identity mismatch.")
        self.trial = None

    def state_arrays(self):
        return {}

    def restore_state_arrays(self, arrays):
        if arrays:
            raise ValueError("Elastic interface has no irreversible state arrays.")
        self.trial = None
