# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Serial composition of certified frictionless faces and reviewed point pairs.

Each pair retains its own physical integration semantics and named evidence.
The wrapper owns no Newton or material State. It rejects a stale combined
trial before adding any PETSc residual or tangent entries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._certified_face_trial_provider import (
    CertifiedFaceContactTrial,
    DolfinxCertifiedFaceContactTrialProvider,
)
from ._petsc_contact_rows import add_contact_nodal_residual
from .dolfinx_implicit_facet_contact import DolfinxImplicitRigidContact
from .implicit_contact import ImplicitRigidContactContribution


@dataclass(frozen=True, eq=False)
class MixedFacePointContactTrial:
    """One atomic additive trial tied to displacement and all rigid poses."""

    pairs: tuple[
        tuple[str, CertifiedFaceContactTrial | ImplicitRigidContactContribution], ...
    ]
    displacement_values: np.ndarray
    motion_state: tuple[tuple[str, float | None], ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.displacement_values, dtype=float).copy()
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(
                "Mixed contact trial needs a finite displacement snapshot."
            )
        values.setflags(write=False)
        object.__setattr__(self, "displacement_values", values)

    @property
    def nodal_structural_residual(self) -> np.ndarray:
        return np.sum(
            [item.nodal_structural_residual for _, item in self.pairs], axis=0
        )

    @property
    def potential_energy(self) -> float:
        return float(sum(item.potential_energy for _, item in self.pairs))

    def tangent_action(self, direction) -> np.ndarray:
        return np.sum(
            [item.tangent_action(direction) for _, item in self.pairs], axis=0
        )


@dataclass(eq=False)
class DolfinxMixedFacePointContactTrialProvider:
    """One bounded Step composition; sharp faces have no tangential law."""

    providers: tuple[
        DolfinxCertifiedFaceContactTrialProvider | DolfinxImplicitRigidContact, ...
    ]

    def __post_init__(self) -> None:
        self.providers = tuple(self.providers)
        if (
            len(self.providers) < 2
            or not any(
                isinstance(item, DolfinxCertifiedFaceContactTrialProvider)
                for item in self.providers
            )
            or not any(
                isinstance(item, DolfinxImplicitRigidContact) for item in self.providers
            )
            or any(
                not isinstance(
                    item,
                    (
                        DolfinxCertifiedFaceContactTrialProvider,
                        DolfinxImplicitRigidContact,
                    ),
                )
                for item in self.providers
            )
        ):
            raise ValueError(
                "Mixed contact needs both certified-face and reviewed point pairs."
            )
        if len({item.name for item in self.providers}) != len(self.providers):
            raise ValueError("Mixed contact pair names must be unique.")
        self.displacement = self.providers[0].displacement
        if any(item.displacement is not self.displacement for item in self.providers):
            raise ValueError("Mixed contact pairs must share one displacement field.")

    @property
    def has_prescribed_motion(self) -> bool:
        return any(item.has_prescribed_motion for item in self.providers)

    def update_motion(self, factor: float) -> None:
        previous = self.snapshot_motion()
        try:
            for item in self.providers:
                item.update_motion(factor)
        except Exception:
            self.restore_motion(previous)
            raise

    def snapshot_motion(self) -> tuple[tuple[str, float | None], ...]:
        return tuple((item.name, item.snapshot_motion()) for item in self.providers)

    def restore_motion(self, saved) -> None:
        records = tuple(saved)
        if tuple(name for name, _ in records) != tuple(
            item.name for item in self.providers
        ):
            raise ValueError("Mixed contact pose identities differ.")
        previous = self.snapshot_motion()
        try:
            for item, (_, state) in zip(self.providers, records, strict=True):
                item.restore_motion(state)
        except Exception:
            for item, (_, state) in zip(self.providers, previous, strict=True):
                item.restore_motion(state)
            raise

    def snapshot_runtime_state(self):
        return tuple(
            (item.name, item.snapshot_runtime_state()) for item in self.providers
        )

    def restore_runtime_state(self, saved) -> None:
        records = tuple(saved)
        if tuple(name for name, _ in records) != tuple(
            item.name for item in self.providers
        ):
            raise ValueError("Mixed contact runtime identities differ.")
        previous = self.snapshot_runtime_state()
        try:
            for item, (_, state) in zip(self.providers, records, strict=True):
                item.restore_runtime_state(state)
        except Exception:
            for item, (_, state) in zip(self.providers, previous, strict=True):
                item.restore_runtime_state(state)
            raise

    def snapshot_accepted_state(self) -> dict[str, object] | None:
        pairs = {
            item.name: item.snapshot_accepted_state() for item in self.providers
        }
        if not any(value is not None for value in pairs.values()):
            return None
        return {
            "schema": "agentfem.implicit-rigid-pair-states.v1",
            "pairs": pairs,
        }

    def restore_accepted_state(self, saved: dict[str, object] | None) -> None:
        has_friction = any(
            isinstance(item, DolfinxImplicitRigidContact)
            and item.friction_state is not None
            for item in self.providers
        )
        if not has_friction:
            if saved is not None:
                raise ValueError("Frictionless mixed contact has no tangential State.")
            return
        if (
            not isinstance(saved, dict)
            or set(saved) != {"schema", "pairs"}
            or saved["schema"] != "agentfem.implicit-rigid-pair-states.v1"
            or not isinstance(saved["pairs"], dict)
            or set(saved["pairs"]) != {item.name for item in self.providers}
        ):
            raise ValueError("Mixed contact accepted State identity differs.")
        previous = tuple(item.snapshot_accepted_state() for item in self.providers)
        try:
            for item in self.providers:
                item.restore_accepted_state(saved["pairs"][item.name])
        except Exception:
            for item, original in zip(self.providers, previous, strict=True):
                item.restore_accepted_state(original)
            raise

    def validate_accepted_configuration(self) -> None:
        self.evaluate()
        for item in self.providers:
            item.validate_accepted_configuration()

    def commit_accepted(self) -> None:
        self.evaluate()
        previous = self.snapshot_runtime_state()
        try:
            for item in self.providers:
                item.commit_accepted()
        except Exception:
            self.restore_runtime_state(previous)
            raise

    def evaluate(self) -> MixedFacePointContactTrial:
        values = self.displacement.x.array.copy()
        motion = self.snapshot_motion()
        # Complete all provider evaluations before publishing one total trial.
        pairs = tuple((item.name, item.evaluate()) for item in self.providers)
        if (
            not np.array_equal(values, self.displacement.x.array)
            or motion != self.snapshot_motion()
        ):
            raise RuntimeError("Mixed contact changed during trial evaluation.")
        return MixedFacePointContactTrial(pairs, values, motion)

    def _check_current(self, trial: MixedFacePointContactTrial) -> None:
        if (
            not isinstance(trial, MixedFacePointContactTrial)
            or tuple(name for name, _ in trial.pairs)
            != tuple(item.name for item in self.providers)
            or not np.array_equal(trial.displacement_values, self.displacement.x.array)
            or trial.motion_state != self.snapshot_motion()
        ):
            raise ValueError(
                "Mixed contact trial differs from current displacement or pose."
            )
        for item, (_, selected) in zip(self.providers, trial.pairs, strict=True):
            if isinstance(item, DolfinxCertifiedFaceContactTrialProvider):
                item._check_current(selected)

    def add_residual(self, vector, trial: MixedFacePointContactTrial) -> None:
        self._check_current(trial)
        add_contact_nodal_residual(vector, trial.nodal_structural_residual)

    def add_tangent(
        self, matrix, trial: MixedFacePointContactTrial, *, constrained_dofs=()
    ) -> None:
        self._check_current(trial)
        for item, (_, selected) in zip(self.providers, trial.pairs, strict=True):
            item.add_tangent(matrix, selected, constrained_dofs=constrained_dofs)

    def dual_evidences(self, problem):
        return tuple(item.dual_evidence(problem) for item in self.providers)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "dolfinx_mixed_certified_face_point_contact_research",
            "pairs": tuple(item.summary() for item in self.providers),
            "execution_scope": (
                "serial_face_frictionless_point_reviewed_friction_research"
                if any(
                    isinstance(item, DolfinxImplicitRigidContact)
                    and item.friction_state is not None
                    for item in self.providers
                )
                else "serial_frictionless_research"
            ),
            "product_release_admitted": False,
        }
