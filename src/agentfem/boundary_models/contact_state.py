# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Accepted/trial closest-point state for finite-sliding contact.

This module owns neither contact search nor contact forces.  It preserves the
accepted projection boundary between nonlinear trials using stable contact
point and surface-entity identities.  A Procedure decides when to replace the
trial projection; a contact Operator may consume it after that decision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .rigid import SurfaceProjection, _readonly_array


_RECORD_SCHEMA = "agentfem.contact-projection-record.v1"
_STATE_SCHEMA = "agentfem.contact-projection-state.v1"


def _projection_in_order(projection: SurfaceProjection, order) -> SurfaceProjection:
    selected = np.asarray(order, dtype=np.int64)
    return SurfaceProjection(
        surface_name=projection.surface_name,
        surface_kind=projection.surface_kind,
        query_points=projection.query_points[selected],
        closest_points=projection.closest_points[selected],
        normals=projection.normals[selected],
        signed_gaps=projection.signed_gaps[selected],
        valid=projection.valid[selected],
        status_codes=projection.status_codes[selected],
        method=projection.method,
        entity_ids=(
            None if projection.entity_ids is None else projection.entity_ids[selected]
        ),
        geometry_fingerprint=projection.geometry_fingerprint,
        local_coordinates=(
            None
            if projection.local_coordinates is None
            else projection.local_coordinates[selected]
        ),
        local_coordinate_system=projection.local_coordinate_system,
    )


@dataclass(frozen=True, eq=False)
class ContactProjectionRecord:
    """One canonically ordered closest-point record.

    Point IDs identify slave/contact points independently of array order,
    process rank, and local degree-of-freedom numbering.  Surface ``entity_ids``
    inside :class:`SurfaceProjection` likewise remain stable facet identities;
    changing the contacted facet is expected during finite sliding.
    """

    point_ids: object
    projection: SurfaceProjection

    def __post_init__(self) -> None:
        if not isinstance(self.projection, SurfaceProjection):
            raise TypeError("Contact projection record requires SurfaceProjection.")
        raw_point_ids = np.asarray(self.point_ids)
        if raw_point_ids.ndim != 1:
            raise ValueError("Contact point IDs must be one-dimensional.")
        if raw_point_ids.size and raw_point_ids.dtype.kind not in {"i", "u"}:
            raise TypeError("Contact point IDs must be explicit integers.")
        if raw_point_ids.dtype.kind == "u" and np.any(
            raw_point_ids > np.iinfo(np.int64).max
        ):
            raise ValueError("Contact point IDs must fit signed 64-bit identity.")
        point_ids = raw_point_ids.astype(np.int64, copy=False)
        if point_ids.shape != (self.projection.point_count,):
            raise ValueError(
                "Contact point IDs must contain one value per projected point."
            )
        if np.any(point_ids < 0):
            raise ValueError("Contact point IDs must be non-negative.")
        if np.unique(point_ids).size != point_ids.size:
            raise ValueError("Contact point IDs must be unique.")
        order = np.argsort(point_ids, kind="stable")
        object.__setattr__(
            self,
            "point_ids",
            _readonly_array(point_ids[order], dtype=np.int64),
        )
        object.__setattr__(
            self,
            "projection",
            _projection_in_order(self.projection, order),
        )

    @property
    def point_count(self) -> int:
        return int(self.point_ids.size)

    def summary(self) -> dict[str, object]:
        return {
            "point_count": self.point_count,
            "point_identity": "stable_int64",
            "point_ids_sorted": True,
            "surface": self.projection.summary(),
        }

    def snapshot(self) -> dict[str, object]:
        projection = self.projection
        return {
            "schema": _RECORD_SCHEMA,
            "point_ids": self.point_ids.copy(),
            "surface_name": projection.surface_name,
            "surface_kind": projection.surface_kind,
            "query_points": projection.query_points.copy(),
            "closest_points": projection.closest_points.copy(),
            "normals": projection.normals.copy(),
            "signed_gaps": projection.signed_gaps.copy(),
            "valid": projection.valid.copy(),
            "status_codes": projection.status_codes.copy(),
            "method": projection.method,
            "entity_ids": (
                None if projection.entity_ids is None else projection.entity_ids.copy()
            ),
            "geometry_fingerprint": projection.geometry_fingerprint,
            "local_coordinates": (
                None
                if projection.local_coordinates is None
                else projection.local_coordinates.copy()
            ),
            "local_coordinate_system": projection.local_coordinate_system,
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> ContactProjectionRecord:
        if not isinstance(snapshot, Mapping):
            raise TypeError("Contact projection record snapshot must be a mapping.")
        if snapshot.get("schema") != _RECORD_SCHEMA:
            raise ValueError("Unsupported contact projection record schema.")
        required = {
            "schema",
            "point_ids",
            "surface_name",
            "surface_kind",
            "query_points",
            "closest_points",
            "normals",
            "signed_gaps",
            "valid",
            "status_codes",
            "method",
            "entity_ids",
            "geometry_fingerprint",
            "local_coordinates",
            "local_coordinate_system",
        }
        if set(snapshot) != required:
            raise ValueError("Contact projection record snapshot fields differ.")
        projection = SurfaceProjection(
            surface_name=snapshot["surface_name"],
            surface_kind=snapshot["surface_kind"],
            query_points=snapshot["query_points"],
            closest_points=snapshot["closest_points"],
            normals=snapshot["normals"],
            signed_gaps=snapshot["signed_gaps"],
            valid=snapshot["valid"],
            status_codes=snapshot["status_codes"],
            method=snapshot["method"],
            entity_ids=snapshot["entity_ids"],
            geometry_fingerprint=snapshot["geometry_fingerprint"],
            local_coordinates=snapshot["local_coordinates"],
            local_coordinate_system=snapshot["local_coordinate_system"],
        )
        return cls(point_ids=snapshot["point_ids"], projection=projection)


def _same_projection_contract(
    accepted: ContactProjectionRecord,
    candidate: ContactProjectionRecord,
) -> bool:
    old = accepted.projection
    new = candidate.projection
    return (
        old.surface_name == new.surface_name
        and old.surface_kind == new.surface_kind
        and old.dimension == new.dimension
        and old.geometry_fingerprint == new.geometry_fingerprint
        and old.local_coordinate_system == new.local_coordinate_system
        and (old.entity_ids is None) == (new.entity_ids is None)
    )


@dataclass
class ContactProjectionState:
    """Atomic accepted/trial projection state for one contact pair."""

    accepted: ContactProjectionRecord | None = None
    trial: ContactProjectionRecord | None = None

    def __post_init__(self) -> None:
        if self.accepted is not None and not isinstance(
            self.accepted, ContactProjectionRecord
        ):
            raise TypeError("Accepted contact projection has the wrong type.")
        if self.trial is not None:
            raise ValueError(
                "Construct contact projection state at an accepted boundary; "
                "use begin() to create a trial."
            )

    def begin(
        self,
        point_ids,
        projection: SurfaceProjection,
    ) -> ContactProjectionRecord:
        """Replace the current trial without mutating accepted evidence."""

        candidate = ContactProjectionRecord(point_ids, projection)
        if self.accepted is not None:
            if not np.array_equal(self.accepted.point_ids, candidate.point_ids):
                raise ValueError(
                    "Contact point identity changed; explicitly initialize a new "
                    "contact state instead of mutating an accepted pair."
                )
            if not _same_projection_contract(self.accepted, candidate):
                raise ValueError(
                    "Contact surface contract changed; explicitly initialize a new "
                    "contact state instead of reusing accepted history."
                )
        self.trial = candidate
        return candidate

    def commit(self) -> None:
        if self.trial is None:
            raise RuntimeError("No contact projection trial is available to commit.")
        self.accepted = self.trial
        self.trial = None

    def rollback(self) -> None:
        self.trial = None

    def snapshot(self) -> dict[str, object]:
        if self.trial is not None:
            raise RuntimeError(
                "Contact projection checkpoints require an accepted boundary; "
                "commit or rollback the active trial first."
            )
        return {
            "schema": _STATE_SCHEMA,
            "accepted": (None if self.accepted is None else self.accepted.snapshot()),
        }

    def restore(self, snapshot: object) -> None:
        if self.trial is not None:
            raise RuntimeError(
                "Rollback the active contact projection trial before restore."
            )
        if not isinstance(snapshot, Mapping):
            raise TypeError("Contact projection state snapshot must be a mapping.")
        if set(snapshot) != {"schema", "accepted"}:
            raise ValueError("Contact projection state snapshot fields differ.")
        if snapshot.get("schema") != _STATE_SCHEMA:
            raise ValueError("Unsupported contact projection state schema.")
        raw = snapshot["accepted"]
        restored = None if raw is None else ContactProjectionRecord.from_snapshot(raw)
        if self.accepted is not None and restored is None:
            raise ValueError("Restored contact projection is unexpectedly empty.")
        if self.accepted is not None and restored is not None:
            if not np.array_equal(self.accepted.point_ids, restored.point_ids):
                raise ValueError("Restored contact point identity differs.")
            if not _same_projection_contract(self.accepted, restored):
                raise ValueError("Restored contact surface contract differs.")
        self.accepted = restored

    def summary(self) -> dict[str, object]:
        return {
            "kind": "contact_projection_state",
            "accepted": (None if self.accepted is None else self.accepted.summary()),
            "trial_available": self.trial is not None,
            "transaction": "trial_commit_rollback",
            "checkpoint_boundary": "accepted_only",
        }


__all__ = ["ContactProjectionRecord", "ContactProjectionState"]
