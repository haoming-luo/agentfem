# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Private face-to-Operator bridge for certified local contact integrals.

This aggregates complete face integrals into structural rows and a COO
Newton tangent. Face identities are never presented as quadrature-point IDs.
The owning nonlinear Procedure, accepted State, work ledger and Result are
intentionally absent; this module cannot admit a public contact Step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact_feature_pair_geometry_research import (
    CertifiedClosedTwoFeaturePenalty,
    CertifiedPartialClosedTwoFeaturePenalty,
)


def _read_only(values, *, dtype=None):
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class CertifiedFaceContactOperator:
    """One tool's serial, same-configuration local Operator contribution."""

    face_ids: np.ndarray
    face_node_ids: np.ndarray
    number_of_nodes: int
    tool_fingerprint: str
    face_potential_energies: np.ndarray
    face_active_reference_areas: np.ndarray
    face_activity_kinds: tuple[str, ...]
    face_feature_kinds: tuple[tuple[str, str], ...]
    nodal_residual: np.ndarray
    tangent_rows: np.ndarray
    tangent_columns: np.ndarray
    tangent_values: np.ndarray
    potential_energy: float
    active_reference_area: float
    force_on_structure: np.ndarray
    force_on_tool: np.ndarray
    moment_on_structure_about_origin: np.ndarray

    def tangent_action(self, direction) -> np.ndarray:
        values = np.asarray(direction, dtype=float)
        if values.shape != (self.number_of_nodes, 3) or not np.all(np.isfinite(values)):
            raise ValueError("Contact Operator direction needs finite nodal 3D values.")
        action = np.bincount(
            self.tangent_rows,
            weights=self.tangent_values * values.reshape(-1)[self.tangent_columns],
            minlength=3 * self.number_of_nodes,
        )
        return action.reshape(self.number_of_nodes, 3)

    def moment_on_structure_about(self, reference_point) -> np.ndarray:
        point = np.asarray(reference_point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("Tool moment reference must be one finite 3D point.")
        return self.moment_on_structure_about_origin - np.cross(
            point, self.force_on_structure
        )


def assemble_certified_contact_faces(
    face_ids,
    face_node_ids,
    responses,
    *,
    reference_nodal_positions,
    current_nodal_positions,
    number_of_nodes: int,
) -> CertifiedFaceContactOperator:
    """Assemble only already-certified finite-feature face integrals.

    No constitutive or geometry result is recomputed here. Every active face
    must have passed its current-configuration proof before this additive
    global-row bridge is called. The caller may not pass raw distance
    integrals lacking the closed-tool sign certificate.
    """

    raw_identities = np.asarray(face_ids)
    raw_nodes = np.asarray(face_node_ids)
    if raw_identities.dtype.kind not in "iu" or raw_nodes.dtype.kind not in "iu":
        raise ValueError("Face and node identities must be exact integers.")
    identities = np.asarray(raw_identities, dtype=np.int64)
    nodes = np.asarray(raw_nodes, dtype=np.int64)
    reference = np.asarray(reference_nodal_positions, dtype=float)
    current = np.asarray(current_nodal_positions, dtype=float)
    accepted = tuple(responses)
    count = number_of_nodes
    if (
        identities.ndim != 1
        or len(identities) == 0
        or len(np.unique(identities)) != len(identities)
        or np.any(identities < 0)
        or nodes.shape != (len(identities), 3)
        or type(count) is not int
        or count < 3
        or reference.shape != (count, 3)
        or current.shape != (count, 3)
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(current))
        or len(accepted) != len(identities)
        or np.any(nodes < 0)
        or np.any(nodes >= count)
        or any(len(set(row)) != 3 for row in nodes)
        or len({tuple(sorted(row)) for row in nodes}) != len(nodes)
        or any(
            not isinstance(
                item,
                (
                    CertifiedClosedTwoFeaturePenalty,
                    CertifiedPartialClosedTwoFeaturePenalty,
                ),
            )
            for item in accepted
        )
    ):
        raise ValueError("Face Operator needs distinct bounded certified P1 faces.")
    fingerprints = {item.tool_fingerprint for item in accepted}
    if len(fingerprints) != 1:
        raise ValueError("One face Operator cannot combine different rigid tools.")
    for face_nodes, item in zip(nodes, accepted, strict=True):
        if not np.array_equal(
            reference[face_nodes], item.reference_vertices
        ) or not np.array_equal(current[face_nodes], item.current_vertices):
            raise ValueError(
                "A certified contact face is paired with different geometry."
            )
    residual = np.zeros((count, 3))
    rows = np.empty(81 * len(accepted), dtype=np.int64)
    columns = np.empty_like(rows)
    values = np.empty(len(rows))
    potential = 0.0
    active_area = 0.0
    face_potential = np.empty(len(accepted))
    face_area = np.empty(len(accepted))
    activity_kinds = []
    feature_kinds = []
    tool_moment = np.zeros(3)
    for face_index, (face_nodes, item) in enumerate(zip(nodes, accepted, strict=True)):
        response = item.response
        if (
            response.nodal_residual.shape != (3, 3)
            or response.nodal_tangent.shape != (3, 3, 3, 3)
            or not np.all(np.isfinite(response.nodal_residual))
            or not np.all(np.isfinite(response.nodal_tangent))
            or not np.isfinite(response.potential)
            or response.potential < 0.0
        ):
            raise ValueError("Certified face produced an invalid local Operator.")
        np.add.at(residual, face_nodes, response.nodal_residual)
        degrees = (3 * face_nodes[:, None] + np.arange(3)).reshape(-1)
        target = slice(81 * face_index, 81 * (face_index + 1))
        rows[target] = np.repeat(degrees, 9)
        columns[target] = np.tile(degrees, 9)
        values[target] = response.nodal_tangent.reshape(-1)
        potential += response.potential
        selected_area = sum(response.branch_areas)
        active_area += selected_area
        face_potential[face_index] = response.potential
        face_area[face_index] = selected_area
        activity_kinds.append(
            "partial"
            if isinstance(item, CertifiedPartialClosedTwoFeaturePenalty)
            else "whole"
        )
        feature_kinds.append(item.geometry.feature_kinds)
        tool_moment += response.rigid_rotation_reaction_about_origin
    force_structure = -np.sum(residual, axis=0)
    return CertifiedFaceContactOperator(
        _read_only(identities, dtype=np.int64),
        _read_only(nodes, dtype=np.int64),
        count,
        next(iter(fingerprints)),
        _read_only(face_potential),
        _read_only(face_area),
        tuple(activity_kinds),
        tuple(feature_kinds),
        _read_only(residual),
        _read_only(rows, dtype=np.int64),
        _read_only(columns, dtype=np.int64),
        _read_only(values),
        float(potential),
        float(active_area),
        _read_only(force_structure),
        _read_only(-force_structure),
        _read_only(-tool_moment),
    )


__all__ = [
    "CertifiedFaceContactOperator",
    "assemble_certified_contact_faces",
]
