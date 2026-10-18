# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Private face-to-Operator bridge for certified local contact integrals.

This aggregates complete face integrals into structural rows and a COO
Newton tangent. Face identities are never presented as quadrature-point IDs.
The owning nonlinear Procedure, accepted State, work ledger and Result are
intentionally absent; this module cannot admit a public contact Step.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .contact_feature_pair_geometry_research import (
    CertifiedClosedTwoFeaturePenalty,
    CertifiedPartialClosedTwoFeaturePenalty,
    CertifiedSeparatedClosedFace,
    integrate_discovered_closed_contact_face,
)
from .contact_trace import ContactTrace
from .rigid import PrescribedRigidMotion


def _read_only(values, *, dtype=None):
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class CertifiedP1FaceLayout:
    """Complete faces recovered from one positive reference-area P1 trace."""

    face_ids: np.ndarray
    face_node_ids: np.ndarray
    number_of_nodes: int


def extract_complete_p1_face_layout(
    trace: ContactTrace,
    reference_nodal_positions,
) -> CertifiedP1FaceLayout:
    """Recover face identities without using point IDs as face evidence.

    The default three-positive-point rule has exactly one triplet per face.
    Any refined, lumped, incomplete or current-area trace is rejected instead
    of being silently reinterpreted as three independent physical faces.
    """

    reference = np.asarray(reference_nodal_positions, dtype=float)
    if (
        not isinstance(trace, ContactTrace)
        or trace.measure_configuration != "reference"
        or trace.point_count == 0
        or trace.point_count % 3
        or trace.nodes_per_point != 3
        or reference.ndim != 2
        or reference.shape[1] != 3
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError(
            "Face extraction needs a complete reference P1 triangle trace."
        )
    ids = trace.point_ids.reshape(-1, 3)
    faces = ids[:, 0] // 3
    nodes = trace.node_ids.reshape(-1, 3, 3)
    expected_shapes = np.asarray(
        (
            (2.0 / 3, 1.0 / 6, 1.0 / 6),
            (1.0 / 6, 2.0 / 3, 1.0 / 6),
            (1.0 / 6, 1.0 / 6, 2.0 / 3),
        )
    )
    if (
        np.any(faces < 0)
        or np.any(faces > (np.iinfo(np.int64).max - 2) // 3)
        or not np.array_equal(ids, 3 * faces[:, None] + np.arange(3))
        or len(np.unique(faces)) != len(faces)
        or not np.array_equal(nodes, np.broadcast_to(nodes[:, :1], nodes.shape))
        or np.any(nodes < 0)
        or np.any(nodes >= len(reference))
        or not np.allclose(
            trace.shape_values.reshape(-1, 3, 3),
            expected_shapes,
            rtol=0.0,
            atol=4 * np.finfo(float).eps,
        )
    ):
        raise ValueError("Trace point triplets do not define complete P1 faces.")
    face_nodes = nodes[:, 0]
    if any(len(set(row)) != 3 for row in face_nodes) or len(
        {tuple(sorted(row)) for row in face_nodes}
    ) != len(face_nodes):
        raise ValueError("P1 face trace repeats a node or a physical face.")
    vertices = reference[face_nodes]
    areas = 0.5 * np.linalg.norm(
        np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]),
        axis=1,
    )
    if (
        not np.all(np.isfinite(areas))
        or np.any(areas <= 0.0)
        or not np.allclose(
            trace.weights.reshape(-1, 3),
            areas[:, None] / 3,
            rtol=2e-12,
            atol=0.0,
        )
    ):
        raise ValueError("P1 contact trace does not preserve positive reference area.")
    return CertifiedP1FaceLayout(
        _read_only(faces, dtype=np.int64),
        _read_only(face_nodes, dtype=np.int64),
        len(reference),
    )


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
    face_feature_kinds: tuple[tuple[str, str] | None, ...]
    face_separation_lower_bounds: tuple[float | None, ...]
    face_forces_on_structure: np.ndarray
    face_moments_on_structure_about_origin: np.ndarray
    nodal_residual: np.ndarray
    tangent_rows: np.ndarray
    tangent_columns: np.ndarray
    tangent_values: np.ndarray
    potential_energy: float
    active_reference_area: float
    force_on_structure: np.ndarray
    force_on_tool: np.ndarray
    moment_on_structure_about_origin: np.ndarray
    motion_factor: float | None = None
    moving_reference_point: np.ndarray | None = None

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
                    CertifiedSeparatedClosedFace,
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
    active_count = sum(
        not isinstance(item, CertifiedSeparatedClosedFace) for item in accepted
    )
    rows = np.empty(81 * active_count, dtype=np.int64)
    columns = np.empty_like(rows)
    values = np.empty(len(rows))
    potential = 0.0
    active_area = 0.0
    face_potential = np.empty(len(accepted))
    face_area = np.empty(len(accepted))
    face_forces = np.zeros((len(accepted), 3))
    face_moments = np.zeros((len(accepted), 3))
    activity_kinds = []
    feature_kinds = []
    separation_bounds = []
    tool_moment = np.zeros(3)
    active_index = 0
    for face_index, (face_nodes, item) in enumerate(zip(nodes, accepted, strict=True)):
        if isinstance(item, CertifiedSeparatedClosedFace):
            face_potential[face_index] = 0.0
            face_area[face_index] = 0.0
            activity_kinds.append("separated")
            feature_kinds.append(None)
            separation_bounds.append(item.separation.minimum_gap_lower_bound)
            continue
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
        target = slice(81 * active_index, 81 * (active_index + 1))
        rows[target] = np.repeat(degrees, 9)
        columns[target] = np.tile(degrees, 9)
        values[target] = response.nodal_tangent.reshape(-1)
        potential += response.potential
        selected_area = sum(response.branch_areas)
        active_area += selected_area
        face_potential[face_index] = response.potential
        face_area[face_index] = selected_area
        face_forces[face_index] = -np.sum(response.nodal_residual, axis=0)
        face_moments[face_index] = -response.rigid_rotation_reaction_about_origin
        activity_kinds.append(
            "partial"
            if isinstance(item, CertifiedPartialClosedTwoFeaturePenalty)
            else "whole"
        )
        feature_kinds.append(item.geometry.feature_kinds)
        separation_bounds.append(None)
        tool_moment += response.rigid_rotation_reaction_about_origin
        active_index += 1
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
        tuple(separation_bounds),
        _read_only(face_forces),
        _read_only(face_moments),
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


def evaluate_certified_closed_p1_contact_operator(
    trace: ContactTrace,
    reference_nodal_positions,
    current_nodal_positions,
    surface,
    *,
    penalty: float,
    order: int = 24,
    maximum_slave_faces: int = 128,
    maximum_master_facets: int = 64,
) -> CertifiedFaceContactOperator:
    """Prove *every* bounded P1 slave face before returning one Operator.

    This pure trial evaluation mutates no material/contact State. A single
    unresolved face refuses the whole contribution with its physical ID.
    No MPI owner reduction, Newton Procedure or accepted path work is implied.
    """

    layout = extract_complete_p1_face_layout(trace, reference_nodal_positions)
    reference = np.asarray(reference_nodal_positions, dtype=float)
    current = np.asarray(current_nodal_positions, dtype=float)
    if (
        type(maximum_slave_faces) is not int
        or not 1 <= len(layout.face_ids) <= maximum_slave_faces <= 1024
        or current.shape != reference.shape
        or not np.all(np.isfinite(current))
    ):
        raise ValueError("Face Operator needs a bounded current P1 nodal layout.")
    responses = []
    for face_id, nodes in zip(layout.face_ids, layout.face_node_ids, strict=True):
        reference_face = reference[nodes]
        current_face = current[nodes]
        reference_double_area = np.linalg.norm(
            np.cross(
                reference_face[1] - reference_face[0],
                reference_face[2] - reference_face[0],
            )
        )
        current_double_area = np.linalg.norm(
            np.cross(
                current_face[1] - current_face[0],
                current_face[2] - current_face[0],
            )
        )
        if (
            not np.isfinite(current_double_area)
            or current_double_area <= 1e-12 * reference_double_area
        ):
            raise ValueError(
                "AFM-CONTACT-CERTIFIED-FACE: collapsed current P1 "
                f"slave_face_id={int(face_id)}."
            )
        try:
            responses.append(
                integrate_discovered_closed_contact_face(
                    reference_face,
                    current_face,
                    surface,
                    penalty=penalty,
                    order=order,
                    maximum_master_facets=maximum_master_facets,
                )
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                "AFM-CONTACT-CERTIFIED-FACE: current trial lacks a complete "
                f"proof for slave_face_id={int(face_id)}; no Operator was returned."
            ) from exc
    return assemble_certified_contact_faces(
        layout.face_ids,
        layout.face_node_ids,
        responses,
        reference_nodal_positions=reference,
        current_nodal_positions=current,
        number_of_nodes=layout.number_of_nodes,
    )


def evaluate_moving_certified_closed_p1_contact_operator(
    trace: ContactTrace,
    reference_nodal_positions,
    current_nodal_positions,
    surface,
    *,
    motion: PrescribedRigidMotion,
    factor: float,
    penalty: float,
    order: int = 24,
    maximum_slave_faces: int = 128,
    maximum_master_facets: int = 64,
) -> CertifiedFaceContactOperator:
    """Certify a prescribed rigid pose without rebuilding the tool topology.

    At fixed pose, the inverse frame map is affine in each slave nodal
    position. The original tool's complete geometry proof therefore applies
    in its reference frame; residual/tangent are then pushed forward exactly.
    Pose work is not integrated here and no accepted State is implied.
    """

    if not isinstance(motion, PrescribedRigidMotion) or motion.dimension != 3:
        raise TypeError("Moving certified faces require a 3D prescribed rigid motion.")
    current = np.asarray(current_nodal_positions, dtype=float)
    if current.ndim != 2 or current.shape[1] != 3 or not np.all(np.isfinite(current)):
        raise ValueError("Moving certified faces need finite current 3D nodes.")
    pose = motion.state(factor)
    rotation = np.asarray(pose["rotation_matrix"], dtype=float)
    reference_point = np.asarray(motion.reference_point, dtype=float)
    moved_reference = np.asarray(pose["reference_point"], dtype=float)
    tool_frame_current = reference_point + (current - moved_reference) @ rotation
    local = evaluate_certified_closed_p1_contact_operator(
        trace,
        reference_nodal_positions,
        tool_frame_current,
        surface,
        penalty=penalty,
        order=order,
        maximum_slave_faces=maximum_slave_faces,
        maximum_master_facets=maximum_master_facets,
    )
    block_rotation = np.kron(np.eye(3), rotation)
    active = sum(kind != "separated" for kind in local.face_activity_kinds)
    transformed_tangent = np.empty_like(local.tangent_values)
    for index in range(active):
        selected = slice(81 * index, 81 * (index + 1))
        local_block = local.tangent_values[selected].reshape(9, 9)
        transformed_tangent[selected] = (
            block_rotation @ local_block @ block_rotation.T
        ).reshape(-1)
    global_face_forces = local.face_forces_on_structure @ rotation.T
    local_face_moments_at_reference = (
        local.face_moments_on_structure_about_origin
        - np.cross(reference_point, local.face_forces_on_structure)
    )
    global_face_moments = local_face_moments_at_reference @ rotation.T + np.cross(
        moved_reference, global_face_forces
    )
    return replace(
        local,
        face_forces_on_structure=_read_only(global_face_forces),
        face_moments_on_structure_about_origin=_read_only(global_face_moments),
        nodal_residual=_read_only(local.nodal_residual @ rotation.T),
        tangent_values=_read_only(transformed_tangent),
        force_on_structure=_read_only(np.sum(global_face_forces, axis=0)),
        force_on_tool=_read_only(-np.sum(global_face_forces, axis=0)),
        moment_on_structure_about_origin=_read_only(
            np.sum(global_face_moments, axis=0)
        ),
        motion_factor=float(factor),
        moving_reference_point=_read_only(moved_reference),
    )


__all__ = [
    "CertifiedP1FaceLayout",
    "CertifiedFaceContactOperator",
    "extract_complete_p1_face_layout",
    "evaluate_certified_closed_p1_contact_operator",
    "evaluate_moving_certified_closed_p1_contact_operator",
    "assemble_certified_contact_faces",
]
