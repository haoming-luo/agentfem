# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Reference-triangle quadrature for one analytical radial contact patch.

This private geometry operator does not decide Newton convergence or accept a
contact state.  On an affine P1 slave face, the sphere/cylinder active set is
the sign of a quadratic in the two reference coordinates.  We split the outer
coordinate at all discriminant and boundary-intersection events, then solve
the inner quadratic exactly at each positive Gauss node.  Integration of the
remaining smooth, generally non-polynomial response is still numerical.
"""

from __future__ import annotations

import numpy as np

from .contact_quadratic_region import (
    QuadraticReferenceLevelSet,
    quadratic_region_triangle_quadrature,
)
from .contact_trace import ContactTrace
from .rigid import (
    RigidCylinderSurface,
    RigidSphereSurface,
    _transformed_point_and_direction,
)


_LOW_ORDER = 12
_HIGH_ORDER = 24
_FACE_BUDGET = 128
_FACE_POINT_BUDGET = 4096
_GLOBAL_POINT_BUDGET = 32768
_DEFAULT_RELATIVE_TOLERANCE = 1.0e-5


def radial_partition_policy_summary() -> dict[str, object]:
    """Algorithm identity for Result and checkpoint provenance."""

    return {
        "kind": "analytic_radial_conic_active_partition",
        "quadrature_orders": (_LOW_ORDER, _HIGH_ORDER),
        "relative_tolerance": _DEFAULT_RELATIVE_TOLERANCE,
        "error_status": "paired_order_screen_not_rigorous_error_bound",
        "maximum_slave_faces": _FACE_BUDGET,
        "maximum_points_per_face": _FACE_POINT_BUDGET,
        "maximum_trial_points": _GLOBAL_POINT_BUDGET,
    }


def _radial_coordinate_safety(positions, center, radius: float) -> float:
    """Conservative length uncertainty from subtracting world coordinates."""

    scale = max(
        float(np.max(np.abs(positions), initial=0.0)),
        float(np.max(np.abs(center), initial=0.0)),
        float(radius),
        np.finfo(float).tiny,
    )
    return 128.0 * np.finfo(float).eps * scale


def radial_active_triangle_quadrature(
    positions,
    *,
    center,
    radius: float,
    axis=None,
    admissible_side: str = "exterior",
    order: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Positive barycentric rule for the active part of a radial P1 face.

    The weights integrate on the unit reference triangle of area 1/2.  An
    exterior surface is active where radial distance is smaller than radius;
    an interior surface has the opposite inequality.  An axial cylinder uses
    the orthogonal projector to its axis.  This rule by itself provides no
    integration-error certificate or Step admission for a mixed active face.
    """

    vertices = np.asarray(positions, dtype=float)
    origin = np.asarray(center, dtype=float)
    selected_radius = float(radius)
    if (
        vertices.shape != (3, 3)
        or origin.shape != (3,)
        or not np.all(np.isfinite(vertices))
        or not np.all(np.isfinite(origin))
        or not np.isfinite(selected_radius)
        or selected_radius <= 0.0
    ):
        raise ValueError("Radial partition requires finite 3D P1 geometry and radius.")
    if admissible_side not in {"exterior", "interior"}:
        raise ValueError("Radial partition needs an exterior or interior side.")
    if type(order) is not int or not 2 <= order <= 64:
        raise ValueError("Radial partition Gauss order must be an integer in [2, 64].")
    if axis is None:
        projector = np.eye(3)
    else:
        direction = np.asarray(axis, dtype=float)
        if (
            direction.shape != (3,)
            or not np.all(np.isfinite(direction))
            or not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-12, rtol=0.0)
        ):
            raise ValueError("Radial partition requires a finite unit cylinder axis.")
        projector = np.eye(3) - np.outer(direction, direction)

    relative = (vertices - origin) @ projector
    u0 = relative[0]
    u1 = relative[1] - u0
    u2 = relative[2] - u0
    a = float(u2 @ u2)
    b0 = 2.0 * float(u0 @ u2)
    b1 = 2.0 * float(u1 @ u2)
    c0 = float(u0 @ u0) - selected_radius**2
    c1 = 2.0 * float(u0 @ u1)
    c2 = float(u1 @ u1)

    return quadratic_region_triangle_quadrature(
        QuadraticReferenceLevelSet(a, b0, b1, c0, c1, c2),
        negative=admissible_side == "exterior",
        order=order,
    )


def _minimum_radial_distance(relative: np.ndarray) -> float:
    """Distance from the origin to a projected affine triangle (including edges)."""

    candidates = [float(vertex @ vertex) for vertex in relative]
    for first, second in ((0, 1), (1, 2), (2, 0)):
        start = relative[first]
        direction = relative[second] - start
        squared_length = float(direction @ direction)
        fraction = (
            0.0
            if squared_length <= 0.0
            else np.clip(-float(start @ direction) / squared_length, 0.0, 1.0)
        )
        closest = start + fraction * direction
        candidates.append(float(closest @ closest))
    edges = np.column_stack((relative[1] - relative[0], relative[2] - relative[0]))
    gram = edges.T @ edges
    if np.linalg.det(gram) > 256.0 * np.finfo(float).eps * max(
        np.max(np.abs(gram)) ** 2,
        np.finfo(float).tiny,
    ):
        barycentric = np.linalg.solve(gram, -edges.T @ relative[0])
        if np.all(barycentric >= 0.0) and np.sum(barycentric) <= 1.0:
            closest = relative[0] + edges @ barycentric
            candidates.append(float(closest @ closest))
    return float(np.sqrt(max(0.0, min(candidates))))


def _radial_local_response(
    shapes,
    unit_weights,
    current,
    reference_area,
    *,
    center,
    axis,
    radius,
    side,
    penalty,
):
    if len(shapes) == 0:
        return np.zeros(1 + 9 + 81)
    weights = 2.0 * reference_area * unit_weights
    points = shapes @ current
    offsets = points - center
    projector = np.eye(3) if axis is None else np.eye(3) - np.outer(axis, axis)
    radial = offsets @ projector
    distances = np.linalg.norm(radial, axis=1)
    if np.any(distances <= 1.0e-12 * max(radius, float(np.max(distances)))):
        raise ValueError("Radial contact reaches an undefined axis/center normal.")
    unit = radial / distances[:, None]
    gaps = side * (distances - radius)
    normals = side * unit
    hessian = (
        side
        / distances[:, None, None]
        * (projector[None, :, :] - unit[:, :, None] * unit[:, None, :])
    )
    active = gaps < 0.0
    weights = weights * active
    potential = 0.5 * penalty * np.dot(weights, gaps**2)
    residual = penalty * np.einsum(
        "q,qa,qi->ai",
        weights * gaps,
        shapes,
        normals,
    )
    point_tangent = (
        normals[:, :, None] * normals[:, None, :] + gaps[:, None, None] * hessian
    )
    tangent = penalty * np.einsum(
        "q,qa,qb,qij->aibj",
        weights,
        shapes,
        shapes,
        point_tangent,
    )
    return np.concatenate(([potential], residual.reshape(-1), tangent.reshape(-1)))


def partition_radial_p1_trace(
    trace: ContactTrace,
    nodal_positions,
    surface: RigidSphereSurface | RigidCylinderSurface,
    *,
    penalty: float,
    motion=None,
    factor: float = 0.0,
    relative_tolerance: float = _DEFAULT_RELATIVE_TOLERANCE,
) -> ContactTrace:
    """Bounded experimental mixed radial partition for a frictionless P1 trace.

    The geometry boundary is found by conic roots, not contact-point sampling.
    Comparing 12/24-point rules checks potential, full nodal residual and
    tangent on every face; it is an *a posteriori numerical screen*, not a
    mathematical quadrature-error certificate.  An inconclusive face refuses
    the trial.  Dynamic point IDs are ephemeral and may not carry frictional
    State.  This function does not independently admit any public Step mode.
    """

    if (
        not isinstance(trace, ContactTrace)
        or trace.measure_configuration != "reference"
    ):
        raise TypeError("Radial partition requires a reference ContactTrace.")
    if not isinstance(surface, (RigidSphereSurface, RigidCylinderSurface)):
        raise TypeError("Radial partition requires an analytical sphere or cylinder.")
    if surface.dimension != 3:
        raise NotImplementedError("Radial P1-face partition requires a 3D surface.")
    selected_penalty = float(penalty)
    if not np.isfinite(selected_penalty) or selected_penalty <= 0.0:
        raise ValueError("Radial partition requires a positive scalar penalty.")
    if not np.isfinite(relative_tolerance) or not 0.0 < relative_tolerance <= 1.0e-2:
        raise ValueError("Radial partition tolerance must lie in (0, 1e-2].")
    ids = np.asarray(trace.point_ids, dtype=np.int64)
    nodes = np.asarray(trace.node_ids, dtype=np.int64)
    weights = np.asarray(trace.weights, dtype=float)
    positions = np.asarray(nodal_positions, dtype=float)
    expected_shapes = np.asarray(
        (
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
            (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
        )
    )
    if (
        ids.size % 3 != 0
        or ids.size // 3 > _FACE_BUDGET
        or nodes.shape[1] != 3
        or positions.ndim != 2
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
        or (nodes.size and int(nodes.max()) >= len(positions))
        or not np.all(ids.reshape(-1, 3) % 3 == np.arange(3)[None, :])
        or not np.all(nodes.reshape(-1, 3, 3) == nodes.reshape(-1, 3, 3)[:, :1, :])
        or not np.allclose(
            trace.shape_values.reshape(-1, 3, 3),
            expected_shapes,
            rtol=0.0,
            atol=1.0e-13,
        )
        or not np.allclose(
            weights.reshape(-1, 3).sum(axis=1), 3.0 * weights.reshape(-1, 3)[:, 0]
        )
    ):
        raise NotImplementedError(
            "Radial partition requires at most 128 default P1 triangle faces."
        )
    if ids.size and int(ids.max()) // 3 > np.iinfo(np.int64).max // _FACE_POINT_BUDGET:
        raise OverflowError("Radial partition point identity exceeds int64.")
    if isinstance(surface, RigidSphereSurface):
        center, axis = _transformed_point_and_direction(
            np.asarray(surface.center),
            None,
            motion=motion,
            factor=factor,
        )
    else:
        center, axis = _transformed_point_and_direction(
            np.asarray(surface.axis_point),
            np.asarray(surface.axis_direction),
            motion=motion,
            factor=factor,
        )
    side = 1.0 if surface.admissible_side == "exterior" else -1.0
    projector = np.eye(3) if axis is None else np.eye(3) - np.outer(axis, axis)
    coordinate_safety = _radial_coordinate_safety(positions, center, surface.radius)
    if coordinate_safety > relative_tolerance * surface.radius:
        raise NotImplementedError(
            "AFM-CONTACT-RADIAL-PARTITION: world-coordinate roundoff is too "
            "large relative to the analytical radius and requested tolerance."
        )
    new_ids, new_nodes, new_shapes, new_weights = [], [], [], []
    any_active = False
    face_rows = nodes.reshape(-1, 3, 3)[:, 0, :]

    def append_face(face_index, row, basis, local_weights):
        if len(basis) > _FACE_POINT_BUDGET:
            raise NotImplementedError("Radial partition point budget exceeded.")
        face_id = int(ids[3 * face_index] // 3)
        for index, (shape, weight) in enumerate(zip(basis, local_weights, strict=True)):
            new_ids.append(face_id * _FACE_POINT_BUDGET + index)
            new_nodes.append(row)
            new_shapes.append(shape)
            new_weights.append(weight)

    for face_index, row in enumerate(face_rows):
        current = positions[row]
        reference_area = float(np.sum(weights[3 * face_index : 3 * face_index + 3]))
        if reference_area <= 0.0:
            raise ValueError("Radial partition reference area must be positive.")
        relative = (current - center) @ projector
        minimum_radial = _minimum_radial_distance(relative)
        maximum_radial = float(np.max(np.linalg.norm(relative, axis=1)))
        if minimum_radial <= max(
            1.0e-12 * max(surface.radius, maximum_radial), coordinate_safety
        ):
            raise NotImplementedError(
                "AFM-CONTACT-RADIAL-PARTITION: a slave face can reach the "
                "undefined sphere center or cylinder axis."
            )
        if (
            isinstance(surface, RigidCylinderSurface)
            and surface.axial_interval is not None
        ):
            axial = (current - center) @ axis
            low, high = surface.axial_interval
            margin = coordinate_safety + 64.0 * np.finfo(float).eps * max(
                surface.radius, abs(low), abs(high), float(np.max(np.abs(axial)))
            )
            if np.min(axial) <= low + margin or np.max(axial) >= high - margin:
                raise NotImplementedError(
                    "AFM-CONTACT-RADIAL-PARTITION: a slave face can reach an "
                    "unsupported cylinder rim."
                )
        radial_margin = coordinate_safety + 128.0 * np.finfo(float).eps * max(
            surface.radius, maximum_radial
        )
        definitely_open = (
            minimum_radial > surface.radius + radial_margin
            if side > 0.0
            else maximum_radial < surface.radius - radial_margin
        )
        if definitely_open:
            append_face(
                face_index,
                row,
                trace.shape_values[3 * face_index : 3 * face_index + 3],
                weights[3 * face_index : 3 * face_index + 3],
            )
            continue
        low_shapes, low_weights = radial_active_triangle_quadrature(
            current,
            center=center,
            axis=axis,
            radius=surface.radius,
            admissible_side=surface.admissible_side,
            order=_LOW_ORDER,
        )
        high_shapes, high_weights = radial_active_triangle_quadrature(
            current,
            center=center,
            axis=axis,
            radius=surface.radius,
            admissible_side=surface.admissible_side,
            order=_HIGH_ORDER,
        )
        possible_active = (
            minimum_radial < surface.radius
            if side > 0.0
            else maximum_radial > surface.radius
        )
        if possible_active and len(high_shapes) == 0:
            raise NotImplementedError(
                "AFM-CONTACT-RADIAL-PARTITION: an active patch is too small "
                "for the bounded positive quadrature rule."
            )
        lower = _radial_local_response(
            low_shapes,
            low_weights,
            current,
            reference_area,
            center=center,
            axis=axis,
            radius=surface.radius,
            side=side,
            penalty=selected_penalty,
        )
        higher = _radial_local_response(
            high_shapes,
            high_weights,
            current,
            reference_area,
            center=center,
            axis=axis,
            radius=surface.radius,
            side=side,
            penalty=selected_penalty,
        )
        low_area = 2.0 * reference_area * float(np.sum(low_weights))
        high_area = 2.0 * reference_area * float(np.sum(high_weights))
        area_error = abs(high_area - low_area)
        if area_error > relative_tolerance * high_area + (
            256.0 * np.finfo(float).eps * reference_area
        ):
            raise NotImplementedError(
                "AFM-CONTACT-RADIAL-PARTITION: active-area quadrature "
                "did not pass its per-face 12/24-order screen."
            )
        maximum_penetration = max(
            0.0,
            surface.radius - minimum_radial
            if side > 0.0
            else maximum_radial - surface.radius,
        )
        scale = selected_penalty * high_area
        channel_scales = np.concatenate(
            (
                [0.5 * scale * maximum_penetration**2],
                np.full(9, scale * maximum_penetration),
                np.full(81, scale * (1.0 + maximum_penetration / minimum_radial)),
            )
        )
        error = np.abs(higher - lower)
        allowed = relative_tolerance * np.maximum(
            channel_scales,
            np.abs(higher),
        ) + 256.0 * np.finfo(float).eps * selected_penalty * reference_area * np.array(
            [surface.radius**2, *([surface.radius] * 9), *([1.0] * 81)]
        )
        if np.any(error > allowed):
            raise NotImplementedError(
                "AFM-CONTACT-RADIAL-PARTITION: mixed smooth contact quadrature "
                "did not pass its per-face 12/24-order residual/tangent screen."
            )
        if len(high_shapes) == 0:
            basis = trace.shape_values[3 * face_index : 3 * face_index + 3]
            local_weights = weights[3 * face_index : 3 * face_index + 3]
        else:
            any_active = True
            basis = high_shapes
            local_weights = 2.0 * reference_area * high_weights
        append_face(face_index, row, basis, local_weights)
    if not any_active:
        return trace
    if len(new_ids) > _GLOBAL_POINT_BUDGET:
        raise NotImplementedError("Radial partition global point budget exceeded.")
    return ContactTrace(
        point_ids=np.asarray(new_ids, dtype=np.int64),
        node_ids=np.asarray(new_nodes, dtype=np.int64).reshape(-1, 3),
        shape_values=np.asarray(new_shapes, dtype=float).reshape(-1, 3),
        weights=np.asarray(new_weights, dtype=float),
        measure_configuration="reference",
        source=f"{trace.source}_trial_radial_active_partition",
    )


__all__ = [
    "radial_active_triangle_quadrature",
    "partition_radial_p1_trace",
    "radial_partition_policy_summary",
]
