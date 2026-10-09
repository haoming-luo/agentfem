# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Affine planar Q1 traces with triangulated integration, never P1 field substitution."""

from dataclasses import dataclass
from hashlib import sha256

import basix
import numpy as np

from ._interface_overlap import planar_overlap_pairing
from ._interface_pairing import _make_pairing
from .boundary_models.rigid import TriangulatedRigidSurface


@dataclass(frozen=True, init=False)
class QuadrilateralReferenceTrace:
    """Parallelogram Q1 facets with cyclic connectivity and fixed geometry.

    Triangles are a search/integration partition only. Basis evaluation always
    uses all four parent nodes and Basix Q1 interpolation. General bilinear,
    warped or curved maps need separate inversion and quadrature admission.
    """

    vertices: np.ndarray
    quadrilaterals: np.ndarray
    triangulated: TriangulatedRigidSurface
    geometry_fingerprint: str
    nodal_measures: np.ndarray

    def __init__(self, vertices, quadrilaterals, *, tolerance):
        vertices = np.asarray(vertices, dtype=float)
        quads = np.asarray(quadrilaterals)
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Quadrilateral tolerance must be positive finite.")
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not np.all(np.isfinite(vertices))
        ):
            raise ValueError("Quadrilateral vertices must be finite (nodes, 3).")
        if (
            quads.ndim != 2
            or quads.shape[1] != 4
            or not len(quads)
            or quads.dtype.kind not in "iu"
        ):
            raise ValueError(
                "Quadrilateral connectivity must be nonempty integer (facets, 4)."
            )
        if np.any(quads < 0) or np.any(quads >= len(vertices)):
            raise ValueError("Quadrilateral node index out of range.")
        points = vertices[quads]
        if any(len(set(row)) != 4 for row in quads):
            raise ValueError("Quadrilateral nodes must be distinct.")
        mismatch = points[:, 0] + points[:, 2] - points[:, 1] - points[:, 3]
        scale = np.linalg.norm(np.ptp(points, axis=1), axis=1)
        # Geometric coincidence tolerance cannot relax the affine-map restriction.
        if np.any(
            np.linalg.norm(mismatch, axis=1) > np.minimum(tolerance, 1e-11 * scale)
        ):
            raise NotImplementedError(
                "Q1 trace currently requires affine planar parallelograms."
            )
        triangles = quads[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)
        surface = TriangulatedRigidSurface(vertices, triangles)
        area = np.linalg.norm(
            np.cross(points[:, 1] - points[:, 0], points[:, 3] - points[:, 0]), axis=1
        )
        measures = np.zeros(len(vertices))
        np.add.at(measures, quads.ravel(), np.repeat(area / 4, 4))
        quads = np.array(quads, dtype=np.int64, copy=True)
        quads.setflags(write=False)
        measures.setflags(write=False)
        digest = sha256(b"agentfem.affine-q1-reference-trace.v1")
        digest.update(surface.geometry_fingerprint.encode())
        digest.update(quads.astype("<i8").tobytes())
        for name, value in dict(
            vertices=surface.vertices,
            quadrilaterals=quads,
            triangulated=surface,
            geometry_fingerprint=digest.hexdigest(),
            nodal_measures=measures,
        ).items():
            object.__setattr__(self, name, value)

    @property
    def triangles(self):
        return self.triangulated.triangles


def _q1_values(trace, triangle_nodes, points):
    lookup = {tuple(nodes): i // 2 for i, nodes in enumerate(trace.triangles)}
    parents = np.array([lookup[tuple(nodes)] for nodes in triangle_nodes])
    nodes = trace.quadrilaterals[parents]
    vertices = trace.vertices[nodes]
    axes = np.stack(
        (vertices[:, 1] - vertices[:, 0], vertices[:, 3] - vertices[:, 0]), axis=2
    )
    gram = np.einsum("qia,qib->qab", axes, axes)
    rhs = np.einsum("qia,qi->qa", axes, points - vertices[:, 0])
    local = np.linalg.solve(gram, rhs[..., None])[..., 0]
    if np.any(local < -1e-9) or np.any(local > 1 + 1e-9):
        raise ValueError("Quadrilateral integration point lies outside its parent.")
    element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.quadrilateral,
        1,
        basix.LagrangeVariant.equispaced,
    )
    # Basix tensor-product node order -> declared cyclic trace connectivity.
    weights = element.tabulate(0, local)[0, :, :, 0][:, [0, 1, 3, 2]]
    return nodes, weights


def quadrilateral_overlap_pairing(
    negative, positive, *, tolerance, maximum_points=200_000
):
    if not all(
        isinstance(s, QuadrilateralReferenceTrace) for s in (negative, positive)
    ):
        raise TypeError("Expected affine Q1 reference traces on both sides.")
    integration = planar_overlap_pairing(
        negative.triangulated,
        positive.triangulated,
        tolerance=tolerance,
        maximum_points=maximum_points,
        quadrature_degree=4,
    )
    query = np.einsum(
        "qi,qij->qj",
        integration.negative_weights,
        negative.vertices[integration.negative_nodes],
    )
    nn, wn = _q1_values(negative, integration.negative_nodes, query)
    pn, wp = _q1_values(positive, integration.positive_nodes, query)
    mismatch = np.einsum("qi,qij->qj", wp, positive.vertices[pn]) - np.einsum(
        "qi,qij->qj", wn, negative.vertices[nn]
    )
    if np.max(np.linalg.norm(mismatch, axis=1)) > tolerance:
        raise ValueError(
            "Quadrilateral reference interpolation violates coincidence tolerance."
        )
    return _make_pairing(
        negative,
        positive,
        tolerance,
        {
            "negative_nodes": nn,
            "positive_nodes": pn,
            "negative_weights": wn,
            "positive_weights": wp,
            "weights": integration.weights,
            "normals": integration.normals,
            "reference_mismatch": mismatch,
        },
        "coplanar-affine-q1-common-refinement",
    )
