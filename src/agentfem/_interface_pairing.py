# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-reference nonmatching trace quadrature (experimental internal layer).

This module owns interpolation and its work-conjugate transpose, not a
cohesive law, time integrator, or MPI owner schedule. Triangle surfaces are
reference snapshots only; reusing their projector does not make either body
rigid. The initial route requires coincident reference surfaces. Finite gaps,
finite-rotation constitutive frames and moving re-pairing are not implemented.
"""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .boundary_models.rigid import TriangulatedRigidSurface
from .boundary_models.search import TriangleSurfaceBVH


_TRIANGLE_RULE = np.array(
    [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]
)


def _subdivided_triangle_rule(level):
    """Uniform reference subdivision, preserving the parent P1 trace basis."""
    if (
        isinstance(level, bool)
        or not isinstance(level, (int, np.integer))
        or not 0 <= level <= 6
    ):
        raise ValueError("quadrature_refinement must be an integer between 0 and 6.")
    cells = np.eye(3)[None, :, :]
    for _ in range(level):
        a, b, c = cells[:, 0], cells[:, 1], cells[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        cells = np.concatenate(
            [
                np.stack(v, axis=1)
                for v in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
            ]
        )
    return np.einsum("qi,fij->fqj", _TRIANGLE_RULE, cells).reshape(-1, 3)


def _field(value, count, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (count, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape ({count}, 3).")
    return result


@dataclass(frozen=True, init=False)
class FixedReferencePairing:
    """Factory-built trace map with independent node tables on both sides.

    ``jump = u_positive - u_negative``. ``residual`` returns the derivative
    of interface energy, NOT the restoring force (its negative). Reference
    moment balance follows only to the declared coincidence tolerance.
    """

    negative: TriangulatedRigidSurface
    positive: TriangulatedRigidSurface
    negative_nodes: np.ndarray
    positive_nodes: np.ndarray
    negative_weights: np.ndarray
    positive_weights: np.ndarray
    weights: np.ndarray
    normals: np.ndarray
    reference_mismatch: np.ndarray
    tolerance: float
    fingerprint: str
    method: str

    def jump(self, negative_displacement, positive_displacement):
        un = _field(
            negative_displacement, len(self.negative.vertices), "negative displacement"
        )
        up = _field(
            positive_displacement, len(self.positive.vertices), "positive displacement"
        )
        return np.einsum(
            "qi,qij->qj", self.positive_weights, up[self.positive_nodes]
        ) - np.einsum("qi,qij->qj", self.negative_weights, un[self.negative_nodes])

    def residual(self, traction):
        """Scatter global work-conjugate traction using the exact trace transpose."""
        t = _field(traction, len(self.weights), "traction") * self.weights[:, None]
        rn = np.zeros_like(self.negative.vertices)
        rp = np.zeros_like(self.positive.vertices)
        np.add.at(
            rn,
            self.negative_nodes.ravel(),
            (-self.negative_weights[:, :, None] * t[:, None, :]).reshape(-1, 3),
        )
        np.add.at(
            rp,
            self.positive_nodes.ravel(),
            (self.positive_weights[:, :, None] * t[:, None, :]).reshape(-1, 3),
        )
        return rn, rp

    def summary(self):
        return {
            "schema": "agentfem.fixed-reference-pairing.v1",
            "fingerprint": self.fingerprint,
            "method": self.method,
            "integration_side": (
                "common-refinement"
                if self.method.startswith("coplanar-")
                else "negative"
            ),
            "quadrature_points": len(self.weights),
            "mean_quadrature_points_per_negative_facet": len(self.weights)
            / len(getattr(self.negative, "quadrilaterals", self.negative.triangles)),
            "negative_basis": "Q1" if self.negative_nodes.shape[1] == 4 else "P1",
            "positive_basis": "Q1" if self.positive_nodes.shape[1] == 4 else "P1",
            "reference_area": float(self.weights.sum()),
            "maximum_reference_mismatch": float(
                np.linalg.norm(self.reference_mismatch, axis=1).max()
            ),
            "coincidence_tolerance": self.tolerance,
            "normal_convention": "negative-side-oriented-normal",
            "execution_scope": "process-local-trace-operator",
            "coverage": (
                "coplanar-per-facet-area-and-self-overlap-checked"
                if self.method.startswith("coplanar-")
                else "integration-points-only-not-overlap-proof"
            ),
            "finite_rotation_law": False,
            "global_step_integrated": False,
        }

    def tangent_action(self, point_tangent, negative_increment, positive_increment):
        """Apply B^T W D B without allocating a dense interface matrix.

        ``point_tangent`` is global d(traction)/d(jump), evaluated at fixed
        committed history by the constitutive owner. No symmetry is assumed.
        """
        tangent = np.asarray(point_tangent, dtype=float)
        if tangent.shape != (len(self.weights), 3, 3) or not np.all(
            np.isfinite(tangent)
        ):
            raise ValueError("point tangent must be finite with shape (points, 3, 3).")
        delta = self.jump(negative_increment, positive_increment)
        return self.residual(np.einsum("qij,qj->qi", tangent, delta))

    def constant_traction_audit(self):
        """Compare each trace's nodal measures to independent surface integrals.

        This necessary patch check is not an overlap proof, nor an error
        estimator for arbitrary traction. It never silently refines a map:
        changing quadrature also changes the constitutive state identity.
        """
        report = {}
        for side in ("negative", "positive"):
            surface = getattr(self, side)
            vertices = surface.vertices[surface.triangles]
            areas = 0.5 * np.linalg.norm(
                np.cross(
                    vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
                ),
                axis=1,
            )
            if hasattr(surface, "nodal_measures"):
                expected = surface.nodal_measures
            else:
                expected = np.zeros(len(surface.vertices))
                np.add.at(expected, surface.triangles.ravel(), np.repeat(areas / 3, 3))
            actual = np.zeros_like(expected)
            np.add.at(
                actual,
                getattr(self, side + "_nodes").ravel(),
                (getattr(self, side + "_weights") * self.weights[:, None]).ravel(),
            )
            error = actual - expected
            report[side] = {
                "absolute_nodal_measure_error_l2": float(np.linalg.norm(error)),
                "relative_nodal_measure_error_l2": float(
                    np.linalg.norm(error) / np.linalg.norm(expected)
                ),
                "surface_area": float(areas.sum()),
                "integrated_area": float(actual.sum()),
            }
        return {
            "check": "constant-traction-nodal-patch",
            "coverage_proven": False,
            "sides": report,
        }


def fixed_reference_pairing(
    negative, positive, *, tolerance, quadrature_refinement=0, maximum_points=200_000
):
    """Project a possibly subdivided negative-side quadrature to the positive side.

    Side choice is explicit: finer triangulation alone is not a guarantee of
    integration accuracy. Both surfaces must have opposing, locally parallel
    oriented normals. Ambiguous, distant or incompatible projections fail
    before a usable map is returned. The three-point rule is not exact when
    a projected triangle crosses a positive-side element boundary; refinement
    and side-swap sensitivity must be checked by the consumer.
    """
    if not isinstance(negative, TriangulatedRigidSurface) or not isinstance(
        positive, TriangulatedRigidSurface
    ):
        raise TypeError("Pairing requires reviewed triangular reference surfaces.")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("coincidence tolerance must be finite and positive.")
    rule = _subdivided_triangle_rule(quadrature_refinement)
    if (
        isinstance(maximum_points, bool)
        or not isinstance(maximum_points, (int, np.integer))
        or maximum_points < 1
    ):
        raise ValueError("maximum_points must be a positive integer.")
    count = len(rule)
    if count * len(negative.triangles) > maximum_points:
        raise ValueError(
            "Interface quadrature exceeds maximum_points; reduce refinement or partition the surface."
        )
    nodes = np.repeat(negative.triangles, count, axis=0)
    shape = np.tile(rule, (len(negative.triangles), 1))
    query = np.einsum("qi,qij->qj", shape, negative.vertices[nodes])
    projection = (
        TriangleSurfaceBVH(positive)
        .project_with_diagnostics(query, maximum_distance=tolerance)
        .projection
    )
    if not np.all(projection.valid):
        failed = np.flatnonzero(~projection.valid)
        raise ValueError(
            f"Interface projection failed at quadrature points {failed[:8].tolist()}."
        )
    lookup = {int(key): index for index, key in enumerate(positive.facet_ids)}
    indices = np.array([lookup[int(key)] for key in projection.entity_ids])
    normals = np.repeat(negative.facet_normals, count, axis=0)
    if not np.allclose(projection.normals, -normals, rtol=0, atol=1e-8):
        raise ValueError("Interface normals must be opposing and locally parallel.")
    xyz = negative.vertices[negative.triangles]
    area = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2
    )
    values = {
        "negative_nodes": nodes,
        "positive_nodes": positive.triangles[indices],
        "negative_weights": shape,
        "positive_weights": projection.local_coordinates,
        "weights": np.repeat(area / count, count),
        "normals": normals,
        "reference_mismatch": projection.closest_points - query,
    }
    return _make_pairing(
        negative,
        positive,
        tolerance,
        values,
        "single-sided-triangle-projected-quadrature",
    )


def _make_pairing(negative, positive, tolerance, values, method):
    """Seal trace arrays and bind method identity for all reference factories."""
    result = object.__new__(FixedReferencePairing)
    digest = sha256(b"agentfem.fixed-reference-pairing.v1")
    digest.update(method.encode())
    digest.update(negative.geometry_fingerprint.encode())
    digest.update(positive.geometry_fingerprint.encode())
    digest.update(np.asarray([tolerance], dtype="<f8").tobytes())
    for name, value in values.items():
        array = np.array(value, copy=True)
        array.setflags(write=False)
        object.__setattr__(result, name, array)
        digest.update(name.encode())
        digest.update(
            array.astype("<i8" if array.dtype.kind in "iu" else "<f8").tobytes()
        )
    for name, value in dict(
        negative=negative,
        positive=positive,
        tolerance=tolerance,
        fingerprint=digest.hexdigest(),
        method=method,
    ).items():
        object.__setattr__(result, name, value)
    return result


@dataclass(frozen=True)
class NonmatchingResponse:
    """Local reference-area response; not a SimulationResult or maturity claim."""

    negative_residual: np.ndarray
    positive_residual: np.ndarray
    traction: np.ndarray
    jump: np.ndarray
    damage: np.ndarray
    stored_energy: float
    dissipated_energy: float


class FixedReferenceCohesiveAssembler:
    """Small-displacement cohesive consumer of the fixed trace map.

    Shares the existing law/transaction and basis implementation. Global dof
    lowering, portable MPI state and finite-rotation response remain separate
    gates. Snapshot restore binds law and exact pairing identity before state
    mutation; it is intentionally not rank-count-portable yet.
    """

    def __init__(self, pairing, law, *, tangential="free", tangential_stiffness=None):
        from .interfaces import _transaction_for_law, _validate_tangential_mode

        if not isinstance(pairing, FixedReferencePairing):
            raise TypeError("Assembler requires FixedReferencePairing.")
        self.pairing = pairing
        self.law = law
        self.tangential, self.tangential_stiffness = _validate_tangential_mode(
            tangential, tangential_stiffness, law
        )
        self.state = _transaction_for_law(law, len(pairing.weights))
        configure = getattr(self.state, "configure_dimension", None)
        if callable(configure):
            configure(3)
        self._trial = None
        # Many integration points share the same pair of trace nodes. Aggregate
        # their small blocks before PETSc insertion, never a dense global matrix.
        rows = np.concatenate((pairing.negative_nodes, pairing.positive_nodes), axis=1)
        self._block_nodes, groups = np.unique(rows, axis=0, return_inverse=True)
        self._block_order = np.argsort(groups, kind="stable")
        self._block_offsets = np.concatenate(([0], np.cumsum(np.bincount(groups))))

    def _point_response(self, negative, positive, *, begin):
        from .interfaces import _point_interface_response

        return _point_interface_response(
            state=self.state,
            law=self.law,
            jump_global=self.pairing.jump(negative, positive)[:, None, :],
            normals=self.pairing.normals,
            tangential=self.tangential,
            tangential_stiffness=self.tangential_stiffness,
            begin=begin,
        )

    def begin(self, negative, positive):
        try:
            point = self._point_response(negative, positive, begin=True)
            self._trial = self._response(point)
            return self._trial
        except Exception:
            self.rollback()
            raise

    def evaluate(self, negative, positive):
        """Read the current response at fixed committed history, without a trial."""
        return self._response(self._point_response(negative, positive, begin=False))

    def _response(self, point):
        traction = point["traction_vector"][:, 0]
        rn, rp = self.pairing.residual(traction)
        stored = float(self.pairing.weights @ point["stored_energy"][:, 0])
        dissipated = float(self.pairing.weights @ point["dissipated_energy"][:, 0])
        if not np.all(np.isfinite([stored, dissipated])):
            raise ValueError("Non-finite cohesive energy.")
        return NonmatchingResponse(
            rn,
            rp,
            traction.copy(),
            point["jump"][:, 0].copy(),
            point["damage"][:, 0].copy(),
            stored,
            dissipated,
        )

    def tangent_action(self, negative, positive, delta_negative, delta_positive):
        point = self._point_response(negative, positive, begin=False)
        return self.pairing.tangent_action(
            point["tangent"][:, 0], delta_negative, delta_positive
        )

    def commit(self):
        if self._trial is None:
            raise RuntimeError("No nonmatching cohesive trial to commit.")
        self.state.commit()
        self._trial = None

    def tangent_blocks(self, negative, positive):
        """Yield integrated trace-pair blocks; never a dense global matrix."""
        point = self._point_response(negative, positive, begin=False)
        pair = self.pairing
        width = pair.negative_nodes.shape[1]
        for group, nodes in enumerate(self._block_nodes):
            q = self._block_order[
                self._block_offsets[group] : self._block_offsets[group + 1]
            ]
            shape = np.concatenate(
                (-pair.negative_weights[q], pair.positive_weights[q]), axis=1
            )
            weighted = pair.weights[q, None, None] * point["tangent"][q, 0]
            left = (shape[:, :, None, None] * weighted[:, None, :, :]).reshape(
                len(q), -1
            )
            matrix = (
                (left.T @ shape)
                .reshape(len(nodes), 3, 3, len(nodes))
                .transpose(0, 1, 3, 2)
            )
            yield (
                nodes[:width],
                nodes[width:],
                matrix.reshape(3 * len(nodes), 3 * len(nodes)),
            )

    def rollback(self):
        self.state.rollback()
        self._trial = None

    def snapshot(self):
        return {
            "schema": "agentfem.fixed-reference-cohesive-state.v1",
            "pairing": self.pairing.fingerprint,
            "law": self.law.summary(),
            "tangential": self.tangential,
            "tangential_stiffness": self.tangential_stiffness,
            "state": self.state.snapshot(),
        }

    def restore(self, snapshot):
        if self._trial is not None:
            raise RuntimeError("Rollback or commit the trial before restore.")
        expected = self.snapshot()
        if set(snapshot) != set(expected) or any(
            snapshot[key] != expected[key] for key in expected if key != "state"
        ):
            raise ValueError("Nonmatching cohesive checkpoint identity mismatch.")
        try:
            self.state.restore(snapshot["state"])
        except Exception:
            self.state.restore(expected["state"])
            raise
