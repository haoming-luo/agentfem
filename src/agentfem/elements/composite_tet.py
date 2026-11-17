# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Affine composite Tet10 local operators, independently integrated.

Piecewise-linear displacement on twelve subtetrahedra, with the centre tied
kinematically to the six edge nodes. Gradients are L2-projected onto parent P1.
Mean-dilatation stiffness adds independently derived deviatoric projection-residual
energy stabilization; it is not asserted identical to the published element.
Reference for the composite topology/projection: Ostien et al. (2016), DOI 10.1002/nme.5218; Sierra/SM Theory 15.3.
This formulation-development kernel is NOT an Abaqus C3D10M implementation,
not a Basix P2 element. The composite_tet10_explicit provider supplies whole-mesh execution.
"""

from dataclasses import dataclass
from functools import lru_cache
import numpy as np

_EDGES = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])
_SUBTETS = np.array(
    [
        [0, 4, 6, 7],
        [1, 5, 4, 8],
        [2, 6, 5, 9],
        [3, 8, 7, 9],
        [4, 8, 5, 10],
        [5, 8, 9, 10],
        [9, 8, 7, 10],
        [7, 8, 4, 10],
        [4, 5, 6, 10],
        [5, 9, 6, 10],
        [9, 7, 6, 10],
        [7, 4, 6, 10],
    ]
)
_A, _B = (5 + 3 * np.sqrt(5)) / 20, (5 - np.sqrt(5)) / 20
_QUAD = np.full((4, 4), _B)
np.fill_diagonal(_QUAD, _A)


def _readonly(value):
    value = np.array(value, dtype=float, copy=True)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class CompositeTet10:
    """Local operator data in Abaqus Tet10 node order, node-major xyz DOFs.

    ``gradient_coefficients[a,i,j]`` gives derivative j of shape i at
    parent vertex a. ``scalar_mass`` is the consistent mass for unit density.
    Engineering-strain order is xx, yy, zz, xy, yz, xz.
    """

    coordinates: np.ndarray
    volume: float
    gradient_coefficients: np.ndarray
    scalar_mass: np.ndarray
    fluctuation_strains: np.ndarray
    fluctuation_weights: np.ndarray

    def gradients(self, barycentric):
        """Projected gradients at parent barycentric points (not P2 gradients)."""
        p = np.asarray(barycentric, dtype=float)
        if (
            p.shape[-1:] != (4,)
            or not np.isfinite(p).all()
            or not np.allclose(p.sum(axis=-1), 1, atol=1e-12, rtol=0)
            or np.any(p < -1e-12)
        ):
            raise ValueError(
                "AFM-COMPOSITE-TET: points must be tetrahedral barycentric coordinates."
            )
        return np.einsum("...a,aij->...ij", p, self.gradient_coefficients)

    def mass(self, density, *, lumped=True):
        """Return node-major vector mass; row-sum lumping is positive here."""
        density = float(density)
        if not np.isfinite(density) or density <= 0:
            raise ValueError("AFM-COMPOSITE-TET: density must be finite and positive.")
        m = density * self.scalar_mass
        return np.repeat(m.sum(axis=1), 3) if lumped else np.kron(m, np.eye(3))

    def critical_time_step(self, constitutive, density, *, mean_dilatation=False):
        """Central-difference local bound 2/sqrt(lambda_max(M_lump^-1 K)).

        Requires the stabilized linear kernel and positive lumped mass. A global
        explicit driver still owns constraints, time-step safety and other
        stiffness contributions; this is a local element bound, not a run dt.
        """
        m = self.mass(density)
        K = self.stiffness(constitutive, mean_dilatation=mean_dilatation)
        invsqrt = 1 / np.sqrt(m)
        maximum = np.linalg.eigvalsh(invsqrt[:, None] * K * invsqrt[None, :])[-1]
        return float(2 / np.sqrt(maximum))

    def strain_matrices(self, *, mean_dilatation=False):
        """Four degree-two exact integration-point B matrices.

        Optional B-bar replaces only volumetric strain by its volume mean.
        It does not change deviatoric strain or introduce a tunable penalty.
        """
        g = self.gradients(_QUAD)
        B = np.zeros((4, 6, 3 * self.gradient_coefficients.shape[1]))
        for i in range(3):
            B[:, i, i::3] = g[:, :, i]
        for row, (i, j) in enumerate(((0, 1), (1, 2), (0, 2)), 3):
            B[:, row, i::3] = g[:, :, j]
            B[:, row, j::3] = g[:, :, i]
        if mean_dilatation:
            trace = B[:, :3].sum(axis=1)
            B[:, :3] += (trace.mean(axis=0) - trace)[:, None, :] / 3
        return B

    def stiffness(self, constitutive, *, mean_dilatation=False, stabilize=None):
        """Small-strain stiffness for constant symmetric positive-definite C.

        With mean dilatation, unresolved deviatoric strain energy stabilizes the
        projection by default. The plain projected formulation needs no correction.
        ``stabilize=False`` exposes the unstabilized research operator, which has
        extra zero-energy modes when combined with mean dilatation and must not be used as a production element.

        C maps engineering strain to physical stress in xx,yy,zz,xy,yz,xz
        order. Spatially varying, finite-strain and history-dependent material
        updates require a separate execution provider, not this linear kernel.
        """
        C = np.asarray(constitutive, dtype=float)
        if C.shape != (6, 6) or not np.isfinite(C).all():
            raise ValueError(
                "AFM-COMPOSITE-TET: expected finite 6 by 6 material tangent."
            )
        scale = np.max(np.abs(C))
        if not np.allclose(C, C.T, rtol=0, atol=1e-12 * scale):
            raise ValueError("AFM-COMPOSITE-TET: symmetric material tangent required.")
        try:
            np.linalg.cholesky(C)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "AFM-COMPOSITE-TET: positive-definite material tangent required."
            ) from exc
        B = self.strain_matrices(mean_dilatation=mean_dilatation)
        K = self.volume / 4 * np.einsum("qai,ab,qbj->ij", B, C, B, optimize=True)
        if stabilize is None:
            stabilize = mean_dilatation or self.gradient_coefficients.shape[1] == 11
        if stabilize:
            # Positive energy of the unresolved deviatoric strain; no fitted
            # hourglass coefficient. It vanishes for every affine displacement.
            R = self.fluctuation_strains
            K += np.einsum(
                "q,qai,ab,qbj->ij", self.fluctuation_weights, R, C, R, optimize=True
            )
        return K


def composite_tet10(coordinates):
    """Build affine straight-sided composite operators; reject curved geometry.

    Coordinates use Abaqus order: vertices, edges 01,12,20,03,13,23.
    The auxiliary centre is eliminated by interpolation, NOT static/dynamic
    condensation of an independent internal mode.
    """
    x = np.asarray(coordinates, dtype=float)
    if x.shape != (10, 3) or not np.isfinite(x).all():
        raise ValueError("AFM-COMPOSITE-TET: expected finite (10, 3) coordinates.")
    J = (x[1:4] - x[0]).T
    h = np.max(np.linalg.norm(x[:4, None] - x[None, :4], axis=2))
    determinant = np.linalg.det(J)
    if h == 0 or abs(determinant) <= 1e-12 * h**3:
        raise ValueError(
            "AFM-COMPOSITE-TET: degenerate or severely distorted geometry."
        )
    if not np.allclose(
        x[4:] - x[0], x[_EDGES].mean(axis=1) - x[0], rtol=0, atol=1e-10 * h
    ):
        raise NotImplementedError(
            "AFM-COMPOSITE-TET: affine mid-edge geometry required."
        )
    coefficients, mass, residual_gradients, weights = _reference_operators()
    inverse = np.linalg.inv(J)
    coefficients = coefficients @ inverse
    delta = residual_gradients @ inverse
    R = np.zeros((len(delta), 6, 30))
    for i in range(3):
        R[:, i, i::3] = delta[:, :, i]
    for row, (i, j) in enumerate(((0, 1), (1, 2), (0, 2)), 3):
        R[:, row, i::3] = delta[:, :, j]
        R[:, row, j::3] = delta[:, :, i]
    R[:, :3] -= R[:, :3].sum(axis=1)[:, None, :] / 3
    return CompositeTet10(
        _readonly(x),
        abs(determinant) / 6,
        _readonly(coefficients),
        _readonly(mass * abs(determinant)),
        _readonly(R),
        _readonly(weights * abs(determinant)),
    )


@lru_cache(maxsize=2)
def _reference_operators(independent_center=False):
    """Geometry-independent integration, shared immutably across affine cells.

    Projection commutes with the constant affine gradient transformation. The
    deviatoric split is deliberately performed AFTER physical transformation.
    """
    vertices = np.vstack((np.zeros(3), np.eye(3)))
    points = np.vstack((vertices, vertices[_EDGES].mean(axis=1), np.full(3, 0.25)))
    T = (
        np.eye(11)
        if independent_center
        else np.vstack((np.eye(10), np.r_[np.zeros(4), np.full(6, 1 / 6)]))
    )
    size = T.shape[1]
    gram = np.zeros((4, 4))
    moments = np.zeros((4, size, 3))
    mass = np.zeros((size, size))
    samples = []
    for ids in _SUBTETS:
        p = points[ids]
        v = abs(np.linalg.det((p[1:] - p[0]).T)) / 6
        basis = T[ids]
        grad = basis.T @ np.linalg.inv(np.column_stack((np.ones(4), p)))[1:].T
        q = _QUAD @ p
        lam = np.column_stack((1 - q.sum(axis=1), q))
        samples.append((v, lam, grad))
        gram += v / 4 * lam.T @ lam
        moments += v * np.einsum("a,ij->aij", lam.mean(axis=0), grad)
        mass += v / 20 * basis.T @ (np.ones((4, 4)) + np.eye(4)) @ basis
    coefficients = np.linalg.solve(gram, moments.reshape(4, -1)).reshape(4, size, 3)
    delta = np.concatenate(
        [
            grad[None] - np.einsum("qa,aij->qij", lam, coefficients)
            for _, lam, grad in samples
        ]
    )
    weights = np.repeat([v / 4 for v, _, _ in samples], 4)
    return tuple(_readonly(a) for a in (coefficients, mass, delta, weights))


class CompositeTet11(CompositeTet10):
    """Experimental 33-DOF composite kernel with independent interior node."""


def composite_tet11(coordinates):
    """Research kernel with THREE independent centre displacement DOFs.

    Ten external coordinates in Tet10 order; the eleventh point is the centroid.
    This is an independently derived enrichment, NOT the C3D10M formula. It is
    deliberately not accepted by the production tied-centre Step. Internal
    inertia is nonzero: static condensation does not preserve dynamic spectra.
    """
    base = composite_tet10(coordinates)  # shared geometry validation
    x = base.coordinates
    coefficients, mass, delta, weights = _reference_operators(True)
    J = (x[1:4] - x[0]).T
    inverse = np.linalg.inv(J)
    delta = delta @ inverse
    R = np.zeros((len(delta), 6, 33))
    for i in range(3):
        R[:, i, i::3] = delta[:, :, i]
    for row, (i, j) in enumerate(((0, 1), (1, 2), (0, 2)), 3):
        R[:, row, i::3] = delta[:, :, j]
        R[:, row, j::3] = delta[:, :, i]
    R[:, :3] -= R[:, :3].sum(axis=1)[:, None, :] / 3
    scale = 6 * base.volume
    return CompositeTet11(
        _readonly(np.vstack((x, x[:4].mean(axis=0)))),
        base.volume,
        _readonly(coefficients @ inverse),
        _readonly(scale * mass),
        _readonly(R),
        _readonly(scale * weights),
    )
