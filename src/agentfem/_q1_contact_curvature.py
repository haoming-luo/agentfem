# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Local smooth closest-point curvature; no whole-path stability admission.

For E(z,xi)=k*w/2*|q-s(xi)|^2, stationary elimination gives
E_zz - E_zxi E_xixi^-1 E_xiz. The subtracted matrix is PSD at a
strict projection minimum. Keeping it distinguishes geometric softening from
the fixed-coordinate positive penalty term. Only interior Q1 minima qualify.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Q1ContactCurvature:
    tangent: np.ndarray
    fixed_coordinate_upper: np.ndarray
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    projection_minimum_eigenvalue: float


@dataclass(frozen=True)
class Q1ContactRowEnclosure:
    """-diag(negative_rows) <= K_contact <= diag(positive_rows), at one state."""
    positive_rows: np.ndarray
    negative_rows: np.ndarray
    active_points: int

    def summary(self):
        return {"scope": "instantaneous_smooth_interior_q1_contact",
                "active_points": self.active_points,
                "geometric_negative_envelope_present": bool(np.any(self.negative_rows)),
                "whole_path_stability": False,
                "implicit_solver_admitted": False}


def contact_row_enclosure(operator, secondary_displacement, primary_displacement):
    """Assemble unscaled signed enclosures; retain projection and activation limits.

    Node order is all secondary nodes followed by all primary nodes. This is
    not a mass-scaled frequency or a choice of time increment. Strictly
    separated points contribute zero at this state; their later activation is
    not covered. Negative rows bound the magnitude from above and do not prove
    that the assembled constrained system actually has negative eigenvalues.
    """
    response = operator.evaluate(secondary_displacement, primary_displacement)
    count = len(operator.secondary.vertices)
    shape = (count + len(operator.primary.vertices), 3)
    upper, lower = np.zeros(shape), np.zeros(shape)
    active = 0
    if response.projection is not None:
        gaps = response.projection.signed_gaps
        if np.any(np.abs(gaps) <= 1024*np.finfo(float).eps*operator.search_radius):
            raise ValueError("Contact activation has no unique classical curvature.")
        selected = (np.arange(len(gaps)) if response.projection_indices is None
                    else response.projection_indices)
        primary = operator.primary.vertices + primary_displacement
        for point in np.flatnonzero(gaps < 0):
            index = selected[point]
            nodes = response.primary_nodes[point]
            weights = response.primary_weights[point]
            local = interior_distance_curvature(response.projection.query_points[point],
                primary[nodes], [weights[1]+weights[2], weights[2]+weights[3]],
                stiffness=float(operator.law.penalty)*operator.weights[index])
            ids = np.r_[operator.nodes[index], count+nodes]
            np.add.at(upper, ids, np.abs(local.fixed_coordinate_upper).sum(axis=1).reshape(5,3))
            schur = local.fixed_coordinate_upper-local.tangent
            np.add.at(lower, ids, np.abs(schur).sum(axis=1).reshape(5,3))
            active += 1
    for array in (upper, lower):
        if not np.isfinite(array).all():
            raise ValueError("Contact row enclosure overflowed.")
        array.setflags(write=False)
    return Q1ContactRowEnclosure(upper, lower, active)


def interior_distance_curvature(query, vertices, local_coordinates, *, stiffness):
    """15x15 distance-potential Hessian in [query, four surface nodes] order.

    stiffness includes the reference integration weight. The caller owns the
    active-contact decision, the selected facet and its uniqueness evidence.
    This routine checks stationarity and local strict minimality, not global
    uniqueness, contact eligibility or an admissible trajectory. It must not
    be used on a seam, rim, contact activation event or ambiguous projection.
    """
    raw = [np.asarray(a) for a in (query, vertices, local_coordinates, stiffness)]
    if any(a.dtype.kind not in "fiu" or not np.isfinite(a).all() for a in raw):
        raise ValueError("Contact curvature requires finite real data.")
    q, x, uv, coefficient = [a.astype(float) for a in raw]
    if q.shape != (3,) or x.shape != (4, 3) or uv.shape != (2,) or coefficient.ndim:
        raise ValueError("Expected query (3), Q1 vertices (4,3), coordinates (2).")
    if coefficient <= 0 or np.any(uv <= 1e-8) or np.any(uv >= 1 - 1e-8):
        raise ValueError("Positive stiffness and strict interior projection required.")
    u, v = uv
    shape = np.array([(1-u)*(1-v), u*(1-v), u*v, (1-u)*v])
    derivative = np.array([[-(1-v), 1-v, v, -v], [-(1-u), -u, u, 1-u]])
    # Translate first to avoid losing small local scales in large coordinates.
    local = x - x[0]
    r = q - x[0] - shape @ local
    tangents = derivative @ local
    mixed = np.array([1., -1., 1., -1.]) @ local
    scale = max(float(np.max(np.linalg.norm(local, axis=1))), np.finfo(float).tiny)
    if np.max(np.abs(tangents @ r)) > 1e-9 * scale * max(scale, np.linalg.norm(r)):
        raise ValueError("Projection is not stationary on the Q1 surface.")
    h = tangents @ tangents.T
    h[0, 1] -= r @ mixed
    h[1, 0] -= r @ mixed
    eigenvalues = np.linalg.eigvalsh(h)
    if eigenvalues[0] <= 256*np.finfo(float).eps*max(scale**2, eigenvalues[-1]):
        raise ValueError("Projection is degenerate or not a strict distance minimum.")
    b = np.kron(np.r_[1., -shape][None], np.eye(3))
    cross = np.empty((15, 2))
    for a in range(2):
        db = np.kron(np.r_[0., -derivative[a]][None], np.eye(3))
        cross[:, a] = db.T @ r - b.T @ tangents[a]
    upper = float(coefficient) * (b.T @ b)
    tangent = upper - float(coefficient) * (cross @ np.linalg.solve(h, cross.T))
    tangent = (tangent + tangent.T) / 2
    if not np.isfinite(tangent).all():
        raise ValueError("Contact curvature overflowed.")
    spectrum = np.linalg.eigvalsh(tangent)
    tangent.setflags(write=False)
    upper.setflags(write=False)
    return Q1ContactCurvature(tangent, upper, float(spectrum[0]),
                              float(spectrum[-1]), float(eigenvalues[0]))
