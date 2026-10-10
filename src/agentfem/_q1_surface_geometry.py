# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private Q1 geometry kernels; no interface integration admission.

Basix owns interpolation. This adapter batches geometric inversion for boundary
queries, preserving all four parent nodes. On-surface inversion and bounded
interior normal projection are distinct; global closest search and curved
intersection integration remain separate.
"""

from functools import lru_cache

import basix
import numpy as np


def q1_closest_point(vertices, point):
    """Exact Q1 closest point, with a convex affine fast path.

    A parallelogram needs only an interior Gram solve and four segment
    minima. General bilinear geometry still uses the polynomial oracle below;
    a user geometry tolerance cannot flatten genuine warpage.
    """
    physical = _real(vertices, (4, 3), "vertices")
    query = _real(point, (3,), "point")
    scale = np.linalg.norm(np.ptp(physical, axis=0))
    if scale <= 0:
        raise ValueError("Degenerate Q1 facet.")
    x = (physical - physical[0]) / scale
    d = x[2] - x[1] - x[3]
    q = (query - physical[0]) / scale
    axes = x[[1, 3]]
    normal = np.cross(*axes)
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        raise ValueError("Degenerate Q1 corner.")
    normal /= norm
    if np.linalg.norm(d) > 8 * np.finfo(float).eps:
        if abs(x[2] @ normal) > 8 * np.finfo(float).eps:
            return _q1_closest_point_polynomial(physical, query)
        # Convex planar nonaffine quads still have a planar distance problem;
        # only the on-surface inverse needs the original bilinear mapping.
        edges = np.roll(x, -1, axis=0) - x
        turns = np.cross(edges, np.roll(edges, -1, axis=0)) @ normal
        if np.any(turns <= 1e-12):
            return _q1_closest_point_polynomial(physical, query)
        projected = q - (q @ normal) * normal
        if np.all(np.cross(edges, projected - x) @ normal >= 0):
            uv, _ = q1_surface_pull_back(
                x[None], projected[None], relative_tolerance=16 * np.finfo(float).eps
            )
            uv = np.clip(uv, 0.0, 1.0)
            basis = _element().tabulate(0, uv)[0, :, :, 0][:, [0, 1, 3, 2]]
            position = physical[0] + scale * (basis[0] @ x)
            return (
                uv[0],
                basis[0],
                position,
                normal,
                float(np.linalg.norm(query - position)),
            )
        # Edge minima are exact even for a nonaffine planar quad.
        uv = np.array([-1.0, -1.0])
    else:
        uv = np.linalg.solve(axes @ axes.T, axes @ q)
    if not np.all((uv >= 0) & (uv <= 1)):
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        edges = np.roll(x, -1, axis=0) - x
        t = np.clip(
            np.einsum("ij,ij->i", q - x, edges) / np.einsum("ij,ij->i", edges, edges),
            0,
            1,
        )
        positions = x + t[:, None] * edges
        best = int(np.argmin(np.linalg.norm(positions - q, axis=1)))
        uv = (1 - t[best]) * corners[best] + t[best] * corners[(best + 1) % 4]
    basis = _element().tabulate(0, uv[None])[0, 0, [0, 1, 3, 2], 0]
    point = physical[0] + scale * (basis @ x)
    return uv, basis, point, normal, float(np.linalg.norm(query - point))


def _q1_closest_point_polynomial(vertices, point):
    """Reference closest point including all edges/corners of one Q1 facet.

    Eliminate u from the interior stationarity equations, solve the resulting
    polynomial in v, and compare admissible stationary points with exact
    straight-edge minima. This bounded scalar oracle complements the batched
    local projector; it is not the production batch search implementation.
    Returns (uv, Q1 basis, position, normal, distance). Tied distinct positions
    are rejected; no arbitrary branch is used to define a contact normal.
    """
    from numpy.polynomial import Polynomial as P

    physical = _real(vertices, (4, 3), "vertices")
    query = _real(point, (3,), "point")
    q1_surface_pull_back(physical[None], physical.mean(axis=0)[None])
    scale = np.linalg.norm(np.ptp(physical, axis=0))
    x = (physical - physical[0]) / scale
    q = (query - physical[0]) / scale
    a, b, c, d = -q, x[1], x[3], x[2] - x[1] - x[3]
    # x-query = (a+c*v) + (b+d*v)*u; optimal u=-A/B.
    e = [P([a[i], c[i]]) for i in range(3)]
    f = [P([b[i], d[i]]) for i in range(3)]
    A = sum((e[i] * f[i] for i in range(3)), P([0]))
    B = sum((f[i] * f[i] for i in range(3)), P([0]))
    equation = sum(
        ((e[i] * B - f[i] * A) * (c[i] * B - d[i] * A) for i in range(3)), P([0])
    )
    # Geometry can reduce this polynomial's degree (e.g. a planar trapezoid).
    # Roundoff-only leading coefficients produce enormous companion roots and
    # can erase a finite root. Trim only relative coefficient roundoff, then
    # check/polish candidates against the uneliminated physical equations.
    equation = equation.trim(
        tol=64 * np.finfo(float).eps * np.max(np.abs(equation.coef))
    )
    candidates = []
    for root in equation.roots():
        if abs(root.imag) > 1e-8 or not 0 < root.real < 1:
            continue
        v = float(root.real)
        u = float(-A(v) / B(v))
        if 0 < u < 1:
            # Near degree reduction, polynomial roots need polishing in the
            # original stationarity equations rather than the eliminated one.
            for _ in range(12):
                residual = a + b * u + c * v + d * u * v
                ju, jv = b + d * v, c + d * u
                gradient = np.array([ju @ residual, jv @ residual])
                if np.linalg.norm(gradient) <= 1e-13:
                    break
                cross = ju @ jv + d @ residual
                hessian = np.array([[ju @ ju, cross], [cross, jv @ jv]])
                try:
                    step = np.linalg.solve(hessian, gradient)
                except np.linalg.LinAlgError as exc:
                    raise ValueError("Unresolved Q1 stationary root.") from exc
                u, v = np.array([u, v]) - step
            residual = a + b * u + c * v + d * u * v
            gradient = np.array([(b + d * v) @ residual, (c + d * u) @ residual])
            if np.linalg.norm(gradient) > 1e-9:
                raise ValueError("Unresolved Q1 stationary root.")
            if 0 < u < 1 and 0 < v < 1:
                candidates.append([u, v])
    corners = np.array([[0.0, 0], [1, 0], [1, 1], [0, 1]])
    for i in range(4):
        j = (i + 1) % 4
        edge = x[j] - x[i]
        t = np.clip((q - x[i]) @ edge / (edge @ edge), 0, 1)
        candidates.append((1 - t) * corners[i] + t * corners[j])
    uv = np.asarray(candidates)
    # Boundary candidates must satisfy the one-sided minimum conditions.
    # Otherwise an edge just beside an interior minimum can look numerically
    # tied in distance while not being a constrained stationary point at all.
    residual = a + uv[:, :1] * b + uv[:, 1:] * c + uv.prod(axis=1)[:, None] * d
    gradient = np.column_stack(
        (
            np.einsum("ni,ni->n", b + uv[:, 1:] * d, residual),
            np.einsum("ni,ni->n", c + uv[:, :1] * d, residual),
        )
    )
    admissible = np.where(
        uv == 0,
        gradient >= -1e-12,
        np.where(uv == 1, gradient <= 1e-12, np.abs(gradient) <= 1e-9),
    )
    uv = uv[np.all(admissible, axis=1)]
    if not len(uv):
        raise ValueError("No resolved constrained Q1 minimum.")
    basis = _element().tabulate(1, uv)[..., 0][:, :, [0, 1, 3, 2]]
    positions = basis[0] @ x
    distances = np.linalg.norm(positions - q, axis=1)
    best = int(np.argmin(distances))
    tied = np.abs(distances - distances[best]) <= 1e-11
    if np.any(np.linalg.norm(positions[tied] - positions[best], axis=1) > 1e-8):
        raise ValueError("Ambiguous Q1 closest points.")
    tangents = basis[1:3, best] @ x
    normal = np.cross(tangents[0], tangents[1])
    normal /= np.linalg.norm(normal)
    return (
        uv[best],
        basis[0, best],
        physical[0] + scale * positions[best],
        normal,
        float(scale * distances[best]),
    )


def q1_interior_normal_projection(vertices, points, *, max_iterations=40):
    """Batch local normal projections on nominated Q1 facets (private).

    Returns local coordinates, original Q1 weights, physical projected points,
    oriented normals and signed normal gaps. Newton solves J.T*(x-query)=0
    with the exact distance Hessian, including bilinear curvature. Only strict
    interior positive-Hessian solutions are admitted. No edge/corner search,
    global nearest-point uniqueness, bond eligibility or force is inferred.
    Candidate discovery and ambiguity resolution belong to the search layer.
    """
    raw = np.asarray(vertices)
    if raw.ndim != 3 or raw.shape[1:] != (4, 3):
        raise ValueError("vertices must have shape (queries, 4, 3).")
    x = _real(vertices, raw.shape, "vertices")
    q = _real(points, (len(x), 3), "points")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer.")
    q1_surface_pull_back(x, x.mean(axis=1))
    if not len(x):
        return (
            np.empty((0, 2)),
            np.empty((0, 4)),
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty(0),
        )
    origin = x[:, 0].copy()
    scale = np.linalg.norm(np.ptp(x, axis=1), axis=1)
    x = (x - origin[:, None]) / scale[:, None, None]
    q = (q - origin) / scale[:, None]
    twist = x[:, 0] - x[:, 1] + x[:, 2] - x[:, 3]

    def evaluate(uv):
        basis = _element().tabulate(1, uv)[..., 0][:, :, [0, 1, 3, 2]]
        position = np.einsum("nk,nki->ni", basis[0], x)
        residual = position - q
        jacobian = np.einsum("ank,nki->nia", basis[1:3], x)
        gradient = np.einsum("nia,ni->na", jacobian, residual)
        hessian = np.einsum("nia,nib->nab", jacobian, jacobian)
        curvature = np.einsum("ni,ni->n", residual, twist)
        hessian[:, 0, 1] += curvature
        hessian[:, 1, 0] += curvature
        return basis[0], position, residual, jacobian, gradient, hessian

    uv = np.full((len(x), 2), 0.5)
    for _ in range(max_iterations):
        _, _, residual, _, gradient, hessian = evaluate(uv)
        if np.any(np.linalg.eigvalsh(hessian)[:, 0] <= 1e-12):
            raise ValueError("Q1 projection has a nonpositive distance Hessian.")
        if np.all(np.linalg.norm(gradient, axis=1) < 1e-12):
            break
        step = np.linalg.solve(hessian, gradient[..., None])[..., 0]
        step[np.linalg.norm(gradient, axis=1) < 1e-12] = 0
        # Independent per-query backtracking, vectorized across the batch.
        alpha = np.ones(len(x))
        old_value = np.sum(residual * residual, axis=1)
        accepted = np.zeros(len(x), dtype=bool)
        for _ in range(25):
            trial = uv - alpha[:, None] * step
            trial_basis = _element().tabulate(0, trial)[0, :, :, 0][:, [0, 1, 3, 2]]
            trial_residual = np.einsum("nk,nki->ni", trial_basis, x) - q
            roundoff = 32 * np.finfo(float).eps * np.maximum(1, old_value)
            accepted = np.sum(trial_residual**2, axis=1) <= old_value + roundoff
            accepted &= np.all((trial >= 0) & (trial <= 1), axis=1)
            if accepted.all():
                break
            alpha[~accepted] *= 0.5
        if not accepted.all():
            raise ValueError("Q1 projection cannot find an interior descent step.")
        uv = trial
    basis, position, residual, jacobian, gradient, hessian = evaluate(uv)
    if np.any(np.linalg.norm(gradient, axis=1) >= 1e-12):
        raise ValueError("Q1 normal projection did not converge.")
    if np.any((uv <= 1e-10) | (uv >= 1 - 1e-10)):
        raise ValueError("Q1 normal projection requires a strict interior point.")
    if np.any(np.linalg.eigvalsh(hessian)[:, 0] <= 1e-12):
        raise ValueError("Q1 projection has a nonpositive distance Hessian.")
    normal = np.cross(jacobian[..., 0], jacobian[..., 1])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    gap = -np.einsum("ni,ni->n", residual, normal) * scale
    return uv, basis, origin + position * scale[:, None], normal, gap


@lru_cache(maxsize=1)
def _element():
    return basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.quadrilateral,
        1,
        basix.LagrangeVariant.equispaced,
    )


def _real(value, shape, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real.")
    value = np.asarray(raw, dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return value


def planar_q1_pull_back(
    vertices, points, *, relative_tolerance=1e-11, max_iterations=30
):
    """Strict planar adapter retained for the existing planar overlap route."""
    return _pull_back(
        vertices,
        points,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
        planar_only=True,
    )


def q1_surface_pull_back(
    vertices, points, *, relative_tolerance=1e-11, max_iterations=30
):
    """Invert a bilinear surface with convex projection onto its corner plane.

    Keeps real warpage. This is an on-surface inverse, not a closest-point
    projection. Convex projected corners ensure a one-to-one planar Q1 map;
    the bilinear surface is a graph over that projection. Folded projections
    are rejected rather than resolved through an arbitrary inverse branch.
    """
    return _pull_back(
        vertices,
        points,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
        planar_only=False,
    )


def _pull_back(vertices, points, *, relative_tolerance, max_iterations, planar_only):
    """Return local coordinates and cyclic Q1 weights for one query per facet.

    Input shapes are (queries, 4, 3) and (queries, 3). Tolerance is relative
    to facet diameter, never a geometric coincidence allowance. Newton steps
    minimize the in-plane map residual; both reference containment and physical
    reconstruction are checked. No nearest-surface projection is performed.
    """
    raw = np.asarray(vertices)
    if raw.ndim != 3 or raw.shape[1:] != (4, 3):
        raise ValueError("vertices must have shape (queries, 4, 3).")
    count = len(raw)
    x = _real(vertices, (count, 4, 3), "vertices")
    query = _real(points, (count, 3), "points")
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance <= 1e-8:
        raise ValueError("relative_tolerance must lie in (0, 1e-8].")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer.")
    if not count:
        return np.empty((0, 2)), np.empty((0, 4))
    # Normalize to avoid units-dependent singularity thresholds and cancellation
    # in basis sums at large coordinate origins.
    diameter = np.linalg.norm(np.ptp(x, axis=1), axis=1)
    if np.any(diameter <= 0):
        raise ValueError("Degenerate Q1 facet.")
    q = (query - x[:, 0]) / diameter[:, None]
    x = (x - x[:, :1]) / diameter[:, None, None]
    edge = np.roll(x, -1, axis=1) - x
    normal = np.cross(edge[:, 0], -edge[:, 3])
    length = np.linalg.norm(normal, axis=1)
    if np.any(length <= 1e-12):
        raise ValueError("Degenerate Q1 corner.")
    normal /= length[:, None]
    if planar_only and np.any(
        np.abs(np.einsum("nki,ni->nk", x, normal)) > relative_tolerance
    ):
        raise ValueError("Warped Q1 facets are not admitted by the planar map.")
    turns = np.einsum("nki,ni->nk", np.cross(edge, np.roll(edge, -1, axis=1)), normal)
    if np.any(turns <= 1e-12):
        raise ValueError("Q1 facets must be strictly convex with cyclic nodes.")
    local = np.full((count, 2), 0.5)
    for _ in range(max_iterations):
        basis = _element().tabulate(1, local)[..., 0][:, :, [0, 1, 3, 2]]
        residual = np.einsum("nk,nki->ni", basis[0], x) - q
        if np.all(np.linalg.norm(residual, axis=1) <= relative_tolerance):
            break
        jacobian = np.einsum("ank,nki->nia", basis[1:3], x)
        gram = np.einsum("nia,nib->nab", jacobian, jacobian)
        rhs = np.einsum("nia,ni->na", jacobian, residual)
        try:
            step = np.linalg.solve(gram, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError as exc:
            raise ValueError("Singular Q1 inverse map.") from exc
        local -= step
        if not np.all(np.isfinite(local)):
            raise ValueError("Nonfinite Q1 inverse map.")
    weights = _element().tabulate(0, local)[0, :, :, 0][:, [0, 1, 3, 2]]
    reconstruction = np.einsum("nk,nki->ni", weights, x)
    if np.any(np.linalg.norm(reconstruction - q, axis=1) > relative_tolerance):
        raise ValueError("Q1 pull-back did not converge to a point on the facet.")
    if np.any(local < -relative_tolerance) or np.any(local > 1 + relative_tolerance):
        raise ValueError("Q1 query lies outside the reference facet.")
    return local, weights


def q1_surface_quadrature(vertices, *, order=3):
    """True bilinear surface measure and oriented normals, one facet batch.

    Returns physical points (facets,q,3), cyclic basis (q,4), physical weights
    (facets,q), normals (facets,q,3). Positive Gauss weights integrate the actual
    surface Jacobian, not its triangular approximation. This supplies no
    cross-surface coverage guarantee; quadrature refinement remains required.
    """
    if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 16:
        raise ValueError("Q1 quadrature order must be an integer from 1 to 16.")
    raw = np.asarray(vertices)
    if raw.ndim != 3 or raw.shape[1:] != (4, 3):
        raise ValueError("vertices must have shape (facets, 4, 3).")
    x = _real(vertices, raw.shape, "vertices")
    # The inverse-map admission checks projected convexity and nondegeneracy.
    q1_surface_pull_back(x, x.mean(axis=1))
    abscissa, weight = np.polynomial.legendre.leggauss(order)
    abscissa = (abscissa + 1) / 2
    weight = weight / 2
    local = np.stack(np.meshgrid(abscissa, abscissa, indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    reference_weight = np.outer(weight, weight).ravel()
    basis = _element().tabulate(1, local)[..., 0][:, :, [0, 1, 3, 2]]
    # Subtract the origin before derivative contractions to limit cancellation.
    shifted = x - x[:, :1]
    tangents = np.einsum("aqk,nki->nqia", basis[1:3], shifted)
    cross = np.cross(tangents[..., 0], tangents[..., 1])
    jacobian = np.linalg.norm(cross, axis=-1)
    if np.any(jacobian <= 0) or not np.isfinite(jacobian).all():
        raise ValueError("Invalid Q1 surface Jacobian.")
    points = np.einsum("qk,nki->nqi", basis[0], shifted) + x[:, :1]
    weights = jacobian * reference_weight
    normals = cross / jacobian[..., None]
    return points, basis[0], weights, normals


def q1_triangulation_distance_bound(vertices):
    """Conservative per-facet Hausdorff bound to the cyclic diagonal split.

    Admission requires the same strictly convex projected graph as the inverse
    map. Both bilinear and triangular surfaces cover that projected polygon;
    their heights lie within the corner-height range. Pairing points at equal
    projected coordinates bounds their distance by that range, in physical
    length units. This is a geometry-search diagnostic, not permission to use
    triangular interpolation for Q1 fields or a surface-coverage certificate.
    """
    raw = np.asarray(vertices)
    if raw.ndim != 3 or raw.shape[1:] != (4, 3):
        raise ValueError("vertices must have shape (facets, 4, 3).")
    x = _real(vertices, raw.shape, "vertices")
    q1_surface_pull_back(x, x.mean(axis=1))
    shifted = x - x[:, :1]
    # Normalize first, as in the inverse map, to avoid unit-dependent products.
    scale = np.linalg.norm(np.ptp(x, axis=1), axis=1)
    normalized = shifted / scale[:, None, None]
    normal = np.cross(normalized[:, 1], normalized[:, 3])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    heights = np.einsum("nki,ni->nk", normalized, normal)
    return np.ptp(heights, axis=1) * scale
