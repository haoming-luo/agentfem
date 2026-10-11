# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Rare exact distance-order certificate for convex Q1 boundary minima.

Binary64 inputs are interpreted exactly as rationals. This is not a general
closest-point solver: unsuccessful convexity/KKT screens return False.
"""

from fractions import Fraction as Q


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _coefficients(vertices, point):
    v = [tuple(Q(float(x)) for x in row) for row in vertices]
    p = tuple(Q(float(x)) for x in point)
    b, c = _sub(v[1], v[0]), _sub(v[3], v[0])
    return _sub(v[0], p), b, c, _sub(_sub(v[2], v[3]), b)


def _value(coefficients, s, t):
    a, b, c, d = coefficients
    return tuple(ai + bi*s + ci*t + di*s*t
                 for ai, bi, ci, di in zip(a, b, c, d, strict=True))


def _strong_convexity_modulus(coefficients):
    """Exact comparison-matrix bound for Hessian(distance squared / 2)."""
    a, b, c, d = coefficients
    dd = _dot(d, d)

    def diagonal_lower(direction):
        r = max(Q(0), min(Q(1), -_dot(direction, d)/dd)) if dd else Q(0)
        vector = tuple(x+r*y for x, y in zip(direction, d, strict=True))
        return _dot(vector, vector)

    off = max(abs(_dot(b, c)+_dot(a, d)+2*x*_dot(b, d)
                  +2*y*_dot(c, d)+2*x*y*dd) for x in (0, 1) for y in (0, 1))
    first, second = diagonal_lower(b), diagonal_lower(c)
    gershgorin = min(first, second)-off
    # For skew/anisotropic traces Gershgorin can fail despite positive
    # definiteness. The comparison matrix [[first,-off],[-off,second]]
    # bounds the quadratic form in absolute components. If its determinant
    # is positive, det/trace is an exact lower bound on its least eigenvalue.
    determinant = first*second-off*off
    if first > 0 and second > 0 and determinant > 0:
        return max(gershgorin, determinant/(first+second))
    return max(gershgorin, _bernstein_hessian_modulus(coefficients))


def _bernstein_hessian_modulus(coefficients):
    """Exact SPD certificate retaining correlated Q1 Hessian entries.

    The determinant of the half squared-distance Hessian is biquadratic.
    Its tensor Bernstein coefficients bound it over the entire unit square.
    Positive coefficients and squared-norm diagonal entries certify SPD;
    det_lower / trace_upper then bounds its least eigenvalue. An inconclusive
    enclosure returns zero, never an approximate positivity decision.
    """
    a, b, c, d = coefficients
    bb, cc, dd = _dot(b, b), _dot(c, c), _dot(d, d)
    bd, cd = _dot(b, d), _dot(c, d)
    first, second = (bb, 2*bd, dd), (cc, 2*cd, dd)
    off = {(0, 0): _dot(b, c)+_dot(a, d), (1, 0): 2*bd,
           (0, 1): 2*cd, (1, 1): 2*dd}
    power = {(i, j): second[i]*first[j] for i in range(3) for j in range(3)}
    for (i, j), x in off.items():
        for (k, l), y in off.items():
            power[i+k, j+l] -= x*y
    # Degree-two power -> Bernstein transform, in each parameter.
    weights = ((Q(1), Q(0), Q(0)), (Q(1), Q(1, 2), Q(0)),
               (Q(1), Q(1), Q(1)))
    lower = min(sum((power[k, l]*weights[i][k]*weights[j][l]
                     for k in range(i+1) for l in range(j+1)), Q(0))
                for i in range(3) for j in range(3))
    if lower <= 0:
        return Q(0)
    trace = max(bb, bb+2*bd+dd)+max(cc, cc+2*cd+dd)
    return lower/trace


def witness_within_distance(vertices, point, uv, tolerance):
    """Prove a feasible Q1 point lies inside a closed distance ball exactly.

    This bounds distance only; it does not establish a unique normal or side.
    """
    s, t = (Q(float(x)) for x in uv)
    radius = Q(float(tolerance))
    if radius < 0 or not (0 <= s <= 1 and 0 <= t <= 1):
        return False
    delta = _value(_coefficients(vertices, point), s, t)
    return _dot(delta, delta) <= radius * radius


def convex_closest_parameters(vertices, point):
    """Rare recovery with an exact strong-convexity/KKT error certificate.

    Each Newton witness is rounded back to binary64 to bound rational growth.
    The returned feasible witness is within 2**-48 in parameter norm of the
    unique box minimizer. This follows from the exact Hessian lower bound and
    a normal-cone stationarity residual; an unproved case returns None.
    """
    coeff = _coefficients(vertices, point)
    a, b, c, d = coeff
    modulus = _strong_convexity_modulus(coeff)
    if modulus <= 0:
        return None
    u = v = Q(1)/2
    tolerance = Q(1)/2**48
    for _ in range(12):
        delta = _value(coeff, u, v)
        du = tuple(x+v*y for x, y in zip(b, d, strict=True))
        dv = tuple(x+u*y for x, y in zip(c, d, strict=True))
        gu, gv = _dot(delta, du), _dot(delta, dv)
        residual = [Q(0) if (z == 0 and g >= 0) or (z == 1 and g <= 0) else g
                    for z, g in ((u, gu), (v, gv))]
        if _dot(residual, residual) <= (tolerance*modulus)**2:
            return float(u), float(v)
        huu, hvv = _dot(du, du), _dot(dv, dv)
        huv = _dot(du, dv)+_dot(delta, d)
        determinant = huu*hvv-huv*huv
        if determinant <= 0:
            return None
        if u in (0, 1) and residual[0] == 0:
            un, vn = u, v-gv/hvv
        elif v in (0, 1) and residual[1] == 0:
            un, vn = u-gu/huu, v
        else:
            un, vn = u-(hvv*gu-huv*gv)/determinant, v-(huu*gv-huv*gu)/determinant
        u, v = (Q(float(max(Q(0), min(Q(1), z)))) for z in (un, vn))
    return None


def _convex_minimum_interval(vertices, point, uv):
    """Exact squared-distance enclosure from strong convexity and Newton witnesses."""
    coeff = _coefficients(vertices, point)
    a, b, c, d = coeff
    modulus = _strong_convexity_modulus(coeff)
    if modulus <= 0:
        return None
    u, v = (Q(float(x)) for x in uv)
    if not (0 <= u <= 1 and 0 <= v <= 1):
        return None
    lower, upper = Q(0), None
    for iteration in range(3):
        delta = _value(coeff, u, v)
        du = tuple(x+v*y for x, y in zip(b, d, strict=True))
        dv = tuple(x+u*y for x, y in zip(c, d, strict=True))
        gu, gv = _dot(delta, du), _dot(delta, dv)
        squared = _dot(delta, delta)
        upper = squared if upper is None else min(upper, squared)
        # For f=distance²/2 and Hessian >= modulus I,
        # 2 f* >= 2 f(x) - |gradient f(x)|²/modulus.
        lower = max(lower, squared-(gu*gu+gv*gv)/modulus)
        if iteration == 2:
            break
        huu, hvv = _dot(du, du), _dot(dv, dv)
        huv = _dot(du, dv)+_dot(delta, d)
        determinant = huu*hvv-huv*huv
        if determinant <= 0:
            break
        u = max(Q(0), min(Q(1), u-(hvv*gu-huv*gv)/determinant))
        v = max(Q(0), min(Q(1), v-(huu*gv-huv*gu)/determinant))
    return lower, upper


def convex_minimum_precedes(vertices, point, uv, other_vertices, other_uv):
    """Strict exact ordering only; genuine ties remain unresolved."""
    first = _convex_minimum_interval(vertices, point, uv)
    if first is None:
        return False
    second = _convex_minimum_interval(other_vertices, point, other_uv)
    return second is not None and first[1] < second[0]


def witness_beats_boundary_minimum(witness_vertices, point, uv, other_vertices):
    """Certify a feasible witness is closer than every point of another patch.

For f=|x(s,t)-p|²/2, bound its Hessian on the entire unit square by
Gershgorin. Then find an exact edge minimizer satisfying the full box KKT
conditions. Convexity makes this a global minimum. Compare squared distances
without square roots or floating-point subtraction. No uncertain result passes.
    """
    s, t = (Q(float(x)) for x in uv)
    if not (0 <= s <= 1 and 0 <= t <= 1):
        return False
    witness = _value(_coefficients(witness_vertices, point), s, t)
    witness_squared = _dot(witness, witness)
    coeff = _coefficients(other_vertices, point)
    a, b, c, d = coeff
    if _strong_convexity_modulus(coeff) <= 0:
        return False
    corners = ((Q(0), Q(0)), (Q(1), Q(0)), (Q(1), Q(1)), (Q(0), Q(1)))
    for start, end in zip(corners, corners[1:] + corners[:1], strict=True):
        x0, x1 = _value(coeff, *start), _value(coeff, *end)
        direction = _sub(x1, x0)
        norm = _dot(direction, direction)
        if not norm:
            continue
        r = max(Q(0), min(Q(1), -_dot(x0, direction)/norm))
        u, v = (start[k] + r*(end[k]-start[k]) for k in (0, 1))
        delta = _value(coeff, u, v)
        gradient = (
            _dot(delta, tuple(bi + di*v for bi, di in zip(b, d, strict=True))),
            _dot(delta, tuple(ci + di*u for ci, di in zip(c, d, strict=True))),
        )
        if all((g >= 0 if z == 0 else g <= 0 if z == 1 else g == 0)
               for z, g in zip((u, v), gradient, strict=True)):
            return witness_squared < _dot(delta, delta)
    return False
