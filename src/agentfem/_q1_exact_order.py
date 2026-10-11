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
    dd = _dot(d, d)

    def diagonal_lower(direction):
        r = max(Q(0), min(Q(1), -_dot(direction, d)/dd)) if dd else Q(0)
        vector = tuple(x + r*y for x, y in zip(direction, d, strict=True))
        return _dot(vector, vector)

    off = max(abs(_dot(b, c) + _dot(a, d) + 2*x*_dot(b, d)
                  + 2*y*_dot(c, d) + 2*x*y*dd)
              for x in (0, 1) for y in (0, 1))
    if min(diagonal_lower(b), diagonal_lower(c)) <= off:
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
