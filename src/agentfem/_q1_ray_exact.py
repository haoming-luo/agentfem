# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded rational-interval fallback for ambiguous bilinear ray crossings."""

from fractions import Fraction as Q
from math import isqrt

import numpy as np

from ._q1_ray import AmbiguousRay


class _Interval:
    def __init__(self, lo, hi=None):
        self.lo, self.hi = Q(lo), Q(lo if hi is None else hi)

    def __add__(self, other):
        other = _coerce(other)
        return _Interval(self.lo + other.lo, self.hi + other.hi)

    __radd__ = __add__

    def __neg__(self):
        return _Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -_coerce(other)

    def __rsub__(self, other):
        return _coerce(other) + -self

    def __mul__(self, other):
        other = _coerce(other)
        values = [a*b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return _Interval(min(values), max(values))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _coerce(other)
        if other.lo <= 0 <= other.hi:
            raise AmbiguousRay("Exact ray denominator includes zero.")
        return self * _Interval(1/other.hi, 1/other.lo)

    def midpoint(self):
        return float((self.lo + self.hi)/2)


def _coerce(value):
    return value if isinstance(value, _Interval) else _Interval(value)


def _sqrt_interval(value, bits):
    # floor(sqrt(n/d)*2**bits) = isqrt(floor(n*2**(2*bits)/d)).
    scale = 1 << bits
    lower = isqrt((value.numerator << (2*bits)) // value.denominator)
    lo = Q(lower, scale)
    return _Interval(lo, lo if lo*lo == value else Q(lower+1, scale))


def _facet_hits(vertices, origin, ray, bits):
    x = [[Q(float(v)) for v in row] for row in vertices]
    p, r = [Q(float(v)) for v in origin], [Q(float(v)) for v in ray]
    a = [x[0][i]-p[i] for i in range(3)]
    b = [x[1][i]-x[0][i] for i in range(3)]
    c = [x[3][i]-x[0][i] for i in range(3)]
    d = [x[2][i]-x[1][i]-x[3][i]+x[0][i] for i in range(3)]
    k = max(range(3), key=lambda i: abs(r[i]))
    axes = [i for i in range(3) if i != k]
    A, B, C, D = [[r[k]*v[i]-r[i]*v[k] for i in axes] for v in (a, b, c, d)]
    det = lambda v, w: v[0]*w[1]-v[1]*w[0]
    constant, linear, quadratic = det(A, B), det(A, D)+det(C, B), det(C, D)
    if not quadratic:
        if not linear:
            if constant:
                return []
            raise AmbiguousRay("Exact ray elimination is degenerate.")
        roots = [_Interval(-constant/linear)]
    else:
        discriminant = linear*linear-4*quadratic*constant
        if discriminant < 0:
            return []
        if not discriminant:
            raise AmbiguousRay("Exact ray has a repeated root.")
        root = _sqrt_interval(discriminant, bits)
        roots = [(-linear + sign*root)/(2*quadratic) for sign in (-1, 1)]
    hits = []
    for v in roots:
        if v.hi < 0 or v.lo > 1:
            continue
        denominator = [B[i]+D[i]*v for i in range(2)]
        j = max(range(2), key=lambda i: abs(denominator[i].midpoint()))
        u = -(A[j]+C[j]*v)/denominator[j]
        if u.hi < 0 or u.lo > 1:
            continue
        offset = [a[i]+b[i]*u+c[i]*v+d[i]*u*v for i in range(3)]
        t = offset[k]/r[k]
        if t.hi < 0:
            continue
        if not (u.lo > 0 and u.hi < 1 and v.lo > 0 and v.hi < 1 and t.lo > 0):
            raise AmbiguousRay("Exact ray cannot separate a seam or origin.")
        du, dv = [b[i]+d[i]*v for i in range(3)], [c[i]+d[i]*u for i in range(3)]
        crossing = sum((du[(i+1)%3]*dv[(i+2)%3]-du[(i+2)%3]*dv[(i+1)%3])*r[i]
                       for i in range(3))
        if crossing.lo > 0:
            sign = 1
        elif crossing.hi < 0:
            sign = -1
        else:
            raise AmbiguousRay("Exact ray cannot separate tangency.")
        hits.append((u.midpoint(), v.midpoint(), t.midpoint(), sign))
    return hits


def exact_q1_ray_crossings(vertices, origin, direction):
    """Certified hit signs for validated inputs; unresolved rays still reject.

Called only after the normal kernel validates inputs and reports ambiguity.
Two finite precision levels bound work. Output coordinates are rounded views
of isolating intervals, never used as an exact contact projection.
    """
    hits = []
    for facet, geometry in enumerate(vertices):
        for bits in (96, 192):
            try:
                local = _facet_hits(geometry, origin, direction, bits)
                break
            except AmbiguousRay:
                if bits == 192:
                    raise
        hits.extend((facet, *row) for row in local)
    return np.asarray(hits, dtype=float).reshape(-1, 5)
