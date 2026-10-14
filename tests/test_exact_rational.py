# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from fractions import Fraction
import runpy
import sys
import numpy as np
import pytest
from agentfem import _exact_rational


def test_standard_library_fallback_without_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, 'gmpy2', None)
    namespace = runpy.run_path(_exact_rational.__file__)
    assert namespace['Rational'] is Fraction
    assert namespace['exact_geometry_arithmetic']()['library'] == 'fractions.Fraction'


def test_exact_binary64_conversion_and_arithmetic():
    Q = _exact_rational.Rational
    rng = np.random.default_rng(482)
    for value in np.ldexp(rng.uniform(-1,1,100), rng.integers(-1000,1000,100)):
        a, b = Fraction(float(value)), Q(float(value))
        for first, second in ((a,b), (a*a+Fraction(1,3), b*b+Q(1,3))):
            assert first.numerator == second.numerator
            assert first.denominator == second.denominator


def test_geometry_predicates_match_fraction_and_gmp(monkeypatch):
    gmpy2 = pytest.importorskip('gmpy2')
    from agentfem import _q1_ray_exact as ray, _q1_exact_order as order, _q1_swept_separation as swept
    x = np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,.25],[0.,1.,0.]])
    def evaluate(Q):
        for module in (ray, order, swept):
            monkeypatch.setattr(module, 'Q', Q)
        hit = ray.exact_q1_ray_crossings(x[None], [.3,.4,-1e-30], [0.,0.,1.])
        point = np.array([[.3125,.4375,.25*.3125*.4375 + 2.**-28]])
        gap = swept.separated_q1_paths(x,x,[[0,1,2,3]],point,point)
        equal = order.witness_within_distance(x, [.3,.4,.03], [.3,.4], 1e-16)
        return hit, gap, equal
    reference = evaluate(Fraction)
    actual = evaluate(gmpy2.mpq)
    for a,b in zip(reference, actual, strict=True):
        np.testing.assert_array_equal(a,b)
