# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent endpoint quadrature and its deliberately discrete work bound."""

import numpy as np
import pytest

from agentfem.boundary_models.contact_work import _fixed_normal_work_increment


def _work(g0, g1, p0, p1, weights=(1.,)):
    return _fixed_normal_work_increment(weights=weights, old_gap=g0, new_gap=g1,
                                        old_pressure=p0, new_pressure=p1)


def test_exact_endpoint_complementarity_does_not_remove_closure_work():
    closed = _work([.02], [0.], [0.], [10.])
    assert closed.work_on_structure == pytest.approx(-.1)
    assert closed.open_gap_work == closed.work_on_structure
    assert closed.absolute_work_bound == pytest.approx(.1)
    opened = _work([0.], [.02], [10.], [0.])
    assert opened.work_on_structure == pytest.approx(.1)
    assert opened.open_gap_work == opened.work_on_structure


def test_closed_contact_and_penetration_have_distinct_work():
    assert _work([0.], [0.], [10.], [20.]).work_on_structure == 0.
    penetrated = _work([0.], [-.001], [10.], [20.])
    assert penetrated.work_on_structure == pytest.approx(-.015)
    assert penetrated.open_gap_work == 0.
    # No force means no work, even with a very large gap change.
    assert _work([.02], [4.], [0.], [0.]).absolute_work_bound == 0.


def test_bound_preserves_large_opposite_work_that_cancels_in_the_sum():
    record = _work([.02, 0.], [0., .02], [0., 10.], [10., 0.], weights=(2., 2.))
    assert record.work_on_structure == pytest.approx(0., abs=1e-16)
    assert record.absolute_work_bound == pytest.approx(.4)


def test_fixed_normal_work_is_work_conjugate_to_nodal_force():
    random = np.random.default_rng(129)
    w = random.uniform(.1, 2., 32)
    n = np.array([1., 2., -3.]) / np.sqrt(14.)
    x0, x1 = random.normal(size=(2, 32, 3))
    p0, p1 = random.uniform(0, 10, (2, 32))
    result = _work(x0@n, x1@n, p0, p1, weights=w)
    force = w[:, None] * (.5*p0 + .5*p1)[:, None] * n
    assert result.work_on_structure == pytest.approx(np.sum(force*(x1-x0)))
    assert abs(result.work_on_structure) <= result.absolute_work_bound
    assert abs(result.open_gap_work) <= result.absolute_work_bound


@pytest.mark.parametrize('field,value', [
    ('weights', [0.]), ('weights', [-1.]), ('old_gap', [np.nan]),
    ('new_gap', [np.inf]), ('new_pressure', [-1.]), ('old_pressure', [1., 2.]),
])
def test_fixed_normal_work_rejects_invalid_samples(field, value):
    arguments = dict(weights=[1.], old_gap=[0.], new_gap=[0.], old_pressure=[0.], new_pressure=[0.])
    arguments[field] = value
    with pytest.raises(ValueError):
        _fixed_normal_work_increment(**arguments)
