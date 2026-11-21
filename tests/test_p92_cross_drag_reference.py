from dataclasses import replace

import numpy as np
import pytest

from tools import p92_cross_drag_reference as cross
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_two_rate_reference import Branch, Material


def material(active=True):
    branch = Branch(10. if active else 1000., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    s = spectrum.Material(Material(10000., (branch, branch)), (20000., 40000.), (.5, 10.))
    return cross.Material(s, 2., 1000., 1000., 1/1800.)


def state():
    return np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001, .002, .3])


def test_zero_interaction_recovers_rates_energy_and_trajectory():
    m = replace(material(), maximum=0.)
    y = state(); y[12] = 0.
    np.testing.assert_array_equal(cross.rates(y, .003, m)[:12], spectrum.rates(y[:12], .003, m.spectrum))
    assert cross.rates(y, .003, m)[12] == 0
    assert cross.stored_energy(y, m) == spectrum.stored_energy(y[:12], m.spectrum)
    np.testing.assert_allclose(cross.integrate(y, 2., -.002, m).y[:12, -1],
        spectrum.integrate(y[:12], 2., -.002, m.spectrum).y[:, -1], atol=1e-9)


@pytest.mark.parametrize("stress_control", [False, True])
def test_power_identity_including_changed_flow_and_signed_symmetry(stress_control):
    m, y = material(), state()
    v = cross.rates(y, .003, m, stress_control=stress_control)
    h = 1e-5
    psi_rate = (cross.stored_energy(y+h*v, m)-cross.stored_energy(y-h*v, m))/(2*h)
    assert v[8] == pytest.approx(psi_rate+v[9], abs=1e-9)
    assert v[9] >= 0
    sign = np.array([-1., -1., -1., 1., -1., -1., 1., -1., 1., 1., -1., -1., 1.])
    np.testing.assert_allclose(cross.rates(y*sign, -.003, m, stress_control=stress_control), v*sign)


def test_inactive_history_exact_decay_and_preserved_input():
    m = material(False)
    old = np.zeros(13); old[12] = 1.
    before = old.copy()
    end = cross.integrate(old, 1800., 0., m).y[:, -1]
    assert end[12] == pytest.approx(np.exp(-1.), abs=1e-10)
    np.testing.assert_array_equal(old, before)
    np.testing.assert_allclose(end[:12], 0.)


def test_inward_boundary_rates_independent_solvers_and_continuation():
    m, y = material(), state()
    y[12] = 0.
    assert cross.rates(y, .003, m)[12] >= 0
    y[12] = m.maximum
    assert cross.rates(y, .003, m)[12] <= 0
    y[12] = .3
    full = cross.integrate(y, 2., -.002, m).y[:, -1]
    bdf = cross.integrate(y, 2., -.002, m, method="BDF", rtol=1e-10).y[:, -1]
    first = cross.integrate(y, .7, -.002, m).y[:, -1]
    split = cross.integrate(first, 1.3, -.002, m).y[:, -1]
    np.testing.assert_allclose(full, bdf, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(full, split, atol=1e-8, rtol=1e-7)
    assert 0 <= full[12] <= m.maximum


def test_more_resistance_suppresses_only_branch_b_at_identical_state():
    m, y = material(), state()
    y[12] = 0.
    zero = cross.rates(y, .003, m)
    y[12] = 1.
    one = cross.rates(y, .003, m)
    assert one[3] == zero[3]
    assert one[6] == pytest.approx(zero[6]/np.e)


@pytest.mark.parametrize("changes", [dict(maximum=-1.), dict(build=-1.),
    dict(erase=np.nan), dict(recovery=-1.)])
def test_invalid_parameters(changes):
    with pytest.raises(ValueError):
        replace(material(), **changes)


def test_wrong_state_and_history_rejected():
    m = material()
    for old in (np.zeros(12), np.full(13, 1j), np.full(13, np.nan)):
        with pytest.raises(ValueError):
            cross.integrate(old, 1., 0., m)
    y = state(); y[12] = -1.
    with pytest.raises(ValueError):
        cross.integrate(y, 1., 0., m)
    assert cross.MODEL_ID != spectrum.MODEL_ID
