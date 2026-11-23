from dataclasses import replace

import numpy as np
import pytest

from tools import p92_recoverable_spectrum as spectrum
from tools import p92_reversal_memory_reference as memory
from tools.p92_two_rate_reference import Branch, Material


def material():
    b = Branch(10., 1e-3, 3., (1000., 2000.), (10., 20.), (.01, .02))
    return memory.Material(spectrum.Material(Material(10000., (b, b)),
        (20000., 40000.), (.5, 10.)))


def test_zero_memory_matches_spectrum_with_explicit_signed_flow():
    mat = material()
    old = np.zeros(13)
    old[7] = .007  # Experimental strain offset must not seed the flow memory.
    a = memory.integrate(old, 1., .01, mat)
    b = spectrum.integrate(old[:12], 1., .01, mat.spectrum)
    np.testing.assert_allclose(a.y[:12, -1], b.y[:, -1], rtol=1e-7, atol=1e-8)
    expected = a.y[7, -1]-old[7]-a.y[0, -1]/10000.-sum(a.y[10:12, -1])
    assert a.y[12, -1] == pytest.approx(expected, abs=1e-12)
    np.testing.assert_array_equal(old, np.r_[np.zeros(7), .007, np.zeros(5)])


@pytest.mark.parametrize("coordinate", ["signed", "accumulated"])
@pytest.mark.parametrize("stress_control", [False, True])
def test_memory_power_identity_and_signed_symmetry(coordinate, stress_control):
    mat = replace(material(), ratios=(2., .5), coordinate=coordinate)
    state = np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001, .002, -.003])
    result = memory.rates(state, .003, mat, stress_control=stress_control)
    step = 1e-5
    energy_rate = (memory.stored_energy(state+step*result, mat)
                   - memory.stored_energy(state-step*result, mat))/(2*step)
    assert result[8] == pytest.approx(energy_rate+result[9], abs=1e-9)
    assert result[9] >= 0
    sign = np.ones(13)
    sign[[0, 1, 2, 4, 5, 7, 10, 11, 12]] = -1
    np.testing.assert_allclose(memory.rates(state*sign, -.003, mat, stress_control=stress_control),
                               result*sign, atol=1e-12)


def test_memory_reversal_split_and_independent_integration():
    mat = replace(material(), ratios=(2., .5))
    old = memory.integrate(np.zeros(13), 2., .03, mat).y[:, -1]
    before = old.copy()
    a = memory.integrate(old, 5., -.03, mat)
    b = memory.integrate(old, 5., -.03, mat, method="BDF", rtol=1e-10)
    split = memory.integrate(old, 1., -.03, mat)
    c = memory.integrate(split.y[:, -1], 4., -.03, mat)
    np.testing.assert_allclose(a.y[:, -1], b.y[:, -1], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(a.y[:, -1], c.y[:, -1], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(old, before)
    assert old[12] > 0 > a.y[12, -1]
    assert a.y[3, -1]+a.y[6, -1] > old[3]+old[6]


def test_signed_and_accumulated_memory_distinguish_same_total_history():
    mat = replace(material(), ratios=(2., .5))
    state = np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001, .002, 0.])
    none = memory.rates(state, .003, material())
    np.testing.assert_allclose(memory.rates(state, .003, mat), none)
    assert not np.allclose(memory.rates(state, .003, replace(mat, coordinate="accumulated")), none)
    with pytest.raises(ValueError, match="state"):
        memory.integrate(state[:12], 1., 0., mat)
    state[12] = .1
    with pytest.raises(ValueError, match="state"):
        memory.integrate(state, 1., 0., mat)
