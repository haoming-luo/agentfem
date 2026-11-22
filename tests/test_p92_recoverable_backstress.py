from dataclasses import replace

import numpy as np
import pytest

from tools import p92_recoverable_backstress_reference as kernel
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_two_rate_reference import Branch, Material


def material(coupling=.2):
    a = Branch(10., 1e-4, 3., (1000., 2000.), (20., 30.), (0., 0.))
    b = replace(a, strength=20., static_recovery=(.1, .01))
    return kernel.Material(spectrum.Material(Material(10000., (a, b)),
                                            (20000., 30000.), (.5, 10.)), (coupling, coupling))


@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("coupling", [-.25, 0., .25])
def test_potential_power_balance_and_nonnegative_dissipation(control, coupling):
    mat = material(coupling)
    state = np.zeros(12)
    state[:3] = [50., -3., 4.]
    state[4:6], state[10:] = [5., -6.], [.0003, -.0002]
    rate = kernel.rates(state, .001, mat, stress_control=control)
    h = np.asarray(mat.spectrum.backbone.branches[1].hardening)
    c = coupling*np.sqrt(h*np.asarray(mat.spectrum.moduli))
    b = (state[4:6]-c*state[10:])/h
    direct = state[0]**2/(2*10000.)+sum(state[1:3]**2/(2*np.array([1000., 2000.])))
    direct += sum(h*b*b/2+np.asarray(mat.spectrum.moduli)*state[10:]**2/2+c*b*state[10:])
    assert kernel.stored_energy(state, mat) == pytest.approx(direct)
    step = 1e-6
    derivative = (kernel.stored_energy(state+step*rate, mat)-kernel.stored_energy(state-step*rate, mat))/(2*step)
    assert derivative+rate[9] == pytest.approx(rate[8], abs=1e-8)
    assert rate[9] >= 0.


def test_zero_coupling_is_original_reference():
    mat = material(0.)
    old = np.zeros(12)
    actual = kernel.integrate(old, 2., .002, mat)
    expected = spectrum.integrate(old, 2., .002, mat.spectrum)
    np.testing.assert_allclose(actual.y[:, -1], expected.y[:, -1], atol=1e-9, rtol=1e-8)


def test_integrators_split_path_and_sign_symmetry():
    mat = material()
    old = np.zeros(12)
    full = kernel.integrate(old, 2., .002, mat)
    bdf = kernel.integrate(old, 2., .002, mat, method="BDF")
    half = kernel.integrate(old, 1., .002, mat)
    split = kernel.integrate(half.y[:, -1], 1., .002, mat)
    negative = kernel.integrate(old, 2., -.002, mat)
    np.testing.assert_allclose(bdf.y[:, -1], full.y[:, -1], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(split.y[:, -1], full.y[:, -1], atol=1e-7, rtol=1e-7)
    sign = np.ones(12)
    sign[[0, 1, 2, 4, 5, 7, 10, 11]] = -1
    np.testing.assert_allclose(negative.y[:, -1], sign*full.y[:, -1], atol=1e-9)


def test_rejects_indefinite_potential_and_added_coefficients():
    with pytest.raises(ValueError):
        material(1.)
    mat = material()
    a, b = mat.spectrum.backbone.branches
    changed = replace(mat.spectrum, backbone=replace(mat.spectrum.backbone,
                      branches=(replace(a, static_recovery=(.1, .1)), b)))
    with pytest.raises(ValueError, match="replace"):
        kernel.Material(changed, (.1, .1))
    with pytest.raises(ValueError, match="original recovery"):
        kernel.Material(replace(mat.spectrum, directional_recovery=True), (.1, .1))


def test_failed_integration_preserves_input(monkeypatch):
    from types import SimpleNamespace
    old = np.zeros(12)
    def fail(fun, time, y, **kwargs):
        y[0] = 999.
        return SimpleNamespace(success=False)
    monkeypatch.setattr(kernel, "solve_ivp", fail)
    with pytest.raises(RuntimeError):
        kernel.integrate(old, 1., .001, material())
    assert not old.any()


def test_delayed_strain_is_recoverable_without_irreversible_flow():
    mat = material(.25)
    branches = tuple(replace(b, strength=1e6, static_recovery=(0., 0.))
                     for b in mat.spectrum.backbone.branches)
    mat = replace(mat, spectrum=replace(mat.spectrum, backbone=replace(mat.spectrum.backbone, branches=branches)))
    old = np.zeros(12)
    old[10:] = [.001, -.002]
    _, c = kernel.forces(old, mat)
    old[4:6] = c*old[10:]  # b=0: no hidden permanent strain is prescribed.
    solution = kernel.integrate(old, 2., 0., mat, stress_control=True)
    expected = old[10:]*np.exp(-2/np.asarray(mat.spectrum.times))
    np.testing.assert_allclose(solution.y[10:, -1], expected, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(solution.y[4:6, -1], c*expected, rtol=1e-8, atol=1e-10)
    assert solution.y[[3, 6], -1].sum() == 0.
