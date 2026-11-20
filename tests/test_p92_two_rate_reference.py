from dataclasses import replace

import numpy as np
import pytest

from tools.p92_two_rate_reference import Branch, Material, integrate, rates, stored_energy


def material():
    a = Branch(30., 1e-4, 3., (20000., 5000.), (30., 60.), (.02, .0001))
    return Material(150000., (a, replace(a, reference_flow=2e-5, exponent=2.,
                                       static_recovery=(.001, .00002))))


def test_exact_elastic_and_exponential_static_recovery():
    m = material()
    m = replace(m, branches=tuple(replace(b, strength=1e6) for b in m.branches))
    old = np.zeros(10)
    old[1:3] = [10., -20.]
    result = integrate(old, 20., .0001, m).y[:, -1]
    assert result[0] == pytest.approx(m.modulus*.002)
    np.testing.assert_allclose(result[1:3], old[1:3]*np.exp(-20*np.array(m.branches[0].static_recovery)), rtol=1e-8)
    assert result[3] == result[6] == 0.
    assert result[9] > 0.
    np.testing.assert_array_equal(old[1:3], [10., -20.])


@pytest.mark.parametrize("stress_control", [False, True])
def test_instantaneous_mechanical_identity_and_positive_dissipation(stress_control):
    m = material()
    rng = np.random.default_rng(884)
    for _ in range(20):
        state = np.zeros(10)
        state[0] = rng.uniform(-300, 300)
        state[[1, 2, 4, 5]] = rng.uniform(-20, 20, 4)
        v = rates(state, -.001, m, stress_control=stress_control)
        psi_rate = state[0]*v[0]/m.modulus
        for offset, b in zip((1, 4), m.branches, strict=True):
            psi_rate += sum(state[offset+i]*v[offset+i]/h for i,h in enumerate(b.hardening))
        assert v[9] >= 0.
        assert state[0]*v[7] == pytest.approx(psi_rate+v[9], abs=1e-12)


def test_reversal_hold_restart_and_independent_integrator():
    m = material()
    old = np.zeros(10)
    for duration, rate, stress in [(2., -.001, False), (180., 0., False),
                                   (2., .001, False), (100., 0., True), (2., -.001, False)]:
        full = integrate(old, duration, rate, m, stress_control=stress).y[:, -1]
        half = integrate(old, duration/2, rate, m, stress_control=stress).y[:, -1]
        split = integrate(half, duration/2, rate, m, stress_control=stress).y[:, -1]
        bdf = integrate(old, duration, rate, m, stress_control=stress, method="BDF", rtol=1e-10).y[:, -1]
        np.testing.assert_allclose(full, split, atol=2e-7, rtol=1e-8)
        np.testing.assert_allclose(full, bdf, atol=2e-6, rtol=1e-7)
        old = full
    assert old[8] == pytest.approx(stored_energy(old, m)+old[9], abs=1e-7)


def test_permutation_and_failure_preserve_state(monkeypatch):
    from types import SimpleNamespace
    from tools import p92_two_rate_reference as module
    m = material()
    old = np.zeros(10)
    old[0] = 200.
    result = integrate(old, 10., 0., m, stress_control=True).y[:, -1]
    permutation = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9]
    swapped = integrate(old, 10., 0., replace(m, branches=m.branches[::-1]), stress_control=True).y[:, -1]
    np.testing.assert_allclose(result[permutation], swapped, atol=1e-8)
    monkeypatch.setattr(module, "solve_ivp", lambda *a, **k: SimpleNamespace(success=False))
    with pytest.raises(RuntimeError):
        integrate(old, 1., 0., m)
    np.testing.assert_array_equal(old, [200.]+[0.]*9)


def test_invalid_material_and_state_rejected():
    m = material()
    with pytest.raises(ValueError):
        replace(m.branches[0], static_recovery=(-1., 0.))
    with pytest.raises(ValueError):
        replace(m, modulus=0.)
    old = np.zeros(10)
    old[1] = 31.
    with pytest.raises(ValueError, match="invariant"):
        integrate(old, 1., 0., m)


@pytest.mark.parametrize("index", [3, 6, 9])
def test_interior_accumulated_drawdown_cannot_hide_behind_endpoint(monkeypatch, index):
    from types import SimpleNamespace
    from tools import p92_two_rate_reference as module
    history = np.zeros((10, 4))
    history[index] = [0., .001, .0009, .002]
    monkeypatch.setattr(module, "solve_ivp", lambda *a, **k:
                        SimpleNamespace(success=True, y=history))
    with pytest.raises(RuntimeError, match="Accumulated state decreased"):
        integrate(np.zeros(10), 1., 0., material())
