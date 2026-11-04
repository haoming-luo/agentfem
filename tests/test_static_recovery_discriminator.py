"""Closed-form evidence for a mechanism, not a new material capability claim."""

import numpy as np
import pytest

from tools.static_recovery_discriminator import (
    Parameters, State, activation_time, exact_hold, refinement_report, update,
)


def test_refinement_and_reactivation():
    report = refinement_report()
    assert report["activation_time"] == pytest.approx(np.log(2.) / .02)
    errors = [r["maximum_stress_error"] for r in report["refinement"]]
    assert all(.4 < b / a < .6 for a, b in zip(errors[:-1], errors[1:]))
    assert errors[-1] < .1


@pytest.mark.parametrize("k", [0., .02])
def test_inactive_strength_can_change_without_stress_relaxation(k):
    p = Parameters(200000., 10000., 100., k)
    initial = State(80., 100., .01)
    current = initial
    for dt in (1., 3., 7., 20.):
        current, tangent = update(current, 0., dt, p)
        assert current.stress == initial.stress
        assert current.plastic_strain == initial.plastic_strain
        assert tangent == p.young
    assert activation_time(initial, p) == np.inf
    assert (current.strength < initial.strength) == (k > 0)


@pytest.mark.parametrize("increment", [0., .001])
def test_consistent_tangent_and_immutable_old_state(increment):
    p = Parameters(200000., 10000., 100., .02)
    old = State(150., 100., .01)
    current, tangent = update(old, increment, .4, p)
    h = 1.e-8
    numerical = (update(old, increment+h, .4, p)[0].stress
                 - update(old, increment-h, .4, p)[0].stress) / (2*h)
    assert tangent == pytest.approx(numerical, rel=1.e-7)
    assert current.stress <= p.yield_stress + current.strength + 1.e-10
    assert current.plastic_strain >= old.plastic_strain
    assert old == State(150., 100., .01)


def test_exact_hold_energy_decreases_and_activation_is_continuous():
    p = Parameters(200000., 10000., 100., .02)
    old = State(150., 100., .01)
    ta = activation_time(old, p)
    times = np.unique(np.r_[np.linspace(0., 100., 1001), ta])
    stress, strength, plastic = exact_hold(times, old, p)
    energy = stress**2/(2*p.young) + strength**2/(2*p.hardening)
    assert np.all(np.diff(energy) <= 0)
    assert np.all(np.diff(plastic) >= 0)
    assert stress[times < ta] == pytest.approx(old.stress)
    assert strength[times == ta] == pytest.approx(old.stress-p.yield_stress)


def test_scalar_exact_query_and_initial_yield_surface():
    p = Parameters(200000., 10000., 100., .02)
    initial = State(150., 50.)
    assert activation_time(initial, p) == 0.
    stress, strength, _ = exact_hold(1., initial, p)
    assert stress == pytest.approx(100. + 50. * np.exp(-.02*200000./210000.))
    assert stress == pytest.approx(p.yield_stress + strength)


@pytest.mark.parametrize("dt", [0., -1., np.nan, np.inf, True])
def test_invalid_time_rejected(dt):
    with pytest.raises(ValueError):
        update(State(150., 100.), 0., dt, Parameters(200000., 10000., 100., .02))
