from types import SimpleNamespace

import numpy as np
import pytest

from tools.p92_equal_stress_hold import mechanism_partition, observed_interval, predicted_interval


def test_observed_increment_retains_strain_drift_and_units():
    value = observed_interval([0., 1., 2.], [0., 1e-5, 2e-5], [220., 210., 200.], 100000.)
    assert value["average_nonelastic_rate_per_s"] == pytest.approx(1.1e-4)
    assert value["one_quantum_endpoint_sensitivity_per_s"] == pytest.approx(1.01e-5)


def test_equal_stress_interval_uses_predicted_crossings_not_measured_time():
    def sol(t):
        t = np.asarray(t)
        y = np.zeros((12, *t.shape))
        y[0] = 250.-2*t
        return y
    solution = SimpleNamespace(t=np.linspace(0., 100., 21), sol=sol)
    solution.y = sol(solution.t)
    actual = predicted_interval(solution, [220., 200.], 100000.)
    assert actual["times_s"] == pytest.approx([15., 25.])
    assert actual["duration_s"] == pytest.approx(10.)
    assert actual["average_nonelastic_rate_per_s"] == pytest.approx(2e-5)
    with pytest.raises(ValueError, match="crossing"):
        predicted_interval(solution, [300., 200.], 100000.)


@pytest.mark.parametrize("time,strain,stress", [
    ([0., 1.], [0., 0.], [200., 190.]),
    ([0., 1., 1.], [0., 0., 0.], [200., 190., 180.]),
    ([0., 1., 2.], [0., float("nan"), 0.], [200., 190., 180.]),
    ([0., 1., 2.], [0., 0., 0.], [180., 190., 200.]),
])
def test_reject_ambiguous_or_invalid_observations(time, strain, stress):
    with pytest.raises(ValueError):
        observed_interval(time, strain, stress, 100000.)


@pytest.mark.parametrize("powers", [(1., 1.), (1.5, 2.)])
def test_partition_closes_and_preserves_recoverable_sign(powers):
    from tools import p92_recoverable_spectrum as spectrum
    from tools.p92_two_rate_reference import Branch, Material
    branch = Branch(10., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    mat = spectrum.Material(Material(10000., (branch, branch)),
        (20000., 40000.), (.5, 10.), recoverable_powers=powers)
    old = np.zeros(12)
    old[0], old[7], old[10] = 100., .01, .01
    sol = spectrum.integrate(old, 2., 0., mat)
    part = mechanism_partition(sol, dict(times_s=[.1, 1.5]), mat)
    assert abs(part["strain_closure_residual"]) < 1e-9
    assert part["strain_increments"][2] < 0
    assert len(part["average_rates_per_s"]) == 4
    assert part["strain_increments"][0] > 0
    for times in ([1.5, .1], [-.1, 1.5], [.1, 3.], [np.nan, 1.5]):
        with pytest.raises(ValueError, match="ordered"):
            mechanism_partition(sol, dict(times_s=times), mat)


def test_transient_resistance_is_not_counted_as_a_strain_component():
    from tools import p92_transient_strength_reference as transient
    from tools import p92_recoverable_spectrum as spectrum
    from tools.p92_two_rate_reference import Branch, Material
    branch = Branch(10., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    base = spectrum.Material(Material(10000., (branch, branch)),
                             (20000., 40000.), (.5, 10.))
    mat = transient.Material(base, .1, 250., 1/1800.)
    old = np.zeros(13)
    old[0], old[7], old[10] = 100., .01, .01
    sol = transient.integrate(old, 2., 0., mat)
    part = mechanism_partition(sol, dict(times_s=[.1, 1.5]), mat, module=transient)
    assert abs(part["strain_closure_residual"]) < 1e-9
    assert len(part["strain_increments"]) == 4
    assert part["end_state"][12] > part["start_state"][12]
    assert part["strain_increments"][2] < 0.
    with pytest.raises(ValueError, match="supported"):
        mechanism_partition(sol, dict(times_s=[.1, 1.5]), mat, module=None)
