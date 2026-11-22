from dataclasses import replace

import numpy as np
import pytest

from tools.p92_history_memory_audit import attenuation_bounds
from tools.p92_recoverable_spectrum import Material as Spectrum
from tools.p92_two_rate_reference import Branch, Material


def material():
    b = Branch(20., 1e-5, 3., (100., 200.), (10., 20.), (.1, .01))
    return Spectrum(Material(10000., (b, b)), (20000., 40000.), (.5, 10.))


def test_exact_constant_static_attenuation_and_regenerated_final_state():
    start, end = np.zeros(12), np.zeros(12)
    start[1:3] = [-2., 3.]
    end[1:3] = [4., -5.]
    rows = attenuation_bounds(material(), start, end, 10.)
    np.testing.assert_allclose(rows[0]["homogeneous_memory_upper"], np.exp([-1., -.1]))
    np.testing.assert_allclose(rows[0]["initial_contribution_abs_upper_MPa"], [2*np.exp(-1), 3*np.exp(-.1)])
    assert rows[0]["final_backstress_MPa"] == [4., -5.]  # Not bounded by the homogeneous part.


def test_monotone_shared_history_bounds_actual_static_integral():
    from scipy.integrate import quad
    mat = replace(material(), static_ratios=(.4, .6), static_history_rates=(200., 300.), static_history="shared")
    start, end = np.zeros(12), np.zeros(12)
    start[[3, 6]], end[[3, 6]] = [.001, .002], [.002, .003]
    rows = attenuation_bounds(mat, start, end, 10.)
    for j, row in enumerate(rows):
        integral = quad(lambda t, j=j: mat.static_ratios[j]+(1-mat.static_ratios[j])
                        *np.exp(-mat.static_history_rates[j]*(.003+.002*t/10.)), 0., 10.)[0]
        exact = np.exp(-np.asarray(mat.backbone.branches[j].static_recovery)*integral)
        assert np.all(exact >= row["static_only_multiplier_lower"])
        assert np.all(exact <= row["homogeneous_memory_upper"])


def test_wrong_history_or_evolution_cannot_claim_a_bound():
    old = np.zeros(12)
    old[3] = .01
    with pytest.raises(ValueError):
        attenuation_bounds(material(), old, np.zeros(12), 10.)
    with pytest.raises(ValueError):
        attenuation_bounds(replace(material(), static_powers=(1., 0.)), old, old, 10.)


@pytest.mark.parametrize("directional", [False, True])
def test_signed_backstress_budget_closes_with_regeneration(directional):
    from tools.p92_history_memory_audit import backstress_budget
    from tools.p92_recoverable_spectrum import integrate
    mat = replace(material(), directional_recovery=directional)
    old = np.zeros(12)
    old[0], old[1], old[4] = 40., -2., -2.
    solution = integrate(old, 2., 0., mat, stress_control=True)
    budget = backstress_budget(solution, mat, stress_control=True)
    np.testing.assert_allclose(budget["closure_MPa"], 0., atol=1e-7)
    assert np.asarray(budget["signed_increments_MPa"])[:, 0].min() > 0.
    assert budget["net_change_MPa"][0] > 0.


def test_factor_contrast_matches_independent_scalar_maximization():
    from scipy.optimize import minimize_scalar
    from tools.p92_history_memory_audit import maximum_factor_contrast
    p, q, rho = .00129, .00135, .51
    exact = maximum_factor_contrast(p, q, rho)
    fit = minimize_scalar(lambda b: -(1-rho)*(np.exp(-b*p)-np.exp(-b*q)),
                          bounds=(0., 10000.), method="bounded")
    assert exact["maximum_absolute_contrast"] == pytest.approx(-fit.fun, rel=1e-10)
    assert exact["rate_at_maximum"] == pytest.approx(fit.x, rel=1e-5)
    assert maximum_factor_contrast(q, p, rho) == exact


def test_factor_contrast_handles_zero_equal_and_nearly_equal_history():
    from tools.p92_history_memory_audit import maximum_factor_contrast as contrast
    assert contrast(.001, .001, .5)["maximum_absolute_contrast"] == 0.
    assert contrast(.001, .002, 1.)["maximum_absolute_contrast"] == 0.
    assert contrast(0., .001, .5) == dict(maximum_absolute_contrast=.5,
                                        rate_at_maximum=None, attained=False)
    assert 0 < contrast(.001, .001+1e-15, .5)["maximum_absolute_contrast"] < 1e-12
    with pytest.raises(ValueError):
        contrast(-.001, .001, .5)
