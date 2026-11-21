from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from tools import p92_coupled_recovery_reference as coupled
from tools import p92_recoverable_spectrum as spectrum
from tools.p92_two_rate_reference import Branch, Material


def material(active=True):
    a = Branch(10. if active else 1000., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    b = replace(a, hardening=(300., 400.), static_recovery=(.03, .04))
    return coupled.Material(spectrum.Material(Material(10000., (a, b)),
                            (20000., 40000.), (.5, 10.)), (.25, -.15))


def state():
    return np.array([70., 1., -2., .01, 3., -1., .02, .01, 0., 0., -.001, .002])


def test_zero_coupling_recovers_reference_rates_energy_and_trajectory():
    m = replace(material(), couplings=(0., 0.))
    y = state()
    np.testing.assert_allclose(coupled.rates(y, .003, m), spectrum.rates(y, .003, m.spectrum), atol=1e-15)
    assert coupled.stored_energy(y, m) == pytest.approx(spectrum.stored_energy(y, m.spectrum))
    np.testing.assert_allclose(coupled.integrate(y, 2., -.002, m).y[:, -1],
                               spectrum.integrate(y, 2., -.002, m.spectrum).y[:, -1], atol=1e-9)


@pytest.mark.parametrize("stress_control", [False, True])
def test_power_identity_finite_difference_and_nonnegative_dissipation(stress_control):
    m, y = material(), state()
    v = coupled.rates(y, .003, m, stress_control=stress_control)
    h = 1e-5
    derivative = (coupled.stored_energy(y+h*v, m)-coupled.stored_energy(y-h*v, m))/(2*h)
    assert v[8] == pytest.approx(derivative+v[9], abs=1e-9)
    assert v[9] >= 0
    assert coupled.stored_energy(y, m) > 0


def test_inactive_flow_matches_coupled_matrix_exponential():
    m = material(False)
    old = np.zeros(12)
    old[[1, 2, 4, 5]] = [5., -7., -3., 9.]
    before = old.copy()
    end = coupled.integrate(old, 30., 0., m, stress_control=True).y[:, -1]
    for i, c in enumerate(coupled.matrices(m)):
        r = np.array([b.static_recovery[i] for b in m.spectrum.backbone.branches])
        indices = [1+i, 4+i]
        expected = expm(-c@np.diag(r/np.diag(c))*30.)@old[indices]
        np.testing.assert_allclose(end[indices], expected, atol=1e-9)
    assert end[9] == pytest.approx(coupled.stored_energy(old, m)-coupled.stored_energy(end, m), abs=1e-9)
    np.testing.assert_array_equal(before, old)


def test_independent_integration_and_split_continuation():
    m, y = material(), state()
    full = coupled.integrate(y, 2., -.002, m).y[:, -1]
    bdf = coupled.integrate(y, 2., -.002, m, method="BDF", rtol=1e-10).y[:, -1]
    first = coupled.integrate(y, .7, -.002, m).y[:, -1]
    split = coupled.integrate(first, 1.3, -.002, m).y[:, -1]
    np.testing.assert_allclose(full, bdf, atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(full, split, atol=1e-8, rtol=1e-7)


def test_sign_and_branch_permutation_symmetry():
    m, y = material(), state()
    sign = np.array([-1., -1., -1., 1., -1., -1., 1., -1., 1., 1., -1., -1.])
    v = coupled.rates(y, .003, m)
    np.testing.assert_allclose(coupled.rates(y*sign, -.003, m), sign*v)
    permutation = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11]
    swapped = replace(m, spectrum=replace(m.spectrum,
        backbone=replace(m.spectrum.backbone, branches=m.spectrum.backbone.branches[::-1])))
    np.testing.assert_allclose(coupled.rates(y[permutation], .003, swapped), v[permutation])
    assert coupled.stored_energy(y[permutation], swapped) == pytest.approx(coupled.stored_energy(y, m))


@pytest.mark.parametrize("couplings", [(1., 0.), (-1., 0.), (float("nan"), 0.), (.1,)])
def test_reject_nonpositive_energy(couplings):
    with pytest.raises(ValueError, match="positive definite"):
        replace(material(), couplings=couplings)


def test_reject_unsupported_history_and_invalid_state_without_mutation():
    m = material()
    with pytest.raises(ValueError, match="constant"):
        replace(m, spectrum=replace(m.spectrum, hardening_ratios=(.8, 1.)))
    for old in (np.zeros(11), np.full(12, np.nan), np.full(12, 1j)):
        with pytest.raises(ValueError):
            coupled.integrate(old, 1., 0., m)
    assert coupled.MODEL_ID != spectrum.MODEL_ID


@pytest.mark.parametrize("control", [False, True])
def test_rate_accounting_retains_signed_recoverable_and_dissipative_parts(control):
    from tools.p92_spectrum_rate_budget import rate_budget
    m, y = material().spectrum, state()
    report = rate_budget(y, m, stress_control=control, strain_rate=.003)
    assert abs(report["rate_closure_per_s"]) < 1e-15
    assert report["recoverable_rates_per_s"][1] < 0
    assert report["branches"][0]["signed_flow_per_s"] > 0
    inactive = rate_budget(np.zeros(12), m, stress_control=control)
    assert inactive["total_nonelastic_rate_per_s"] == 0


def test_rate_report_requires_matching_numerical_state_and_source_identity():
    from copy import deepcopy
    from dataclasses import asdict
    from tools.p92_spectrum_rate_budget import report
    segment = dict(segment=0, cycle=1, stress_control=False, end_state=state().tolist())
    prediction = dict(protocol="RF", source_sha256="test", model_id=spectrum.MODEL_ID,
        state_layout=spectrum.STATE_LAYOUT, segments=[segment],
        hold_diagnostics=[dict(cycle=1, segment=0, stress_control=False, duration_s=180.)])
    frozen = dict(parameters=asdict(material().spectrum), model_id=spectrum.MODEL_ID,
        source_sha256={"RF": "test"}, protocols={"RF": dict(numerical=dict(passed=True), prediction=prediction)})
    assert abs(report(frozen)["protocols"]["RF"][0]["endpoint"]["rate_closure_per_s"]) < 1e-15
    for mutation in ("source", "model", "segment", "cycle", "numerical"):
        bad = deepcopy(frozen)
        p = bad["protocols"]["RF"]["prediction"]
        if mutation == "source":
            p["source_sha256"] = "other"
        elif mutation == "model":
            p["model_id"] = coupled.MODEL_ID
        elif mutation == "segment":
            p["segments"][0]["segment"] = 2
        elif mutation == "cycle":
            p["hold_diagnostics"][0]["cycle"] = 2
        else:
            bad["protocols"]["RF"]["numerical"]["passed"] = False
        with pytest.raises(ValueError):
            report(bad)
