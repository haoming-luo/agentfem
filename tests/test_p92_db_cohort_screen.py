"""Source bookkeeping and independent integration, not material qualification."""

import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from agentfem.constitutive import KachanovRabotnovCreep
from tools.p92_db_cohort_screen import SOURCE, rupture_curve, screen


def test_source_and_protocol_are_not_pooled_with_fully_reversed_cohorts():
    data = json.loads(SOURCE.read_text())
    assert len(data["source_sha256"]) == 64
    assert len(data["creep"]) == 12
    assert len(data["fatigue"]) == len(data["dwell"]) == 3
    report = screen()
    json.dumps(report, allow_nan=False)
    assert not report["engineering_qualified"] and not report["fitted_parameters"]
    for row in report["cycles"]:
        assert row["minimum_strain"] == 0
        assert row["mean_strain"] == row["strain_amplitude"]
        assert row["strain_range"] == 2 * row["strain_amplitude"]
        assert row["period_s"] == pytest.approx(row["calculated_period_s"])
    # The 600 s hold duration is not nearest-hour rounding of N * period.
    # Retain both published numbers rather than changing N or the waveform.
    hold = next(row for row in report["cycles"] if row["dwell_s"] == 600)
    assert hold["reported_duration_h"] == 304
    assert hold["calculated_duration_h"] == pytest.approx(304.8516666666667)


@pytest.mark.parametrize("temperature,a,n,chi,b,phi", json.loads(SOURCE.read_text())["constants"])
def test_core_reproduces_published_equation_and_independent_time_ode(temperature, a, n, chi, b, phi):
    observations = [r for r in json.loads(SOURCE.read_text())["creep"] if r[0] == temperature]
    for _, stress, _, _ in observations:
        law = KachanovRabotnovCreep(a, n, b, chi, phi)
        exact_life = 1 / (b * (1 + phi) * stress**chi)
        assert law.rupture_time(stress) == pytest.approx(exact_life, rel=1e-13)
        # Independent normalized physical-time ODE retains both evolving states.
        def rhs(_, y, exact_life=exact_life, stress=stress):
            return [exact_life * a * stress**n / (1 - y[1])**n,
                    exact_life * b * stress**chi / (1 - y[1])**phi]
        reference = solve_ivp(rhs, (0, .9), [0, 0], rtol=1e-10, atol=1e-12,
                              method="DOP853", t_eval=[.1, .5, .9])
        assert reference.success
        previous = 0.
        state = None
        for fraction, expected in zip(reference.t, reference.y.T, strict=True):
            update = law.update(stress, (fraction - previous) * exact_life, state)
            direct = law.update(stress, fraction * exact_life)
            state = update.state
            exponent = (1 + phi - n) / (1 + phi)
            equation = a * stress**(n - chi) / (b * (n - phi - 1)) * ((1 - fraction)**exponent - 1)
            assert state.equivalent_creep_strain == pytest.approx(equation, rel=1e-12)
            np.testing.assert_allclose([state.equivalent_creep_strain, state.damage], expected, rtol=2e-8, atol=1e-11)
            assert state.equivalent_creep_strain == pytest.approx(direct.state.equivalent_creep_strain, rel=1e-12)
            assert not update.failed
            previous = fraction


def test_discrepancies_are_visible_not_corrected_by_retuning():
    report = screen()
    row = next(r for r in report["creep"] if r["temperature_C"] == 650 and r["stress_MPa"] == 120)
    assert row["predicted_rupture_h"] == pytest.approx(868.9822316005437)
    assert row["life_relative_error"] < -.28
    # Fixed 675 C / 700 C coefficients yield pre-rupture strain far above
    # the approximate measured failure strain. This must remain visible.
    for row in report["creep"]:
        if row["temperature_C"] > 650:
            assert row["strain_at_90_percent_model_life"] > 10 * row["approximate_observed_failure_strain"]


@pytest.mark.parametrize("temperature", [650, 675, 700])
def test_experimental_rupture_nodes_units_and_range_are_preserved(temperature):
    curve = rupture_curve(temperature)
    converted = curve.in_units(stress_unit="Pa", time_unit="s")
    for stress, duration in zip(curve.stresses, curve.rupture_times, strict=True):
        assert curve(stress, temperature + 273.15) == duration
        assert converted(stress * 1e6, curve.temperature) == duration * 3600
    with pytest.raises(ValueError, match="extrapolation"):
        curve(300, curve.temperature)
    with pytest.raises(ValueError, match="isothermal"):
        curve(curve.stresses[0], curve.temperature + 1)
    with pytest.raises(ValueError, match="three measured"):
        rupture_curve(660)


def test_sparse_interpolation_uses_only_declared_nodes_and_independent_log_formula():
    result = screen()
    assert "not a blind" in result["interpolation_role"]
    assert sum(len(r["comparisons"]) for r in result["interpolation"]) == 5
    for record in result["interpolation"]:
        stresses = np.asarray(record["curve"]["stresses"])
        times = np.asarray(record["curve"]["rupture_times"])
        for row in record["comparisons"]:
            stress = row["stress_MPa"]
            assert stress not in stresses
            right = np.searchsorted(stresses, stress)
            left = right - 1
            weight = np.log(stress / stresses[left]) / np.log(stresses[right] / stresses[left])
            expected = times[left] * (times[right] / times[left])**weight
            assert row["predicted_h"] == pytest.approx(expected, rel=1e-13)
