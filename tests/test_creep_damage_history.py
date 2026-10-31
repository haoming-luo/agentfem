# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import numpy as np
import pytest

from agentfem import assessments, results
from agentfem.constitutive.fatigue import FatigueAssessment
from agentfem.constitutive.fatigue import BasquinCurve, assess_history


def _result(stress_times=(0., 1., 3.), stress=(100., 80., 40.),
            temperature_times=(0., 2., 3.), temperature=(800., 800., 800.)):
    result = results.SimulationResult("relaxation")
    result.add_history("stress", stress_times, stress, unit="MPa")
    result.add_history("temperature", temperature_times, temperature, unit="K")
    return result


def _integrate(result=None, **changes):
    options = dict(stress_history="stress", temperature_history="temperature",
                   dwells=(assessments.DwellInterval(0., 3.),),
                   rupture_time=lambda stress, temperature: 100_000. / stress,
                   rupture_source="synthetic verification in MPa,K,s", stress_unit="MPa",
                   time_unit="s", stress_measure="signed_axial")
    options.update(changes)
    return assessments.creep_damage_from_result(_result() if result is None else result, **options)


def test_union_grid_integrates_linear_rate_exactly_and_keeps_provenance():
    result = _result()
    original = result.histories["stress"].values.copy()
    assessment = _integrate(result)
    assert assessment.damage == pytest.approx(210. / 100_000.)
    assert len(assessment.intervals) == 3
    assert [s.start for s in assessment.intervals] == [0., 1., 2.]
    assert assessment.maximum_stress_sample_interval == 2.
    assert assessment.maximum_temperature_sample_interval == 2.
    record = json.loads(json.dumps(assessment.as_dict(), allow_nan=False))
    assert record["source"] == "relaxation:stress,temperature"
    assert record["rupture_source"].startswith("synthetic")
    assert record["temperature_unit"] == "K"
    assert record["intervals"][1]["stress"] == [80., 60.]
    assert not record["quadrature_error_certified"]
    np.testing.assert_array_equal(result.histories["stress"].values, original)


def test_clipped_dwell_endpoints_and_repetitions():
    assessment = _integrate(dwells=(assessments.DwellInterval(.5, 2.5, repetitions=3.),))
    assert assessment.damage == pytest.approx(3. * 2. * 70. / 100_000.)
    assert assessment.intervals[0].start == .5
    assert assessment.intervals[-1].end == 2.5
    assert all(s.repetitions == 3. for s in assessment.intervals)


@pytest.mark.parametrize("start,end,gap", [(0., 1., 1.), (1., 3., 2.),
    (1.2, 1.8, 2.), (0., 3., 2.), (2., 3., 2.)])
def test_indexed_dwell_preserves_enclosing_source_gap(start, end, gap):
    assessment = _integrate(dwells=(assessments.DwellInterval(start, end),))
    assert assessment.maximum_stress_sample_interval == gap
    assert assessment.intervals[0].start == start
    assert assessment.intervals[-1].end == end
    assert assessment.damage == pytest.approx(
        ((100. * end - 10. * end**2) - (100. * start - 10. * start**2)) / 100_000.)


def test_adjacent_dwells_partition_damage_without_duplicate_intervals():
    assessment = _integrate(dwells=tuple(
        assessments.DwellInterval(start, end)
        for start, end in [(0., .5), (.5, 1.), (1., 2.), (2., 3.)]))
    assert assessment.damage == pytest.approx(_integrate().damage)
    assert len(assessment.intervals) == 4


def test_damage_rate_not_rupture_life_is_averaged():
    assessment = _integrate(_result((0., 2.), (1., 2.), (0., 2.), (800., 800.)),
                            dwells=(assessments.DwellInterval(0., 2.),),
                            rupture_time=lambda stress, temperature: stress)
    assert assessment.damage == pytest.approx(1.5)
    assert assessment.damage != pytest.approx(2. / 1.5)


def test_same_physical_history_in_hours_and_seconds_has_same_damage():
    seconds = _result()
    hours = _result()
    for name, history in hours.histories.items():
        hours.histories[name] = replace(history, abscissa=history.abscissa / 3600., abscissa_unit="h")
    assessment = _integrate(hours, time_unit="h",
        dwells=(assessments.DwellInterval(0., 3. / 3600.),),
        rupture_time=lambda stress, temperature: 100_000. / stress / 3600.)
    assert assessment.damage == pytest.approx(_integrate(seconds).damage)


@pytest.mark.parametrize("name, changes", [
    ("stress", dict(unit="Pa")), ("temperature", dict(unit="degC")),
    ("stress", dict(abscissa_unit="h")), ("temperature", dict(abscissa_name="load_factor")),
])
def test_unit_or_pseudo_time_mismatch_is_not_silently_converted(name, changes):
    result = _result()
    result.histories[name] = replace(result.histories[name], **changes)
    with pytest.raises(ValueError):
        _integrate(result)


@pytest.mark.parametrize("dwells", [(),
    (assessments.DwellInterval(-.1, 2.),), (assessments.DwellInterval(0., 3.1),),
    (assessments.DwellInterval(0., 2.), assessments.DwellInterval(1., 3.)),
    (assessments.DwellInterval(2., 3.), assessments.DwellInterval(0., 1.)),
])
def test_invalid_dwell_selection_fails_before_callback(dwells):
    calls = []
    with pytest.raises(ValueError):
        _integrate(dwells=dwells, rupture_time=lambda s, t: calls.append((s, t)))
    assert not calls


def test_disjoint_and_touching_dwells_are_additive():
    all_damage = _integrate().damage
    adjacent = _integrate(dwells=(assessments.DwellInterval(0., 1.), assessments.DwellInterval(1., 3.)))
    assert adjacent.damage == pytest.approx(all_damage)
    separated = _integrate(dwells=(assessments.DwellInterval(0., 1.), assessments.DwellInterval(2., 3.)))
    assert separated.damage == pytest.approx((90. + 50.) / 100_000.)


def test_dense_other_channel_cannot_hide_sparse_source():
    result = _result((0., 3.), (100., 40.), np.linspace(0., 3., 31), np.full(31, 800.))
    with pytest.raises(ValueError, match="Source history"):
        _integrate(result, maximum_source_interval=.2)
    assert _integrate(result).maximum_stress_sample_interval == 3.


def test_declared_regular_sampling_tolerates_roundoff_but_keeps_measured_gap():
    times = np.linspace(0., 1., 11)
    history = _result(times, np.ones(11), times, np.full(11, 800.))
    assessment = _integrate(history, dwells=[assessments.DwellInterval(0., 1.)],
                            maximum_source_interval=.1)
    assert assessment.damage == pytest.approx(1.e-5)
    assert assessment.maximum_stress_sample_interval == float(np.diff(times).max())
    assert assessment.as_dict()["source_interval_relative_tolerance"] == 1.e-12


@pytest.mark.parametrize("times, limit", [
    ((0., .1 + 1.e-10), .1),
    ((1.e12, 1.e12 + .25), .1),
    ((0., 1.e-20), 1.e-21),
])
def test_sampling_allowance_cannot_hide_real_gaps_large_time_origins_or_tiny_scales(times, limit):
    history = _result(times, (1., 1.), times, (800., 800.))
    with pytest.raises(ValueError, match="Source history"):
        _integrate(history, dwells=[assessments.DwellInterval(*times)], maximum_source_interval=limit)


@pytest.mark.parametrize("value", [0., -1., np.nan, -np.inf, 1.j, [1.], 1.e-320])
def test_invalid_rupture_relation_reports_dwell_and_time(value):
    with pytest.raises(ValueError, match="Rupture relation failed.*time"):
        _integrate(rupture_time=lambda s, t: value)


def test_infinite_rupture_time_is_explicit_zero_rate_and_strict_json():
    assessment = _integrate(rupture_time=lambda s, t: np.inf)
    assert assessment.damage == 0.
    assert all(interval.damage_rate == (0., 0.) for interval in assessment.intervals)
    json.dumps(assessment.as_dict(), allow_nan=False)


def test_signed_stress_is_not_implicitly_absolutized():
    result = _result(stress=(-100., -80., -40.))
    with pytest.raises(ValueError, match="positive"):
        _integrate(result)
    assessment = _integrate(result, rupture_time=lambda s, t: 100_000. / abs(s),
                            rupture_source="explicit signed-to-magnitude synthetic relation")
    assert assessment.intervals[0].stress == (-100., -80.)


def test_nonpositive_absolute_temperature_is_rejected():
    with pytest.raises(ValueError, match="Absolute temperature"):
        _integrate(_result(temperature=(0., 800., 800.)))


def test_time_resolved_damage_composes_without_upgrading_trust():
    creep = _integrate()
    fatigue = FatigueAssessment((), .1, 10., source="synthetic")
    combined = assessments.creep_fatigue(fatigue=fatigue, creep=creep)
    assert combined.margin == pytest.approx(1. - .1 - creep.damage)
    result = _result()
    combined.attach(result)
    assert result.metadata["assessments"]["creep_fatigue"]["creep"]["method"] == "trapezoidal_damage_rate"
    json.dumps(combined.as_dict(), allow_nan=False)


def test_no_fatigue_cycles_remains_strict_json_not_infinite_life_claim():
    fatigue = assess_history([0., 0., 0.], BasquinCurve(1000., -.1))
    combined = assessments.creep_fatigue(fatigue=fatigue, creep=_integrate())
    payload = json.loads(json.dumps(combined.as_dict(), allow_nan=False))
    assert payload["fatigue"]["repeated_history_life"] is None
    assert payload["fatigue"]["life_status"] == "no_counted_damage"
    assert payload["creep_damage"] > 0.


def test_nonlinear_rate_time_refinement_against_analytic_integral():
    # Exact recorded stress s=100 exp(-t); rupture=1e5/s. Input samples
    # are the continuous reference, independent of any FE integrator.
    errors = []
    expected = 100. * (1. - np.exp(-3.)) / 1.e5
    for count in (16, 32, 64):
        time = np.linspace(0., 3., count + 1)
        history = _result(time, 100. * np.exp(-time), time, np.full(count + 1, 800.))
        errors.append(abs(_integrate(history).damage - expected))
    assert 3.9 < errors[0] / errors[1] < 4.1
    assert 3.9 < errors[1] / errors[2] < 4.1


def test_variable_temperature_rate_against_independent_integral():
    # Synthetic temperature-dependent rupture relation, NOT an alloy fit.
    # T=800+10t, rate=1e-3 exp((T-800)/100): exact integral is known.
    errors = []
    expected = .01 * np.expm1(.3)
    for count in (16, 32, 64):
        time = np.linspace(0., 3., count + 1)
        history = _result((0., 3.), (100., 100.), time, 800. + 10. * time)
        actual = _integrate(history, rupture_time=lambda s, t: 1000. * np.exp(-(t - 800.) / 100.))
        errors.append(abs(actual.damage - expected))
    assert 3.9 < errors[0] / errors[1] < 4.1
    assert 3.9 < errors[1] / errors[2] < 4.1


def test_late_domain_rejection_does_not_publish_partial_assessment():
    result = _result()
    metadata = dict(result.metadata)
    calls = []
    def bounded_relation(stress, temperature):
        calls.append(stress)
        if stress < 70.:
            raise ValueError("outside calibration domain")
        return 1000.
    with pytest.raises(ValueError, match="time 2.*outside calibration domain"):
        _integrate(result, rupture_time=bounded_relation)
    assert len(calls) == 3
    assert result.metadata == metadata
    assert not result.quantities


def test_positive_damage_underflow_is_not_reported_as_zero():
    result = _result((0., 1.e-100), (1., 1.), (0., 1.e-100), (800., 800.))
    with pytest.raises(ValueError, match="representable range"):
        _integrate(result, dwells=[assessments.DwellInterval(0., 1.e-100)],
                   rupture_time=lambda s, t: 1.e300)


def test_zero_repetitions_remains_explicit_zero_damage():
    assert _integrate(dwells=[assessments.DwellInterval(0., 3., repetitions=0.)]).damage == 0.
