# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import json

import numpy as np
import pytest

from agentfem import assessments, constitutive


def _curve():
    return constitutive.CoffinMansonBasquinCurve(
        young=200_000., fatigue_strength_coefficient=1_000.,
        fatigue_strength_exponent=-0.1, fatigue_ductility_coefficient=0.5,
        fatigue_ductility_exponent=-0.6, calibrated_cycles=(1., 1.e8),
        stress_unit="MPa", source="synthetic software verification, not material calibration",
    )


def _assess(values, curve=None, **kwargs):
    arguments = dict(strain_measure="mechanical_total", source="signed axial mechanical strain",
                     mean_stress_correction="none")
    arguments.update(kwargs)
    return constitutive.assess_strain_history(values, _curve() if curve is None else curve, **arguments)


def test_strain_life_inverse_matches_equal_exponent_closed_form():
    curve = replace(_curve(), fatigue_strength_exponent=-0.5, fatigue_ductility_exponent=-0.5)
    for amplitude in (0.1, 0.02, 0.001):
        # Same exponent makes the sum invertible independently; N is full
        # cycles, not the 2N reversals appearing in the constitutive equation.
        expected = 0.5 * (amplitude / (1000. / 200000. + 0.5)) ** -2
        assert curve.cycles_to_failure(amplitude) == pytest.approx(expected, rel=2.e-13)


def test_strain_life_roundtrip_domain_and_stress_units():
    curve = _curve()
    pascals = replace(curve, young=curve.young * 1.e6,
                     fatigue_strength_coefficient=1.e9, stress_unit="Pa")
    for cycles in np.geomspace(1., 1.e8, 40):
        amplitude = curve.strain_amplitude(cycles)
        assert curve.cycles_to_failure(amplitude) == pytest.approx(cycles, rel=1.e-13)
        assert pascals.strain_amplitude(cycles) == pytest.approx(amplitude, rel=1.e-14)
    for amplitude in (curve.strain_amplitude(1.) * 1.001, curve.strain_amplitude(1.e8) * .999):
        with pytest.raises(ValueError, match="outside"):
            curve.cycles_to_failure(amplitude)
    for cycles in (0., np.nan, np.inf, .99, 1.01e8):
        with pytest.raises(ValueError):
            curve.strain_amplitude(cycles)


def test_strain_life_counts_full_and_half_cycles_and_preserves_source():
    curve = _curve()
    amplitude = curve.strain_amplitude(1_000.)
    history = np.array([-amplitude, amplitude, -amplitude])
    before = history.copy()
    assessment = _assess(history, curve)
    assert sum(c.count for c in assessment.cycles) == 1.
    assert assessment.damage == pytest.approx(.001)
    assert assessment.repeated_history_life == pytest.approx(1000.)
    assert _assess(history[:2], curve).damage == pytest.approx(.0005)
    assert _assess(history + 0.02, curve).damage == pytest.approx(assessment.damage)
    np.testing.assert_array_equal(history, before)
    data = json.loads(json.dumps(assessment.as_dict(), allow_nan=False))
    assert data["curve"]["life_unit"] == "full_cycles"
    assert data["curve"]["source"] == curve.source
    assert data["mean_stress_correction"] == "none"
    assert data["strain_measure"] == "mechanical_total"


def test_constant_strain_has_no_counted_damage_not_certified_infinite_life():
    assessment = _assess([0.02, 0.02, 0.02])
    assert assessment.damage == 0.
    assert assessment.cycles == ()
    payload = json.loads(json.dumps(assessment.as_dict(), allow_nan=False))
    assert payload["repeated_history_life"] is None
    assert payload["life_status"] == "no_counted_damage"


@pytest.mark.parametrize("measure", ["PEEQ", "CEEQ", "plastic", "total", "thermal", "mises"])
def test_ambiguous_or_accumulated_strain_measure_is_rejected(measure):
    with pytest.raises(ValueError, match="mechanical_total"):
        _assess([0., .001, .002], strain_measure=measure)


@pytest.mark.parametrize("changes", [dict(young=0.), dict(fatigue_ductility_coefficient=-1.),
                                    dict(fatigue_strength_exponent=0.), dict(source=""),
                                    dict(stress_unit=""), dict(calibrated_cycles=(10., 1.)),
                                    dict(calibrated_cycles=(1., np.inf))])
def test_invalid_strain_life_calibration_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(_curve(), **changes)


def test_invalid_histories_and_unsupported_mean_correction():
    for values in ([1.], [0., np.inf], [[0., .001], [.001, 0.]]):
        with pytest.raises(ValueError):
            _assess(values)
    with pytest.raises(ValueError, match="mean_stress_correction"):
        _assess([-.01, .01], mean_stress_correction="Morrow")
    with pytest.raises(ValueError, match="source"):
        _assess([-.01, .01], source="")


def test_strain_life_reuses_declared_creep_fatigue_interaction():
    fatigue = _assess([-.01, .01, -.01])
    creep = assessments.creep_time_fraction([
        assessments.CreepDamageBlock(duration=10., rupture_time=100., source="synthetic rupture data")
    ])
    result = assessments.creep_fatigue(fatigue=fatigue, creep=creep)
    assert result.creep_damage == pytest.approx(.1)
    assert result.margin == pytest.approx(.9 - fatigue.damage)
    payload = json.loads(json.dumps(result.as_dict(), allow_nan=False))
    assert payload["fatigue"]["kind"] == "strain_life_fatigue_assessment"
    assert payload["interaction"]["source"].startswith("AgentFEM transparent reference")
