"""Synthetic analytic verification, not alloy calibration."""

from dataclasses import FrozenInstanceError, replace
import json
import numpy as np
import pytest
from agentfem import assessments
from test_creep_damage_history import _result, _integrate


def curve(**options):
    defaults = dict(
        stresses=(40.0, 100.0),
        rupture_times=(2500.0, 1000.0),
        temperature=800.0,
        stress_unit="MPa",
        time_unit="s",
        stress_measure="signed_axial",
        material_condition="synthetic",
        failure_definition="synthetic rupture",
        source="analytic t=100000/sigma",
    )
    defaults.update(options)
    return assessments.TabulatedRuptureCurve(**defaults)


def test_nodes_and_power_law_interpolation():
    relation = curve()
    assert relation(40.0, 800.0) == 2500.0
    assert relation(100.0, 800.0) == 1000.0
    assert relation(60.0, 800.0) == pytest.approx(100000.0 / 60.0)
    assert relation(np.sqrt(4000.0), 800.0) == pytest.approx(np.sqrt(2500000.0))
    assert json.loads(json.dumps(relation.as_dict()))["interpolation"] == "log_log"


def test_input_arrays_are_frozen():
    stresses, times = [40.0, 100.0], [2500.0, 1000.0]
    relation = curve(stresses=stresses, rupture_times=times)
    stresses[0], times[0] = 1.0, 1.0
    assert relation(40.0, 800.0) == 2500.0
    with pytest.raises(FrozenInstanceError):
        relation.temperature = 900.0


@pytest.mark.parametrize(
    "changes",
    [
        dict(stresses=(100.0, 40.0)),
        dict(stresses=(40.0, 40.0)),
        dict(stresses=(0.0, 100.0)),
        dict(stresses=(True, 100.0)),
        dict(rupture_times=(1000.0, 2500.0)),
        dict(rupture_times=(2500.0,)),
        dict(rupture_times=(np.inf, 1000.0)),
        dict(rupture_times=(2500.0 + 0j, 1000.0)),
        dict(temperature=0.0),
        dict(temperature=True),
        dict(source=""),
        dict(material_condition=""),
        dict(stress_unit=""),
        dict(time_unit=""),
        dict(stress_measure=""),
        dict(failure_definition=""),
    ],
)
def test_invalid_curve_rejected(changes):
    with pytest.raises(ValueError):
        curve(**changes)


@pytest.mark.parametrize(
    "stress,temperature",
    [
        (39.0, 800.0),
        (101.0, 800.0),
        (-40.0, 800.0),
        (True, 800.0),
        (40.0 + 0j, 800.0),
        (60.0, 801.0),
        (60.0, np.nan),
    ],
)
def test_no_extrapolation_or_temperature_interpolation(stress, temperature):
    with pytest.raises(ValueError):
        curve()(stress, temperature)


def test_composition_matches_analytic_damage_and_keeps_relation():
    relation = curve()
    assessment = _integrate(rupture_time=relation, rupture_source=relation.source)
    assert assessment.damage == pytest.approx(210.0 / 100000.0)
    record = assessment.as_dict()
    assert record["rupture_relation"] == relation.as_dict()
    record["rupture_relation"]["stresses"][0] = -1
    assert assessment.as_dict()["rupture_relation"]["stresses"][0] == 40.0


@pytest.mark.parametrize(
    "changes",
    [dict(stress_unit="Pa"), dict(time_unit="h"), dict(stress_measure="von_mises")],
)
def test_assessment_checks_relation_semantics(changes):
    with pytest.raises(ValueError, match="differs"):
        _integrate(rupture_time=curve(), **changes)


def test_explicit_unit_conversion_preserves_damage():
    baseline = _integrate(rupture_time=curve())
    result = _result()
    for name, history in result.histories.items():
        result.histories[name] = replace(
            history,
            abscissa=history.abscissa / 3600.0,
            abscissa_unit="h",
            values=history.values * (1e6 if name == "stress" else 1),
            unit="Pa" if name == "stress" else "K",
        )
    converted = curve(
        stresses=(40e6, 100e6),
        rupture_times=(2500.0 / 3600.0, 1000.0 / 3600.0),
        stress_unit="Pa",
        time_unit="h",
    )
    actual = _integrate(
        result,
        rupture_time=converted,
        stress_unit="Pa",
        time_unit="h",
        dwells=(assessments.DwellInterval(0.0, 3.0 / 3600.0),),
    )
    assert actual.damage == pytest.approx(baseline.damage)


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_boolean_callback_is_not_a_rupture_lifetime(value):
    with pytest.raises(ValueError, match="real scalar"):
        _integrate(rupture_time=lambda stress, temperature: value)
