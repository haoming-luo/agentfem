"""Existing rate-dependent structural consumers share the history clock guard."""
import numpy as np
import pytest

from agentfem import histories, units
from agentfem.constitutive.viscoelasticity import WLFShift
from test_coupled_creep_plasticity import _structure
from test_viscoelasticity import _global_viscoelastic_relaxation_patch


def assert_same_state(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same_state(actual[key], expected[key])
    else:
        np.testing.assert_array_equal(actual, expected)


def build(family, source_unit, target_unit):
    if family != "viscoelastic":
        return _structure(thermal=True, pure_creep=family == "creep",
                          temperature_time_unit=source_unit, time_unit=target_unit)
    thermal = histories.temperature(293.15, coordinate_unit=source_unit)
    thermal.record(0.).record(2., 303.15)
    return _global_viscoelastic_relaxation_patch(
        shift=WLFShift(reference_temperature=293.15, c1=8., c2=80.),
        temperature=thermal, model_units=None if target_unit is None else
        units.consistent(length="mm", mass="tonne", time=target_unit))


@pytest.mark.parametrize("family", ["creep", "joint", "viscoelastic"])
@pytest.mark.parametrize("target", [None, "s"])
def test_clock_mismatch_rejected_at_construction(family, target):
    with pytest.raises(ValueError, match="coordinate_unit.*time_unit"):
        build(family, "h", target)


@pytest.mark.parametrize("family", ["creep", "joint", "viscoelastic"])
def test_matching_clock_preserves_numerics(family):
    legacy = build(family, None, "s")
    declared = build(family, "s", "s")
    legacy.solve()
    declared.solve()
    np.testing.assert_array_equal(declared.solution.x.array, legacy.solution.x.array)
    assert_same_state(declared.state.snapshot(), legacy.state.snapshot())


@pytest.mark.parametrize("family", ["creep", "joint", "viscoelastic"])
def test_mutated_live_history_rejected_before_loading(family):
    step = build(family, "s", "s")
    before = step.solution.x.array.copy()
    state = step.state.snapshot()
    step.temperature.coordinate_unit = "h"
    with pytest.raises(ValueError, match="coordinate_unit.*time_unit"):
        step._apply_loading(1.)
    np.testing.assert_array_equal(step.solution.x.array, before)
    assert_same_state(step.state.snapshot(), state)
