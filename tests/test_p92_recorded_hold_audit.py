from copy import deepcopy

import numpy as np
import pytest

from tests.test_p92_measured_controls import data
from tools.p92_measured_controls import measured_path
from tools.p92_recorded_hold_audit import recorded_strain_segments


@pytest.mark.parametrize("protocol", ["RF", "SLR", "SLC"])
def test_only_strain_controls_receive_recorded_sensitivity(protocol):
    source = data(protocol)
    path = measured_path(source, cycles=1)
    original = deepcopy(path)
    result = recorded_strain_segments(path, source)
    for before, after in zip(original["segments"], result["segments"], strict=True):
        assert before["source_indices"] == after["source_indices"]
        np.testing.assert_array_equal(before["measured_stress"], after["measured_stress"])
        assert before["duration"] == after["duration"]
        assert before["stress_control"] == after["stress_control"]
        assert ("waveform" in after) != before["stress_control"]
    assert all("waveform" not in s for s in path["segments"])


def test_creep_response_is_not_inserted_as_a_command():
    source = data("SLC")
    path = measured_path(source, cycles=1)
    first = recorded_strain_segments(path, source)
    # Interior creep data can alter observations, not prescribed motion.
    source["strain"][(source["time_s"] > 200) & (source["time_s"] < 1900)] += .5
    changed = recorded_strain_segments(path, source)
    for a, b in zip(first["segments"], changed["segments"], strict=True):
        assert a["target"] == b["target"]
        if "waveform" in a:
            np.testing.assert_array_equal(a["waveform"]["deviation"], b["waveform"]["deviation"])


def test_identity_is_checked_before_attaching_observations():
    source = data()
    path = measured_path(source, cycles=1)
    source["sha256"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        recorded_strain_segments(path, source)
