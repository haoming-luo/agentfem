from copy import deepcopy

import numpy as np
import pytest

from tools.p92_two_rate_reference import Branch, Material
from tools.p92_two_rate_controls import replay


def fixture():
    branch = Branch(1e6, 1e-4, 3., (20000., 5000.), (30., 60.), (.02, .0001))
    path = dict(initial_stress=0., initial_strain=0., source_sha256="synthetic", segments=[
        dict(duration=2., target=.001, stress_control=False, cycle=1,
             sample_times=np.array([1., 2.]), measured_stress=np.array([75., 150.]),
             measured_strain=np.array([.0005, .001]), source_indices=[1, 2]),
        dict(duration=2., target=None, stress_control=True, cycle=1,
             sample_times=np.array([1., 2.]), measured_stress=np.array([150., 150.]),
             measured_strain=np.array([.001, .001]), source_indices=[3, 4])])
    return path, Material(150000., (branch, branch))


def test_control_state_layout_and_observation_weights():
    path, material = fixture()
    residual, report = replay(path, material)
    np.testing.assert_allclose(residual, 0., atol=1e-12)
    assert report["state_layout"][7] == "strain"
    assert report["segments"][-1]["end_state"][7] == pytest.approx(.001)
    assert abs(report["mechanical_balance_MPa"]) < 1e-8
    changed = deepcopy(path)
    changed["segments"][1]["measured_strain"] += .0002
    altered, result = replay(changed, material)
    assert altered[-1] == pytest.approx(-.1/np.sqrt(24))
    # Observations must not alter the simulated hold trajectory.
    np.testing.assert_array_equal(result["segments"][-1]["end_state"],
                                  report["segments"][-1]["end_state"])


def test_waveform_and_forbidden_hold_input():
    path, material = fixture()
    path["segments"][0]["waveform"] = dict(times=[0., 1., 2.], deviation=[0., .0001, 0.])
    _, result = replay(path, material)
    assert result["segments"][0]["sampled_stress_MPa"][0] == pytest.approx(90.)
    assert result["segments"][0]["sampled_stress_MPa"][1] == pytest.approx(150.)
    path["segments"][1]["waveform"] = path["segments"][0]["waveform"]
    with pytest.raises(ValueError, match="Stress hold"):
        replay(path, material)
