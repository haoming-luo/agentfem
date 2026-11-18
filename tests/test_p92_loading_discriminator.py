# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from tools import p92_loading_discriminator as module
from tools.p92_published_recovery_reference import published_parameters


def inputs():
    data = {"protocol": "RF", "sha256": "source", "cycle": np.array([1, 1, 1, 2, 2, 2])}
    calibration = {"source_sha256": "source", "fit": {"success": True},
                   "parameters": asdict(published_parameters(600)), "initial_state": [0.] * 5}
    return data, calibration


def test_same_initial_state_no_refit_and_peak_error_decomposition(monkeypatch):
    data, calibration = inputs()
    histories = {"strain": SimpleNamespace(abscissa=np.arange(6.), values=np.zeros(6)),
                 "stress_Pa": SimpleNamespace(values=np.array([0., 2., 1., 0., 3., 1.]) * 1e6)}
    monkeypatch.setattr(module, "replay_window", lambda *a, **kw: {"histories": histories})
    calls = []
    def replay(t, e, p, initial, **kwargs):
        calls.append((p, initial.copy(), kwargs))
        states = np.zeros((6, 5))
        states[:, 0] = [0., 2.5, 1., 0., 4., 1.]
        return states
    monkeypatch.setattr(module, "replay", replay)
    monkeypatch.setattr(module, "nominal_cycles", lambda *a, **kw: {
        "failure": None, "cycles": [{"cycle": 1, "sampled_peak_MPa": 3.},
                                     {"cycle": 2, "sampled_peak_MPa": 5.}]})
    report = module.compare_loading(data, calibration, cycles=2, subdivisions=8)
    assert report["BE_subdivisions"] == 8
    assert calls[1][2]["subdivisions"] == 8
    assert calls[0][0] == calls[1][0]
    for _, initial, _ in calls:
        np.testing.assert_array_equal(initial, calibration["initial_state"])
    for item in report["cycles"]:
        assert item["nominal_response_peak_error_MPa"] == pytest.approx(
            item["recorded_response_peak_error_MPa"] + item["loading_choice_peak_difference_MPa"])
    assert report["missing_recorded_cycles_in_source_span"] == 0
    assert not report["engineering_qualified"] and not report["life_qualified"]


@pytest.mark.parametrize("problem", ["protocol", "source", "fit"])
def test_reject_incompatible_source_or_unconverged_fit(problem):
    data, calibration = inputs()
    if problem == "protocol":
        data["protocol"] = "SLR"
    elif problem == "source":
        calibration["source_sha256"] = "other"
    else:
        calibration["fit"]["success"] = False
    with pytest.raises(ValueError):
        module.compare_loading(data, calibration)
