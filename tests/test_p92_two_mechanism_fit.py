from dataclasses import asdict

import numpy as np
import pytest

from tools.p92_published_recovery_reference import published_parameters, isotropic_strength_lower_bound
from tools.p92_two_mechanism_fit import decode, observations, residual_path, fit


def source(protocol):
    t = np.linspace(0., 2200., 22001)
    return dict(protocol=protocol, time_s=t, cycle=np.ones(len(t)),
                strain=np.zeros(len(t)), stress_Pa=np.zeros(len(t)), sha256="synthetic")


def test_training_selection_excludes_all_later_cycles():
    data = source("SLR")
    selected = observations(data)
    extended = dict(data)
    for key in ("time_s", "cycle", "strain", "stress_Pa"):
        values = {"time_s": data[key][-1]+np.arange(1, 101),
                  "cycle": np.full(100, 2.), "strain": np.full(100, 100.),
                  "stress_Pa": np.full(100, 1e20)}[key]
        extended[key] = np.r_[data[key], values]
    other = observations(extended)
    for a, b in zip(selected["segments"], other["segments"]):
        assert a["source_indices"] == b["source_indices"]
        np.testing.assert_array_equal(a["measured_stress"], b["measured_stress"])


def test_slc_uses_stress_control_and_both_observation_channels():
    path = observations(source("SLC"))
    assert [s["stress_control"] for s in path["segments"]] == [False, False, False, True]
    assert sum(s["duration"] for s in path["segments"]) == 1984.
    m = decode([-3., -1., -2., -4.5, -1., 0.], published_parameters(600))
    r = residual_path(path, m)
    assert len(r) == sum(len(s["sample_times"])*(2 if s["stress_control"] else 1)
                         for s in path["segments"])
    assert np.isfinite(r).all()


def test_decoding_preserves_strength_and_shared_modulus():
    p = published_parameters(600)
    a, b = decode([-3., -1., -2., -4.5, -1., 0.], p)
    assert a.parameters.E == b.parameters.E
    assert isotropic_strength_lower_bound(a.parameters) == pytest.approx(.1)
    assert isotropic_strength_lower_bound(b.parameters) == pytest.approx(.1)
    with pytest.raises(ValueError):
        decode([1., 2.], p)


def test_budget_exhaustion_does_not_publish_later_predictions(monkeypatch):
    from types import SimpleNamespace
    from tools import p92_two_mechanism_fit as module
    monkeypatch.setattr(module, "least_squares", lambda *a, **k: SimpleNamespace(
        x=np.array([-3., -1., -2., -4.5, -1., 0.]), success=False,
        message="budget exhausted", nfev=1, cost=1., active_mask=np.zeros(6), jac=np.eye(6)))
    result = fit(dict(parameters=asdict(published_parameters(600))),
                 [source(p) for p in ("RF", "SLR", "SLC")], max_nfev=1)
    assert "frozen_predictions" not in result
    assert not result["fit"]["success"] and not result["engineering_qualified"]


def test_gapped_training_is_rejected():
    data = source("SLR")
    keep = (data["time_s"] < 200.) | (data["time_s"] > 220.)
    for key in ("time_s", "cycle", "strain", "stress_Pa"):
        data[key] = data[key][keep]
    with pytest.raises(ValueError, match="contiguous"):
        observations(data)


def test_success_freezes_identical_parameters_for_all_predictions(monkeypatch):
    from types import SimpleNamespace
    from tools import p92_two_mechanism_fit as module
    z = np.array([-3., -1., -2., -4.5, -1., 0.])
    monkeypatch.setattr(module, "least_squares", lambda *a, **k: SimpleNamespace(
        x=z, success=True, message="synthetic optimizer success", nfev=1,
        cost=1., active_mask=np.zeros(6), jac=np.eye(6)))
    monkeypatch.setattr(module, "residual_path", lambda *a, **k: np.ones(3))
    calls = []
    def prediction(mechanisms, **kwargs):
        calls.append((tuple(asdict(m) for m in mechanisms), kwargs))
        return {"cycles": []}
    monkeypatch.setattr(module, "run", prediction)
    monkeypatch.setattr(module, "verify_frozen", lambda report: [])
    result = fit(dict(parameters=asdict(published_parameters(600))),
                 [source(p) for p in ("RF", "SLR", "SLC")])
    assert len(calls) == 3
    assert calls[0][0] == calls[1][0] == calls[2][0]
    assert all(c[1]["cycles"] == 10 for c in calls)
    assert len(result["parameter_sha256"]) == 64
    assert not result["engineering_qualified"]


def test_frozen_verification_rejects_changed_parameters():
    from tools.p92_two_mechanism_fit import verify_frozen
    with pytest.raises(ValueError, match="intact"):
        verify_frozen(dict(mechanisms=[], parameter_sha256="wrong", fit={"success": True}))
