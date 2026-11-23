from dataclasses import replace

import numpy as np
import pytest

from tools import p92_shared_drag_reference as shared


def material():
    return shared.Material(10000., 100., (.001, .002), (2., 4.),
        (1000., 2000., 3000.), (10., 20., 30.), (20., 30., 40.),
        (100., 200., 300.), (.01, .02, .03), 20., 1., 2., 10., .0001,
        (1/20000., 1/40000.), (.5, 10.))


def test_unverified_transcription_and_exact_zero_load_drag_decay():
    mat = shared.unverified_table2_transcription()
    assert mat.modulus == 132000. and mat.initial_drag == 327.
    assert mat.exponents == (12.1, 278.)
    old = np.zeros(12)
    a = shared.integrate(old, 1800., 0., mat)
    assert a.y[5, -1] == pytest.approx(-mat.drag_static_recovery*1800., abs=1e-12)
    np.testing.assert_allclose(a.y[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], -1], 0., atol=1e-12)
    np.testing.assert_array_equal(old, np.zeros(12))


@pytest.mark.parametrize("stress_control", [True, False])
def test_shared_power_identity_and_signed_symmetry(stress_control):
    mat = material()
    old = np.array([70., 1., -2., .02, 3., -.1, -.003, .01, 0., 0., -.001, .002])
    rate = shared.rates(old, .003, mat, stress_control=stress_control)
    energy_rate = old[0]*rate[0]/mat.modulus+sum(
        old[i]*rate[i]/h for i, h in zip(shared.X_INDICES, mat.hardening))
    energy_rate += sum(a*v/c for a, v, c in zip(old[10:], rate[10:], mat.compliances))
    assert rate[8] == pytest.approx(energy_rate+rate[9], abs=1e-12)
    assert rate[9] >= 0
    sign = np.ones(12)
    sign[[0, 1, 2, 4, 6, 7, 10, 11]] = -1
    np.testing.assert_allclose(shared.rates(old*sign, -.003, mat, stress_control=stress_control), rate*sign)


def test_positive_drag_law_matches_direct_scalar_equation():
    mat = material()
    old = np.array([70., 1., -2., .02, 3., -.1, -.003, .01, 0., 0., 0., 0.])
    rate = shared.rates(old, 0., mat)
    drag = mat.initial_drag*np.exp(old[5])
    gamma = mat.drag_dynamic_final+(mat.drag_dynamic_initial-mat.drag_dynamic_final)*np.exp(-mat.drag_dynamic_rate*old[3])
    expected = mat.drag_hardening*rate[3]-(gamma*rate[3]+mat.drag_static_recovery)*drag
    assert drag*rate[5] == pytest.approx(expected, abs=1e-12)
    effective = old[0]-sum(old[list(shared.X_INDICES)])
    assert rate[6] == pytest.approx(sum(a*(effective/drag)**n for a, n in zip(mat.flow_coefficients, mat.exponents)))


def test_split_reversal_and_independent_integration():
    mat = material()
    current = shared.integrate(np.zeros(12), 2., .02, mat).y[:, -1]
    before = current.copy()
    full = shared.integrate(current, 4., -.02, mat)
    bdf = shared.integrate(current, 4., -.02, mat, method="BDF", rtol=1e-10)
    part = shared.integrate(current, 1., -.02, mat)
    rest = shared.integrate(part.y[:, -1], 3., -.02, mat)
    np.testing.assert_allclose(full.y[:, -1], bdf.y[:, -1], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(full.y[:, -1], rest.y[:, -1], rtol=1e-7, atol=1e-7)
    np.testing.assert_array_equal(current, before)


def test_wrong_state_and_disabled_component_fail_without_mutation():
    old = np.zeros(12)
    old[10] = .001
    with pytest.raises(ValueError, match="Disabled"):
        shared.integrate(old, 1., 0., shared.unverified_table2_transcription())
    with pytest.raises(ValueError):
        replace(material(), initial_drag=0.)
    old[3], old[6] = .001, .01
    with pytest.raises(ValueError, match="state"):
        shared.integrate(old, 1., 0., material())


def test_replay_keeps_new_internal_identity_and_checks_log_drag():
    from tools.p92_two_rate_controls import replay
    from tools.p92_two_rate_frozen import numerical_difference
    from copy import deepcopy
    path = dict(initial_stress=0., initial_strain=0., source_sha256="synthetic", protocol="RF",
        segments=[dict(duration=.1, target=.0001, stress_control=False, cycle=1,
          sample_times=[.05, .1], measured_stress=[.5, 1.], measured_strain=[.00005, .0001], source_indices=[1, 2])])
    _, report = replay(path, material(), kernel=shared)
    assert report["state_layout"][5] == "log_relative_drag"
    assert report["model_id"] == shared.MODEL_ID
    changed = deepcopy(report)
    changed["segments"][0]["end_state"][5] += 1e-6
    options = dict(extra_strain_indices=(10, 11), stress_state_indices=(0, 1, 2, 4), dimensionless_indices=(5,))
    assert not numerical_difference(report, changed, **options)["passed"]
    with pytest.raises(ValueError, match="cover each"):
        numerical_difference(report, changed)


def test_joint_coordinates_preserve_positive_pairs_and_zero_limit():
    from dataclasses import asdict
    from tools.p92_shared_drag_joint import decode
    mat = material()
    old = np.array([70., 1., -2., .02, 3., -.1, -.003, .01, 0., 0., 0., 0.])
    np.testing.assert_array_equal(shared.rates(old, .003, mat),
                                 shared.rates(old, .003, decode(asdict(mat), np.zeros(14))))
    z = np.zeros(14)
    z[10] = .2
    scaled = decode(asdict(mat), z)
    assert scaled.dynamic_final[0]/scaled.dynamic_initial[0] == pytest.approx(
        mat.dynamic_final[0]/mat.dynamic_initial[0])
    with pytest.raises(ValueError, match="fourteen"):
        decode(asdict(mat), np.zeros(13))


def test_ideal_elastic_stress_step_has_exact_work_without_history_reset():
    from tools.p92_shared_drag_benchmark import elastic_stress_step
    mat = material()
    old = np.array([70., 1., -2., .02, 3., -.1, -.003, .01, 0., 0., 0., 0.])
    before = old.copy()
    new = elastic_stress_step(old, -25., mat)
    assert new[8]-old[8] == pytest.approx(shared.stored_energy(new, mat)-shared.stored_energy(old, mat))
    np.testing.assert_array_equal(new[[1, 2, 3, 4, 5, 6, 9, 10, 11]], old[[1, 2, 3, 4, 5, 6, 9, 10, 11]])
    np.testing.assert_array_equal(old, before)


def test_frozen_report_retains_rejected_paths_and_checks_source_first(monkeypatch):
    from dataclasses import asdict
    from pathlib import Path
    from tools import p92_recoverable_frozen as frozen
    fit = dict(training_cycles=[1, 2, 3], model_id=shared.MODEL_ID,
        fit=dict(success=True, parameters=asdict(material())),
        source_sha256={name: name for name in ("RF", "SLR", "SLC")})
    monkeypatch.setattr(frozen, "read_source", lambda path, protocol: dict(sha256=protocol))
    monkeypatch.setattr(frozen, "measured_path", lambda source, cycles: source)
    monkeypatch.setattr(frozen, "attach_waveforms", lambda path, source: path)
    seen = []
    def rejected(path, mat, **kwargs):
        seen.append(path["sha256"])
        raise ValueError("Predicted creep leaves the declared subsequent ramp direction")
    monkeypatch.setattr(frozen, "replay", rejected)
    report = frozen.evaluate(fit, Path("unused"))
    assert seen == ["RF", "SLR", "SLC"]
    assert not report["engineering_qualified"]
    for result in report["protocols"].values():
        assert result["response_screen"]["status"] == "failed"
        assert result["failure"]["stage"] == "prediction"
        assert not result["numerical"]["completed"]
    fit["source_sha256"]["RF"] = "changed"
    with pytest.raises(ValueError, match="Source identity"):
        frozen.evaluate(fit, Path("unused"))
