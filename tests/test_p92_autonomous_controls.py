import numpy as np
import pytest

from tools import p92_recoverable_spectrum as kernel
from tools.p92_autonomous_controls import run
from tools.p92_two_rate_reference import Branch, Material
from tools.validate_p92_loading_path import nominal_path


def path(protocol="SLC", cycles=1):
    t, e = nominal_path("SLR" if protocol == "SLC" else protocol)
    segments = []
    for cycle in range(1, cycles+1):
        for i, (dt, left, right) in enumerate(zip(np.diff(t), e[:-1], e[1:], strict=True)):
            stress = protocol == "SLC" and i == 3
            segments.append(dict(cycle=cycle, duration=dt, target=None if stress else right,
                strain_hold=left == right and not stress, stress_control=stress,
                sample_times=np.array([.1*dt, .9*dt]), source_indices=[2*len(segments), 2*len(segments)+1],
                measured_stress=np.zeros(2), measured_strain=np.zeros(2)))
    return dict(protocol=protocol, cycles=cycles, segments=segments,
                initial_stress=0., initial_strain=0., source_sha256="synthetic")


def material(*, elastic=False):
    branch = Branch(1e6 if elastic else .1, 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    return kernel.Material(Material(10000., (branch, branch)),
                           (1e20, 1e20) if elastic else (20000., 40000.), (.5, 10.))


@pytest.mark.parametrize("protocol", ["RF", "SLR", "SLC"])
def test_elastic_limit_and_nominal_time(protocol):
    result = run(path(protocol), material(elastic=True), kernel)
    assert result["cycles"][0]["sampled_peak_MPa"] == pytest.approx(20., abs=1e-8)
    assert result["cycles"][0]["sampled_valley_MPa"] == pytest.approx(-20., abs=1e-8)
    expected = 368. if protocol == "RF" else 2168.
    assert result["switches"][-1]["end_s"] == pytest.approx(expected, abs=1e-8)


def test_post_creep_duration_uses_predicted_strain_and_subdivision_preserves_state():
    p, mat = path(cycles=2), material()
    result = run(p, mat, kernel)
    split = run(p, mat, kernel, subdivisions=2)
    for whole, half in zip(result["cycles"], split["cycles"], strict=True):
        np.testing.assert_allclose(whole["end_state"], half["end_state"], atol=2e-7, rtol=1e-6)
    after = result["switches"][4]
    assert after["strain_rate_per_s"] == .001
    assert after["duration_s"] == pytest.approx((.002-after["incoming_strain"])/.001)
    assert after["outgoing_strain"] == pytest.approx(.002, abs=1e-12)
    assert after["incoming_strain"] == result["segments"][3]["end_state"][7]
    assert result["cycles"][1]["end_state"][3] >= result["cycles"][0]["end_state"][3]


def test_creep_overshoot_is_not_reversed_into_a_completed_cycle():
    class Overshoot:
        STATE_LAYOUT = kernel.STATE_LAYOUT
        MODEL_ID = kernel.MODEL_ID
        stored_energy = staticmethod(kernel.stored_energy)

        @staticmethod
        def integrate(state, duration, rate, mat, **kwargs):
            sol = kernel.integrate(state, duration, rate, mat, **kwargs)
            if kwargs["stress_control"]:
                sol.y[7, -1] = .003
            return sol

    with pytest.raises(ValueError, match="exceeds positive target"):
        run(path(), material(), Overshoot)


def test_bad_subdivision_rejected():
    with pytest.raises(ValueError):
        run(path(), material(), kernel, subdivisions=0)
