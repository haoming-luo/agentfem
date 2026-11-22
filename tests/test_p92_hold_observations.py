import numpy as np
import pytest

from tools.p92_hold_observations import summarize


def test_hold_secants_keep_actual_samples_and_explicit_layout():
    times = np.array([1., 3., 90., 100.])
    segment = dict(duration=100., sample_times=times, cycle=1, stress_control=True,
        measured_stress=100.+0*times, measured_strain=times*1e-5,
        source_indices=[1, 3, 90, 100],
        tail=dict(times=np.array([81., 99.]), stress=np.array([100., 100.]),
                  strain=np.array([.00081, .00099]), source_indices=[81, 99]))

    def evaluate(t):
        result = np.zeros((10, len(t)))
        result[0] = 100.
        result[7] = t*2e-5
        result[9] = 999.  # Work/dissipation must not be mistaken for strain.
        return result

    result = summarize(segment, evaluate, index=3, strain_index=7)
    field = result["fields"]["strain"]
    assert field["early_secant_observed_per_s"] == pytest.approx(1e-5)
    assert field["late_secant_predicted_per_s"] == pytest.approx(2e-5)
    assert field["last_predicted"] == pytest.approx(.002)
    assert field["first_predicted"] == pytest.approx(2e-5)
    assert field["first_observed"] == pytest.approx(1e-5)
    assert field["last_residual"] == pytest.approx(.001)
    assert field["anchored_shape_last_residual"] == pytest.approx(.00099)
    assert result["first_observation_time_s"] == 1.
    assert "not extrapolated" in result["anchor_scope"]
    assert result["late_secant_source_indices"] == [81, 99]


def test_missing_late_samples_do_not_fabricate_a_zero_secant():
    segment = dict(duration=100., sample_times=np.array([1., 2.]), cycle=1,
        stress_control=False, measured_stress=np.array([1., 2.]), measured_strain=np.zeros(2))
    result = summarize(segment, lambda t: np.zeros((10, len(t))), index=0, strain_index=7)
    assert "late_secant_predicted_per_s" not in result["fields"]["stress_MPa"]


@pytest.mark.parametrize("times", [[], [1., 1.], [2., 1.], [-1., 2.], [1., 101.], [1., np.nan]])
def test_invalid_observation_times_are_not_silently_used(times):
    segment = dict(duration=100., sample_times=times, cycle=1, stress_control=False,
                   measured_stress=np.zeros(len(times)), measured_strain=np.zeros(len(times)))
    with pytest.raises(ValueError, match="ordered observations"):
        summarize(segment, lambda t: np.zeros((10, len(t))), index=0, strain_index=7)


def test_shape_diagnostic_does_not_hide_constant_prediction_bias():
    segment = dict(duration=100., sample_times=[1., 2., 90., 99.], cycle=1,
                   stress_control=False, measured_stress=[100., 90., 80., 70.],
                   measured_strain=np.zeros(4))
    values = np.zeros((10, 4))
    values[0] = np.array(segment["measured_stress"])+20.

    def evaluate(t):
        return values[:, np.searchsorted(segment["sample_times"], t)]

    result = summarize(segment, evaluate, index=0, strain_index=7)["fields"]["stress_MPa"]
    assert result["rms"] == result["first_residual"] == result["last_residual"] == 20.
    assert result["anchored_shape_rms"] == result["anchored_shape_last_residual"] == 0.


def test_complex_tail_times_are_rejected_before_prediction():
    segment = dict(duration=100., sample_times=[1., 2.], cycle=1, stress_control=False,
                   measured_stress=[1., 2.], measured_strain=[0., 0.],
                   tail=dict(times=[90.+1j, 99.+1j], stress=[0., 0.], strain=[0., 0.], source_indices=[]))
    with pytest.raises(ValueError, match="Hold tail"):
        summarize(segment, lambda t: np.zeros((10, len(t))), index=0, strain_index=7)
