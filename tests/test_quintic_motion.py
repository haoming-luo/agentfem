import numpy as np
import pytest

from agentfem import amplitudes, boundary_models


def test_quintic_reference_and_derivatives():
    curve = amplitudes.quintic_step(0.1, 1.0, start_time=0.12, end_time=0.9)
    for s in np.linspace(0, 1, 41):
        t = 0.12 + 0.78 * s
        assert curve(t) == pytest.approx(
            0.1 + 0.9 * (10 * s**3 - 15 * s**4 + 6 * s**5), abs=1e-14
        )
    for t in (-1, 0.12, 0.9, 2):
        assert curve.velocity(t) == curve.acceleration(t) == 0
    for t in (0.2, 0.4, 0.6, 0.8):
        h = 1e-6
        assert curve.velocity(t) == pytest.approx(
            (curve(t + h) - curve(t - h)) / (2 * h), rel=1e-8
        )
        assert curve.acceleration(t) == pytest.approx(
            (curve.velocity(t + h) - curve.velocity(t - h)) / (2 * h), rel=1e-8
        )
    assert amplitudes.smooth_step()(0.25) != pytest.approx(
        amplitudes.quintic_step()(0.25)
    )


def test_rigid_motion_uses_physical_amplitude_and_fingerprints_it():
    motion = boundary_models.prescribed_rigid_motion(translation=(0, 0, 6))
    schedule = boundary_models.prescribed_rigid_motion_schedule(
        motion, end_time=0.1, amplitude=amplitudes.quintic_step(end_time=0.1)
    )
    assert schedule.factor_at(0.025) == pytest.approx(0.103515625)
    np.testing.assert_allclose(
        schedule.generalized_coordinate_at(0.025), [0, 0, 6 * 0.103515625, 0, 0, 0]
    )
    assert schedule.factor_at(0.2) == 1
    assert schedule.summary()["amplitude"]["kind"] == "quintic_step"
    linear = boundary_models.prescribed_rigid_motion_schedule(motion, end_time=0.1)
    assert linear.summary() != schedule.summary()
    with pytest.raises(ValueError, match="zero to one"):
        boundary_models.prescribed_rigid_motion_schedule(
            motion, end_time=0.1, amplitude=amplitudes.quintic_step(end_time=1)
        )


@pytest.mark.parametrize(
    "parameter", ["start_value", "end_value", "start_time", "end_time"]
)
def test_quintic_nonfinite_inputs_rejected(parameter):
    with pytest.raises(ValueError):
        amplitudes.quintic_step(**{parameter: np.nan})


def test_quintic_moving_cylinder_material_bond_damping_restart(tmp_path):
    from test_response_hex_explicit import make_step

    options = dict(contact="cylinder", bond=True, viscosity=True, quintic=True)
    reference = make_step(**options)
    reference.run()
    partial = make_step(**options)
    partial.run(until_step=4)
    path = partial.save_checkpoint(tmp_path / "quintic")
    resumed = make_step(**options)
    resumed.load_checkpoint(path)
    resumed.run()
    assert resumed.residual.snapshot() == reference.residual.snapshot()
    assert resumed.history_records == reference.history_records
    linear = make_step(**{**options, "quintic": False})
    before = linear.residual.snapshot()
    with pytest.raises(ValueError):
        linear.load_checkpoint(path)
    assert linear.residual.snapshot() == before
