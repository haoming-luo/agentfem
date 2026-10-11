from copy import deepcopy

import numpy as np
import pytest

from agentfem.elements._bulk_viscosity_policy import BulkViscosityPolicy
from test_response_hex_explicit import make_step


def test_policy_is_explicit_and_does_not_change_physical_density():
    policy = BulkViscosityPolicy(2.0, 3.0, 4.0)
    options = policy.options(np.array([1.0, 8.0]))
    np.testing.assert_array_equal(options["density"], [2.0, 0.25])
    np.testing.assert_array_equal(options["characteristic_length"], [4.0, 8.0])
    assert policy.summary()["vendor_equivalent"] is False
    for value in (0.0, -1.0, np.nan, np.inf, True, 1j):
        with pytest.raises(ValueError):
            BulkViscosityPolicy(value, 3.0, 4.0)
    with pytest.raises(ValueError):
        policy.options([-1.0])


def test_damping_ceiling_rejection_is_atomic():
    # Below the undamped caller ceiling, but above its damped counterpart.
    step = make_step(dt=1e-3, viscosity=True)
    before = step.residual.snapshot()
    nodal = step.state.snapshot()
    with pytest.raises(ValueError, match="damping-adjusted"):
        step.run()
    assert step.residual.snapshot() == before
    for name, value in nodal["fields"].items():
        np.testing.assert_array_equal(value, step.state.snapshot()["fields"][name])


def test_viscosity_checkpoint_identity_and_corruption():
    step = make_step(viscosity=True)
    step.run(until_step=2)
    saved = step.residual.snapshot()
    for name in ("viscosity_work", "damping_rate_bound"):
        corrupt = deepcopy(saved)
        corrupt[name] = -1.0
        with pytest.raises(ValueError, match="scalars"):
            step.residual.restore(corrupt)
        assert step.residual.snapshot() == saved
    with pytest.raises(ValueError, match="identity"):
        make_step().residual.restore(saved)


def test_prescribed_homogeneous_path_viscosity_work_converges():
    # Same physical path/end time, not ten steps at differing end times.
    # Analytic dilation x=(1+a*t)X: theta=3a/(1+a*t), rho=rho0/J,
    # L=L0*(1+a*t), so artificial power is proportional to 1/(1+a*t).
    # Endpoint secants exactly recover this affine velocity in this test.
    expected = 1.0 * 0.06 * 5.0 * 0.5 * (3 * 0.01) ** 2 * 0.001
    for count in (10, 20, 40):
        step = make_step(viscosity=True)
        residual = step.residual
        for i in range(1, count + 1):
            t = i * 0.001 / count
            step.state.u.value.interpolate(lambda x, t=t: 0.01 * t * x)
            residual.update_time(t)
            vector = residual.assemble_vector()
            vector.destroy()
            residual.commit()
        # Power = rho0*b1*c*L0*9a²/(1+a*t), not constant: one length
        # factor remains after the two rate factors. Integrate logarithm.
        exact = expected * np.log1p(0.01 * 0.001) / (0.01 * 0.001)
        assert residual.viscosity_work == pytest.approx(exact, rel=1e-6)


def test_viscosity_rigid_translation_has_no_artificial_work():
    step = make_step(viscosity=True)
    step.state.v.value.x.array.reshape(-1, 3)[:] = [0.1, -0.2, 0.3]
    step.run()
    assert abs(step.residual.viscosity_work) < 1e-25


def test_one_geometry_admission_per_damped_trial(monkeypatch):
    step = make_step(viscosity=True)
    cells = step.residual.internal.cells
    original = cells.deformation_gradient
    calls = []

    def checked(displacement):
        calls.append(1)
        return original(displacement)

    monkeypatch.setattr(cells, "deformation_gradient", checked)
    step.run(until_step=1)
    assert len(calls) == 1
    # Standalone damping still performs its own geometry admission.
    force, _ = step.residual.internal.assemble_bulk_viscosity(
        step.state.v.value,
        **step.residual.bulk_viscosity.options(np.ones(len(cells.volume))),
    )
    force.destroy()
    assert len(calls) == 2
