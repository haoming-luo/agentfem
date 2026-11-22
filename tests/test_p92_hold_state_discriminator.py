import numpy as np
import pytest

from tools import p92_recoverable_spectrum as kernel
from tools.p92_hold_state_discriminator import state_geometry, hold_metrics, reconstruct
from tools.p92_two_rate_reference import Branch, Material


def setup():
    b = Branch(10., .001, 3., (100., 200.), (10., 20.), (.01, .02))
    m = kernel.Material(Material(10000., (b, b)), (20000., 40000.), (.5, 10.))
    y = np.zeros(12)
    y[0] = 100.
    y[[1, 2, 4, 5]] = [2., 5., 2., 5.]
    return y, m


def test_state_geometry_preserves_observable_partition_and_box():
    y, m = setup()
    d, lo, hi = state_geometry(y, m)
    np.testing.assert_array_equal(d[[0, 3, 6, 7, 8, 9]], 0.)
    np.testing.assert_array_equal(d[10]+d[11], 0.)
    for u in (lo, hi, np.zeros(5)):
        z = y+d@u
        assert max(abs(z[[1, 4]])) <= 10.
        assert max(abs(z[[2, 5]])) <= 20.
        assert z[10]+z[11] == pytest.approx(y[10]+y[11], abs=1e-15)
    y[1] = 100.
    with pytest.raises(ValueError):
        state_geometry(y, m)


def test_synthetic_hold_reconstruction_keeps_material_and_causal_identity():
    y, m = setup()
    times = np.array([.01, .1, .5, 1., 2., 5., 9., 15., 25., 50., 80., 100.])
    solution = kernel.integrate(y, 100., 0., m)
    observed = solution.sol(times)
    tail_t = np.array([80., 100.])
    tail = solution.sol(tail_t)
    segment = dict(segment_index=5, cycle=1, stress_control=False, strain_hold=True,
        duration=100., sample_times=times.tolist(), measured_stress=observed[0].tolist(),
        measured_strain=observed[7].tolist(), source_indices=list(range(len(times))),
        tail=dict(times=tail_t.tolist(), stress=tail[0].tolist(), strain=tail[7].tolist(), source_indices=[10, 11]))
    residual, report = hold_metrics(segment, solution)
    np.testing.assert_array_equal(residual, 0.)
    assert report["all_hold_gates_passed"]
    result = reconstruct(y, m, segment)
    assert result["fit"]["success"] and result["candidate"]["all_hold_gates_passed"]
    assert result["numerical"]["passed"]
    assert not result["continuous_prediction"] and not result["state_reset_for_prediction"]
    assert not result["material_parameters_changed"] and not result["engineering_qualified"]
    np.testing.assert_array_equal(result["reconstructed_state"], y)
def test_optimizer_difference_retains_absolute_scale_at_zero_and_bounds():
    from tools.p92_hold_state_discriminator import bounded_jacobian
    calls = []

    def response(u):
        calls.append(u.copy())
        return np.array([2*u[0]+3*u[1], -u[0]+4*u[1]])

    actual = bounded_jacobian(response, [0., .3], [-.3, -.3], [.3, .3])
    np.testing.assert_allclose(actual, [[2., 3.], [-1., 4.]], atol=1e-12)
    assert calls[0][0] == .0005 and calls[1][0] == -.0005
    assert calls[2][1] == .3 and calls[3][1] == .2995


