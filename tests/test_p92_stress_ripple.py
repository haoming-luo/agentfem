import numpy as np
import pytest

from tools import p92_recoverable_spectrum as kernel
from tools import p92_critical_recovery_reference as critical
from tools.p92_stress_ripple_audit import stress_path, wrapper, refine_stress_controls
from tools.p92_two_rate_reference import Branch, Material


def material():
    b = Branch(10., .001, 3., (100., 200.), (10., 20.), (.01, .02))
    return kernel.Material(Material(10000., (b, b)), (20000., 40000.), (.5, 10.))


def test_zero_ripple_and_independent_solution():
    m = material()
    y = np.zeros(12)
    y[0] = 110.
    reference = kernel.integrate(y, 2., 0., m, stress_control=True)
    result = stress_path(y, [0., 1., 2.], [0., 0., 0.], m, kernel)
    np.testing.assert_allclose(result.y[:, -1], reference.y[:, -1], atol=1e-9, rtol=2e-7)
    a = stress_path(y, [0., 1., 2.], [0., 2., -1.], m, kernel)
    b = stress_path(y, [0., 1., 2.], [0., 2., -1.], m, kernel, method="BDF", rtol=1e-11)
    np.testing.assert_allclose(a.y[:, -1], b.y[:, -1], atol=1e-8, rtol=2e-6)
    np.testing.assert_allclose(a.sol([0., 1., 2.])[0], [110., 112., 109.], atol=1e-9)


def test_split_continuation_and_input_immutability():
    m = material()
    y = np.zeros(12)
    y[0] = 110.
    before = y.copy()
    a = stress_path(y, [0., 1., 2.], [0., 2., -1.], m, kernel)
    b = stress_path(y, [0., 1.], [0., 2.], m, kernel)
    c = stress_path(b.y[:, -1], [0., 1.], [0., -3.], m, kernel)
    np.testing.assert_array_equal(a.y[:, -1], c.y[:, -1])
    np.testing.assert_array_equal(y, before)
    assert a.sol([]).shape == (12, 0)
    with pytest.raises(ValueError, match="outside"):
        a.sol([-1.])


def test_elastic_power_increment_is_not_omitted():
    m = material()
    y = np.zeros(12)
    a = stress_path(y, [0., .01], [0., 20.], m, kernel)
    end = a.y[:, -1]
    assert end[7] >= 20./m.backbone.modulus
    assert end[8] >= 20.**2/(2*m.backbone.modulus)
    assert end[8]-kernel.stored_energy(end, m)-end[9] == pytest.approx(0., abs=1e-9)


@pytest.mark.parametrize("scale", [0., -1., 2., True, np.nan, np.inf, 1j, [1.]])
def test_absolute_tolerance_scale_can_only_tighten(scale):
    with pytest.raises(ValueError):
        stress_path(np.zeros(12), [0., 1.], [0., 1.], material(), kernel, atol_scale=scale)


def test_tighter_absolute_tolerance_preserves_physical_solution():
    m = material()
    old = np.zeros(12)
    old[0] = 110.
    a = stress_path(old, [0., 1.], [0., -2.], m, kernel)
    b = stress_path(old, [0., 1.], [0., -2.], m, kernel,
                    method="BDF", rtol=1e-12, atol_scale=.01)
    np.testing.assert_allclose(a.y[:, -1], b.y[:, -1], rtol=1e-7, atol=1e-9)


def test_repeated_stress_knots_energy_is_checked_independently_of_strain():
    m = material()
    old = np.zeros(12)
    old[0] = 110.
    times = np.linspace(0., 4., 33)
    delta = 4.*np.sin(2.*np.pi*times)
    with pytest.raises(RuntimeError, match="mechanical balance"):
        stress_path(old, times, delta, m, kernel, method="BDF", rtol=1e-6)
    loose = stress_path(old, times, delta, m, kernel, method="BDF", rtol=1e-10)
    tight = stress_path(old, times, delta, m, kernel,
                        method="BDF", rtol=1e-12, atol_scale=.01)

    def imbalance(sol):
        last = sol.y[:, -1]
        return abs(last[8]-last[9]-kernel.stored_energy(last, m)+kernel.stored_energy(old, m))

    assert imbalance(tight) < 1e-8
    assert imbalance(tight) < imbalance(loose)/10.
    # Tightening integration does not alter the prescribed stress path.
    np.testing.assert_allclose(tight.sol(times)[0], old[0]+delta, atol=1e-8, rtol=0.)


@pytest.mark.parametrize("times,delta", [([0.], [0.]), ([0., 0.], [0., 1.]),
    ([0., 1.], [1., 2.]), ([0., 1.], [0., np.nan])])
def test_invalid_control(times, delta):
    with pytest.raises(ValueError):
        stress_path(np.zeros(12), times, delta, material(), kernel)


def test_unknown_event_kernel_rejected_and_cursor_is_transactional():
    with pytest.raises(ValueError, match="smooth"):
        stress_path(np.zeros(12), [0., 1.], [0., 1.], critical.Material(material()), critical)
    proxy = wrapper(kernel, [(np.array([0., 1.]), np.array([0., 1.]))])
    with pytest.raises(ValueError, match="identity"):
        proxy.integrate(np.zeros(12), 2., 0., material(), stress_control=True)
    assert proxy.count() == 0
    proxy.integrate(np.zeros(12), 1., 0., material(), stress_control=True)
    assert proxy.count() == 1
    with pytest.raises(ValueError, match="additional"):
        proxy.integrate(np.zeros(12), 1., 0., material(), stress_control=True)


def test_stress_refinement_preserves_the_identical_history():
    times, delta = np.array([0., .3, 1.]), np.array([0., 2., -1.])
    refined = refine_stress_controls([(times, delta)])
    np.testing.assert_array_equal(refined[0][0], [0., .15, .3, .65, 1.])
    query = np.linspace(0., 1., 31)
    np.testing.assert_allclose(np.interp(query, *refined[0]), np.interp(query, times, delta), atol=1e-15)
    y = np.zeros(12)
    y[0] = 110.
    a = stress_path(y, times, delta, material(), kernel)
    b = stress_path(y, *refined[0], material(), kernel)
    np.testing.assert_allclose(a.y[:, -1], b.y[:, -1], atol=1e-8)
    with pytest.raises(ValueError, match="refinement"):
        refine_stress_controls([([0., 0.], [0., 1.])])
