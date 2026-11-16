import numpy as np
import pytest

from tools.p92_joint_protocol_fit import decode, fit_joint, training_path
from tools.p92_published_recovery_reference import isotropic_strength_lower_bound


def source():
    return dict(protocol="RF", sha256="fixture", cycle=np.repeat([1, 2, 3, 4], 2),
                time_s=np.arange(8.), strain=np.arange(8.)*.0001,
                stress_Pa=np.arange(8.)*1e6, temperature_K=np.full(8, 893.15))


def test_decode_keeps_sufficient_positive_strength_bound():
    for q in (.2, 1.5):
        for b in (.1, 3.):
            p = decode([1.5, .01, 1.2, q, b, 0.])
            assert isotropic_strength_lower_bound(p) == pytest.approx(1.)
    for bad in ([1.]*5, [1., 0., 1., 1., 1., 0.], [1., 1., np.nan, 1., 1., 0.]):
        with pytest.raises(ValueError):
            decode(bad)


def test_training_uses_only_first_three_cycles_and_time_weights():
    data = source()
    t, e, s, w, old = training_path(data)
    assert t.tolist() == list(range(6))
    assert np.sum(w*w) == pytest.approx(1.)
    assert w[0]**2 == pytest.approx(.1)
    assert w[1]**2 == pytest.approx(.2)
    assert old.tolist() == [s[0], 0., 0., 0., 0.]
    data["stress_Pa"][-2:] = 1e12
    assert np.array_equal(training_path(data)[2], s)


def test_rejects_wrong_protocol_and_invalid_budget_before_optimization():
    with pytest.raises(ValueError, match="RF and SLR"):
        fit_joint(source(), source())
    second = source()
    second["protocol"] = "SLR"
    for budget in (0, True, 2.5):
        with pytest.raises(ValueError, match="budget"):
            fit_joint(source(), second, max_nfev=budget)


def test_balanced_objective_retains_both_holds_and_loading():
    data = source()
    data["strain"] = np.array([0., 1., 1., 1., 0., 0., 0., 0.])*.001
    weights = training_path(data, balance_strain=True)[3]**2
    assert weights.sum() == pytest.approx(1.)
    assert weights[2] > 0  # Interior hold observation is not discarded.
    assert weights[1] > .1  # Rapid loading is not submerged by a long hold.
    data["strain"][:] = 0
    with pytest.raises(ValueError, match="nonconstant"):
        training_path(data, balance_strain=True)
