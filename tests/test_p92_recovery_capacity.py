import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from tools.p92_recovery_capacity_audit import capacity


@pytest.mark.parametrize("lo,hi", [(1., 2.), (1441., 1794.), (100., 100.00001)])
def test_analytic_capacity_matches_independent_optimization(lo, hi):
    result = capacity([.001, .002], lo, hi)
    def rate(log_tau):
        tau = np.exp(log_tau)
        return np.exp(-lo/tau)*(-np.expm1(-(hi-lo)/tau))/(hi-lo)
    fit = minimize_scalar(lambda x: -rate(x), bounds=(np.log(lo)-15, np.log(hi)+15),
                          method="bounded", options={"xatol": 1e-12})
    assert result["maximum_average_rate_per_s"] == pytest.approx(-fit.fun*.003, rel=1e-10)
    assert result["finite_maximum_attained"]


def test_negative_modes_cannot_be_counted_as_positive_capacity():
    result = capacity([.001, -.004], 100., 200.)
    assert not result["finite_maximum_attained"]
    assert result["maximum_average_rate_per_s"] == capacity([.001], 100., 200.)["maximum_average_rate_per_s"]
    assert capacity([-.001], 100., 200.)["maximum_average_rate_per_s"] == 0.


@pytest.mark.parametrize("a,lo,hi", [([], 1., 2.), ([np.nan], 1., 2.), ([1.], 0., 1.),
                                         ([1.], 2., 1.), ([[1.]], 1., 2.)])
def test_invalid_conditional_bound_rejected(a, lo, hi):
    with pytest.raises(ValueError):
        capacity(a, lo, hi)
