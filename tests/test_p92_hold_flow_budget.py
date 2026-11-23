import numpy as np
import pytest

from tools import p92_recoverable_spectrum as kernel
from tools.p92_hold_flow_budget import budget
from tools.p92_two_rate_reference import Branch, Material


def material(*, elastic=False):
    branch = Branch(1e6 if elastic else .1, 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    return kernel.Material(Material(10000., (branch, branch)), (20000., 40000.), (.5, 10.))


@pytest.mark.parametrize("sign", [-1., 1.])
def test_signed_flow_and_accumulated_flow_are_not_interchangeable(sign):
    old = np.zeros(12)
    old[0] = sign*30.
    solution = kernel.integrate(old, 3., 0., material())
    row = budget(solution, material(), .1, 2.9)
    assert row["accounting_passed"]
    assert all(sign*x > 0 for x in row["signed_branch_strain_changes"])
    assert all(x > 0 for x in row["accumulated_branch_changes"])
    total = (row["total_strain_stress_secant_MPa_per_s"]
             + sum(row["signed_branch_stress_secants_MPa_per_s"])
             + sum(row["recoverable_stress_secants_MPa_per_s"]))
    assert total == pytest.approx(row["predicted_stress_secant_MPa_per_s"], abs=1e-6)


def test_exact_two_time_creep_budget_and_unloaded_recovery():
    mat, old = material(elastic=True), np.zeros(12)
    old[0] = 30.
    creep = kernel.integrate(old, 3., 0., mat, stress_control=True)
    row = budget(creep, mat, 0., 3.)
    expected = 30/np.asarray(mat.moduli)*(1-np.exp(-3/np.asarray(mat.times)))
    np.testing.assert_allclose(row["recoverable_strain_changes"], expected, atol=1e-12)
    assert row["signed_branch_strain_changes"] == [0., 0.]
    assert row["accounting_passed"]
    unloaded = creep.y[:, -1].copy()
    unloaded[0] = 0.
    recovery = kernel.integrate(unloaded, 3., 0., mat, stress_control=True)
    row = budget(recovery, mat, 0., 3.)
    assert all(x < 0 for x in row["recoverable_strain_changes"])
    assert row["total_strain_change"] < 0
    assert row["accounting_passed"]


def test_budget_rejects_extrapolation():
    solution = kernel.integrate(np.zeros(12), 1., .001, material())
    for a, b in ((-.1, .9), (.1, 1.1), (.5, .5), (np.nan, 1.)):
        with pytest.raises(ValueError, match="inside"):
            budget(solution, material(), a, b)
