from dataclasses import asdict

import numpy as np
import pytest

from tools.p92_subthreshold_probe import candidates, matched_branch
from tools.p92_two_rate_reference import Branch, Material
from tools import p92_recoverable_spectrum as spectrum


@pytest.mark.parametrize("anchor", [200., 250., 300.])
def test_matching_preserves_rate_and_derivative_without_mutation(anchor):
    old = Branch(125., 6e-5, 2.9, (1e5, 2e4), (25., 15.), (.1, .001))
    before = asdict(old)
    new = matched_branch(old, anchor)
    def value(b, stress):
        return b.reference_flow*(max(abs(stress)-b.strength, 0.)/120.)**b.exponent*np.sign(stress)
    assert value(new, anchor) == pytest.approx(value(old, anchor), rel=1e-12)
    delta = .001
    assert (value(new, anchor+delta)-value(new, anchor-delta))/(2*delta) == pytest.approx(
        (value(old, anchor+delta)-value(old, anchor-delta))/(2*delta), rel=1e-8)
    stresses = np.linspace(-300., 300., 101)
    values = np.array([value(new, s) for s in stresses])
    assert np.all(values*stresses >= 0.) and np.all(np.diff(values) > 0.)
    assert value(new, 120.) > 0. and value(old, 120.) == 0.
    assert asdict(old) == before
    assert new.strength == 0.


def test_spectrum_retained_with_mechanical_balance_and_independent_integration():
    branch = Branch(20., 1e-5, 3., (100., 200.), (10., 20.), (.01, .02))
    mat = spectrum.Material(Material(10000., (branch, branch)), (20000., 40000.), (.5, 10.))
    arms = candidates(mat)
    assert len(arms) == 4 and arms[0][1] is mat
    modified = arms[2][1]
    assert modified.moduli == mat.moduli and modified.times == mat.times
    assert modified.backbone.branches[0] == branch
    old = np.zeros(12)
    old[0], old[10:] = 70., [.001, .002]
    result = spectrum.integrate(old, 5., 0., modified, stress_control=True).y[:, -1]
    other = spectrum.integrate(old, 5., 0., modified, stress_control=True,
                               method="BDF", rtol=1e-10).y[:, -1]
    np.testing.assert_allclose(result, other, rtol=1e-7, atol=1e-8)
    assert result[9] >= 0.
    for anchor in (0., 20., np.nan):
        with pytest.raises(ValueError):
            matched_branch(branch, anchor)
