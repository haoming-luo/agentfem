import numpy as np
import pytest

from tools.p92_recovery_budget_sensitivity import cards, contrast
from tools.p92_recoverable_spectrum import Material
from tools.p92_two_rate_reference import Branch, Material as Backbone


def test_declared_lattice_does_not_expand_log_radius_or_reorder_times():
    b = Branch(10., .001, 3., (100., 200.), (10., 20.), (.01, .02))
    m = Material(Backbone(155176., (b, b)), (20000., 40000.), (.5, 80.))
    records = list(cards(m))
    assert len(records) == 211
    assert {r["modulus_MPa"] for r in records} == {151000., 155176., 162000.}
    assert sum(not np.any(r["log10_coordinates"]) for r in records) == 1
    for r in records:
        assert np.linalg.norm(r["log10_coordinates"]) <= .3+1e-14
        assert 0 < r["times"][0] < r["times"][1]
        np.testing.assert_allclose(r["moduli"], np.array(m.moduli)*10.**np.array(r["log10_coordinates"][1:3]))


def test_contrast_interval_preserves_units_and_direction():
    a = dict(remainder_interval=[2., 4.], duration_s=2.)
    b = dict(remainder_interval=[3., 6.], duration_s=3.)
    assert contrast(a, b) == pytest.approx([-1., 1.])
    c = dict(remainder_interval=[1., 2.], duration_s=4.)
    assert contrast(a, c) == pytest.approx([.5, 1.75])
    assert contrast(c, a) == pytest.approx([-1.75, -.5])
