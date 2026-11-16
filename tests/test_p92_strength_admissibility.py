"""Scalar invariant checks for the bounded published P92 research law."""

from dataclasses import replace
import numpy as np
import pytest

from tools.p92_published_recovery_reference import (
    isotropic_strength_lower_bound, published_parameters, rates,
)


def test_published_cards_and_short_path_fit_have_distinct_domain_bounds():
    p = published_parameters(600)
    lower = p.Q+p.H/p.b
    assert isotropic_strength_lower_bound(p) == pytest.approx(p.k+lower)
    assert isotropic_strength_lower_bound(p) > 0
    assert isotropic_strength_lower_bound(published_parameters(650)) > 0
    assert isotropic_strength_lower_bound(replace(p, k=108.23149217400956)) < 0
    assert isotropic_strength_lower_bound(p, initial_R=-200.) == -20.


def test_vector_field_points_into_lower_invariant_for_every_flow_direction():
    p = published_parameters(600)
    lower = min(0., p.Q+p.H/p.b, p.Qr)
    for accumulated in (0., .01, 1., 10.):
        for stress in (-500., -100., 0., 100., 500.):
            for backstress in (-30., 0., 30.):
                state = np.array([stress, backstress, -backstress/2, lower, accumulated])
                assert rates(state, 0., p)[3] >= -1e-13
                state[3] = lower-1.
                assert rates(state, 0., p)[3] > 0


@pytest.mark.parametrize("changes", [dict(b=0.), dict(gamma_r1=-1.),
    dict(gamma_r2=-1.), dict(r=-1.), dict(mr=.5), dict(k=np.nan), dict(k=True)])
def test_bound_rejects_unsupported_coefficient_domain(changes):
    with pytest.raises(ValueError):
        isotropic_strength_lower_bound(replace(published_parameters(600), **changes))


def test_calibration_constraint_only_changes_the_declared_lower_bound(monkeypatch):
    from types import SimpleNamespace
    from tools import p92_bam_cyclic_screen as module
    captured = []
    def optimizer(function, initial, **kwargs):
        captured.append(kwargs["bounds"])
        return SimpleNamespace(x=initial, success=True, message="fixture", nfev=1,
                               active_mask=np.zeros(3), cost=0., jac=np.eye(3))
    monkeypatch.setattr(module, "least_squares", optimizer)
    args = ([0., 1.], [0., .001], [0., 100.], np.zeros(5))
    _, original = module.calibrate(*args)
    _, constrained = module.calibrate(*args, positive_strength=True)
    assert original["k_lower_bound_MPa"] == 50.
    assert constrained["k_lower_bound_MPa"] == pytest.approx(115.99827260493827)
    assert captured[0][0][0::2] == captured[1][0][0::2]
    assert captured[0][1] == captured[1][1]
    assert constrained["positive_strength_constraint"]


def test_incompatible_initial_history_is_rejected_before_fitting(monkeypatch):
    from tools import p92_bam_cyclic_screen as module
    def unexpected(*args, **kwargs):
        pytest.fail("infeasible calibration must not invoke the optimizer")
    monkeypatch.setattr(module, "least_squares", unexpected)
    with pytest.raises(ValueError, match="outside the frozen"):
        module.calibrate([0., 1.], [0., .001], [0., 100.],
                         np.array([0., 0., 0., -251., 0.]), positive_strength=True)
