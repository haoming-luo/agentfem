from dataclasses import asdict

import pytest

from tools.p92_published_recovery_reference import published_parameters
from tools.p92_reversal_sensitivity import perturb
from tools.p92_two_mechanism import Mechanism


@pytest.mark.parametrize("branch", [0, 1])
def test_shape_preserves_signed_tangent_and_other_branch(branch):
    original = (Mechanism(published_parameters(600)),)*2
    before = [asdict(m) for m in original]
    result = perturb(original, branch, "shape", .1)
    p, q = original[branch].parameters, result[branch].parameters
    assert p.C2*p.a2 == pytest.approx(q.C2*q.a2)
    assert q.a2/p.a2 == pytest.approx(10**.1)
    assert result[1-branch] == original[1-branch]
    assert [asdict(m) for m in original] == before


def test_tangent_preserves_saturation():
    original = (Mechanism(published_parameters(600)),)*2
    q = perturb(original, 0, "tangent", -.1)[0].parameters
    assert q.a2 == original[0].parameters.a2
    assert q.C2/original[0].parameters.C2 == pytest.approx(10**-.1)


@pytest.mark.parametrize("branch,direction,scale", [(2, "shape", .1), (0, "other", .1), (0, "shape", float("nan"))])
def test_invalid_coordinate_rejected(branch, direction, scale):
    with pytest.raises(ValueError):
        perturb((Mechanism(published_parameters(600)),)*2, branch, direction, scale)


def test_joint_coordinates_preserve_unselected_material_terms():
    from tools.p92_reversal_identification import decode
    original = (Mechanism(published_parameters(600)),)*2
    result = decode([.1, -.2], original)
    before, after = asdict(original[0].parameters), asdict(result[0].parameters)
    assert after.pop("a2") == pytest.approx(before.pop("a2")*10**.1)
    assert after.pop("C2") == pytest.approx(before.pop("C2")*10**-.3)
    assert before == after
    assert result[1] == original[1]
    assert decode([0., 0.], original) == original


def test_fit_budget_is_bounded_before_source_access():
    from tools.p92_reversal_identification import identify
    with pytest.raises(ValueError, match="budget"):
        identify({}, [], max_nfev=26)
