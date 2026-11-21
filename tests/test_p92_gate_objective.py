import pytest

from tools.p92_gate_objective import residuals


def test_tolerance_scaling_reports_unresolved_without_admitting_it():
    report = dict(checks=[dict(name="coverage", status="passed"),
        dict(name="stress", error=10., limit=5., status="failed", cycle=1),
        dict(name="slope", status="unresolved", cycle=1)])
    value, info = residuals(report)
    assert value.tolist() == [4.]
    assert info["failed_count"] == 1
    assert info["unresolved"] == [("slope", 1, None)]


def test_incomplete_path_is_not_a_finite_penalty():
    with pytest.raises(ValueError, match="Incomplete"):
        residuals(dict(checks=[dict(name="path", status="failed")]))


@pytest.mark.parametrize("error,limit", [(1., 0.), (-1., 5.), (float("nan"), 5.)])
def test_invalid_scale_fails(error, limit):
    with pytest.raises(ValueError, match="finite"):
        residuals(dict(checks=[dict(name="stress", status="failed", error=error, limit=limit)]))
