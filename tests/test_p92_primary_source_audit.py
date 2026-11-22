import numpy as np
import pytest

from tools.p92_primary_source_audit import audit, components, primary_rates


@pytest.mark.parametrize("regime,stress", [(0, 60.), (1, 100.), (2, 170.)])
def test_published_primary_curve_differentiation_and_core_units(regime, stress):
    seconds, hours = components(regime)[0], components(regime, seconds=False)[0]
    for t in (.001, .1, 1., 100.):
        row = primary_rates(regime, stress, t)
        delta = t*1e-5
        numerical = (hours.constant_stress_strain(stress, t+delta)
                     - hours.constant_stress_strain(stress, t-delta))/(2*delta)
        assert numerical == pytest.approx(row["differentiated_rate_per_h"], rel=1e-8)
        assert row["eliminated_rate_per_h"] == pytest.approx(numerical, rel=1e-8)
        assert seconds.equivalent_rate(stress, t*3600.)*3600. == pytest.approx(numerical, rel=1e-8)
        assert row["literal_to_derivative_ratio"] == pytest.approx(row["analytic_ratio"], rel=1e-12)
        assert row["literal_to_derivative_ratio"] > 10


def test_singular_initial_rate_has_finite_integral_without_time_reset():
    primary, secondary = components(0)
    assert np.isinf(primary.equivalent_rate(60., 0.))
    assert primary.constant_stress_strain(60., 0.) == 0
    full = sum(c.constant_stress_increment(60., 0., 3600.) for c in (primary, secondary))
    split = sum(c.constant_stress_increment(60., a, b)
                for a, b in ((0., 1.), (1., 1800.), (1800., 3600.)) for c in (primary, secondary))
    assert full == pytest.approx(split, rel=1e-12)
    reset = 2*sum(c.constant_stress_increment(60., 0., 1800.) for c in (primary, secondary))
    assert reset > full


def test_source_audit_never_promotes_the_other_heat_or_literal_equation():
    report = audit()
    assert all(r["numerical_passed"] for r in report["records"])
    assert not any(report[k] for k in ("fitted", "engineering_qualified", "life_qualified",
                                      "bam_validation", "printed_equation6_admitted"))
    assert len(report["records"]) == 6
    assert all(r["ratio_right_to_left"] != 1 for r in report["regime_boundary_limits"])


def test_audit_rejects_undefined_regime_and_zero_time_elimination():
    for index in (-1, 3, True):
        with pytest.raises(ValueError):
            components(index)
    with pytest.raises(ValueError):
        primary_rates(0, 60., 0.)
