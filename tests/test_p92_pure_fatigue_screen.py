"""Published coefficients, preserved replicate scatter, explicit domain limits."""

import numpy as np
from tools.p92_pure_fatigue_screen import screen


def test_published_coefficient_screen_preserves_observations():
    result = screen()
    assert not result["engineering_qualified"]
    assert "not independent" in result["validation_role"]
    table = result["records"][0]
    row = next(r for r in table["predictions"] if r["amplitude"] == .004)
    assert row["cycles"] == [2045, 2120, 2186, 1426]
    np.testing.assert_allclose(row["predicted_cycles"], 1800.219766506373)
    assert len(row["prediction_ratios"]) == 4


def test_no_silent_domain_widening():
    result = screen()
    table = result["records"][0]
    assert table["curve"]["calibrated_cycles"] == [422., 9078.]
    rejected = [r for r in table["predictions"] if not r["admitted"]]
    assert {r["amplitude"] for r in rejected} == {.002, .008}
    assert all("predicted_cycles" not in r for r in rejected)


def test_direct_equation_reproduces_admitted_amplitudes():
    for table in screen()["records"]:
        curve = table["curve"]
        for row in table["predictions"]:
            if not row["admitted"]:
                continue
            reversals = 2 * row["predicted_cycles"]
            amplitude = (
                curve["fatigue_strength_coefficient"] / curve["young"]
                * reversals ** curve["fatigue_strength_exponent"]
                + curve["fatigue_ductility_coefficient"]
                * reversals ** curve["fatigue_ductility_exponent"]
            )
            np.testing.assert_allclose(amplitude, row["amplitude"], rtol=1e-12)
