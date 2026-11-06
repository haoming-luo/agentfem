"""Published-coefficient reproduction, not a new experimental qualification."""

import json

import numpy as np
from scipy.optimize import brentq

from tools.p92_rate_fatigue_screen import SOURCE, screen


def test_source_keeps_seven_valid_tests_and_two_invalid_tests_separate():
    data = json.loads(SOURCE.read_text())
    assert len(data["observations"]) == 7
    assert len(data["excluded"]) == 2
    assert len({row["id"] for row in data["observations"] + data["excluded"]}) == 9
    assert data["temperature_K"] == 873.15
    assert "stage-II" in data["failure_definition"]
    assert data["curves"][0]["epsilon_f"] == .57  # Not the pooled-fit alternative .42.
    assert {row["amplitude"] for row in data["observations"]} == {.004, .006, .008}


def test_source_and_calibration_roles_are_preserved():
    result = screen()
    json.dumps(result, allow_nan=False)
    assert not result["engineering_qualified"] and not result["fitted_parameters"]
    assert "not blind" in result["validation_role"]
    assert "540 C for 22000 h" in result["material_condition"]
    for record, bounds in zip(result["records"], ((543., 2233.), (534., 1640.)), strict=True):
        assert record["curve"]["calibrated_cycles"] == list(bounds)
        assert sum(row["role"] == "coefficient_reproduction" for row in record["comparisons"]) == 3
        assert all("inverse_rejection" in row for row in record["comparisons"] if row["amplitude"] == .008)


def test_forward_amplitudes_match_direct_published_equation():
    for record in screen()["records"]:
        p = record["curve"]
        for row in record["comparisons"]:
            if "amplitude_at_observed_life" not in row:
                continue
            reversals = 2 * row["cycles"]
            expected = p["fatigue_strength_coefficient"] * reversals**p["fatigue_strength_exponent"]
            expected += p["fatigue_ductility_coefficient"] * reversals**p["fatigue_ductility_exponent"]
            np.testing.assert_allclose(row["amplitude_at_observed_life"], expected, rtol=1e-13)


def test_cross_rate_error_is_not_hidden_by_retuning():
    high = screen()["records"][0]
    rows = {row["id"]: row for row in high["comparisons"]}
    np.testing.assert_allclose(rows["LCF_2.4e-5_0.4"]["predicted_cycles"], 2173.9624932518586)
    assert .32 < rows["LCF_2.4e-5_0.4"]["life_relative_error"] < .33
    assert .20 < rows["LCF_2.4e-5_0.6"]["life_relative_error"] < .21
    assert rows["LCF_2.4e-5_0.4"]["role"] == "cross_rate_diagnostic"


def test_explicit_tested_amplitude_domain_preserves_coefficients_and_matches_independent_root():
    legacy = screen()
    explicit = screen(domain="tested_amplitudes")
    for old, new in zip(legacy["records"], explicit["records"], strict=True):
        p = new["curve"]
        assert p["calibrated_strain_amplitudes"] == [.004, .008]
        for key in ("young", "fatigue_strength_coefficient", "fatigue_strength_exponent",
                    "fatigue_ductility_coefficient", "fatigue_ductility_exponent"):
            assert p[key] == old["curve"][key]
        for row in new["comparisons"]:
            def residual(cycles):
                return (p["fatigue_strength_coefficient"] * (2 * cycles)**p["fatigue_strength_exponent"]
                        + p["fatigue_ductility_coefficient"] * (2 * cycles)**p["fatigue_ductility_exponent"]
                        - row["amplitude"])
            expected = brentq(residual, 1., 1.e6, xtol=1.e-10)
            np.testing.assert_allclose(row["predicted_cycles"], expected, rtol=1.e-12)
    assert not explicit["engineering_qualified"]
