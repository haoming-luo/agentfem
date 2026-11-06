import pytest

from tools.p92_prior_equation_screen import screen


def test_complete_source_reconstruction_preserves_discrepancies():
    result = screen()
    rows = {row["table_id"]: row for row in result["records"]}
    assert set(rows) == set(range(1, 16))
    assert rows[1]["computed_h"] == pytest.approx(1188.0432345, rel=1e-8)
    assert rows[10]["relative_to_published"] > 0.09
    assert rows[12]["relative_to_observed"] > 3
    assert rows[13]["relative_to_observed"] > 1.5
    for row in rows.values():
        lower, upper = row["reading_sensitivity_h"]
        assert lower <= row["computed_h"] <= upper
    assert not result["fitted"]
    assert not result["engineering_qualified"]
    assert not result["independent_experimental_validation"]
