"""Coverage diagnostics distinguish missing evidence from successful coverage."""

import pytest
from tools.life_interval_coverage import coverage_screen


def test_seeded_reproducibility_and_complete_accounting():
    report = coverage_screen(sample_sizes=(1, 5), trials=30, seed=42)
    assert report == coverage_screen(sample_sizes=(1, 5), trials=30, seed=42)
    for row in report["records"]:
        assert row["finite_intervals"]+row["outside_followup"]+row["undefined_boundary_intervals"] == 30
        assert 0 <= row["covered"] <= row["finite_intervals"]
    assert report["records"][0]["conditional_coverage"] is None


@pytest.mark.parametrize("kwargs", [dict(trials=0), dict(trials=True),
    dict(sample_sizes=()), dict(sample_sizes=(1.5,)), dict(seed=-1)])
def test_invalid_sampling_plan(kwargs):
    with pytest.raises(ValueError):
        coverage_screen(**kwargs)
