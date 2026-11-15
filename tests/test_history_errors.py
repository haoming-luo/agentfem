import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from agentfem.results import HistoryResult
from agentfem.verification import history_errors


def history(x, y, **kwargs):
    return HistoryResult("stress", x, y, unit="MPa", **kwargs)


def compare(a, b, **kwargs):
    return history_errors(a, b, reference_source="analytical fixture", max_interval=10, **kwargs)


def test_linear_error_exact_integral_is_sampling_independent():
    reports = []
    for x in ([0, 1, 10], [0, .1, .2, .3, 5, 10]):
        report = compare(history(x, np.asarray(x)*2), history(x, np.zeros(len(x))))
        reports.append(report)
        assert report["coordinate_weighted_mean_error"] == pytest.approx(10)
        assert report["coordinate_weighted_rms_error"] == pytest.approx(20/np.sqrt(3))
        assert report["maximum_absolute_error"] == 20
        json.dumps(report, allow_nan=False)
    assert reports[0]["sample_rms_error"] != reports[1]["sample_rms_error"]


def test_components_remain_separate_and_signed():
    a = history([0, 1], [[1, -2], [1, -2]])
    result = compare(a, history([0, 1], np.zeros((2, 2))))
    assert result["coordinate_weighted_mean_error"] == [1, -2]
    assert result["coordinate_weighted_rms_error"] == [1, 2]
    identity = result["actual"]["numeric_sha256"]
    a.values[:] = 0
    assert result["coordinate_weighted_mean_error"] == [1, -2]
    assert compare(a, a)["actual"]["numeric_sha256"] != identity


@pytest.mark.parametrize("changes", [dict(unit=None), dict(unit="Pa"),
    dict(abscissa_unit="h"), dict(abscissa_name="cycle"), dict(abscissa=[0, 2]),
    dict(values=[[0], [0]])])
def test_no_silent_conversion_alignment_or_broadcast(changes):
    a = history([0, 1], [0, 1])
    with pytest.raises(ValueError):
        compare(a, replace(a, **changes))


def test_missing_cycles_are_not_integrated_across():
    h = history([0, 1, 100], [0, 1, 1])
    with pytest.raises(ValueError, match="separate segments"):
        compare(h, h)


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_explicit_interval_contract(value):
    h = history([0, 1], [0, 1])
    with pytest.raises(ValueError, match="max_interval"):
        history_errors(h, h, reference_source="fixture", max_interval=value)


def test_reject_mutated_nonfinite_and_empty_samples():
    h = history([0, 1], [0, 1]); h.values[1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compare(h, h)
    for x, y in [([], []), ([0], [0]), ([0, 1], np.zeros((2, 0)))]:
        h = history(x, y)
        with pytest.raises(ValueError):
            compare(h, h)


def test_large_representable_errors_do_not_overflow_in_square():
    result = compare(history([0, 1], [1e200, 1e200]), history([0, 1], [0, 0]))
    assert result["coordinate_weighted_rms_error"] == 1e200
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="not representable"):
        compare(history([0, 1], [1e308, 1e308]), history([0, 1], [-1e308, -1e308]))


def test_doc_example_and_reference_identity():
    path = Path(__file__).resolve().parents[1]/"docs/scientific_verification.md"
    section = path.read_text().split("## Aligned history errors", 1)[1]
    example = section.split("```python\n", 1)[1].split("```", 1)[0]
    exec(compile(example, str(path), "exec"), {})
    h = history([0, 1], [0, 1])
    for source in ("", None, " "):
        with pytest.raises(ValueError, match="reference_source"):
            history_errors(h, h, reference_source=source, max_interval=1)
