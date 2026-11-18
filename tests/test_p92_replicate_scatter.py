import copy
import json

import pytest

from tools.p92_replicate_scatter import SOURCE, screen


def test_real_source_counts_and_unpooled_conditions():
    report = screen()
    assert len(report["groups"]) == 11
    assert sum(row["replicates"] for row in report["groups"]) == 18
    group = next(row for row in report["groups"]
                 if row["temperature_C"] == 600 and row["strain_amplitude"] == .004)
    assert group["replicates"] == 4
    result = group["survival"]
    assert not report["engineering_qualified"]
    json.dumps(result, allow_nan=False)


def test_duplicate_condition_not_silently_pooled():
    data = json.loads(SOURCE.read_text())
    data["temperatures"][0]["observations"].append(copy.deepcopy(data["temperatures"][0]["observations"][0]))
    with pytest.raises(ValueError, match="Repeated source condition"):
        screen(data)
