import json

import pytest

from tools.p92_prior_damage_audit import SOURCE, audit


def test_published_values_and_difficult_points_are_retained():
    result = audit()
    assert len(result["records"]) == 15
    rows = {row["id"]: row for row in result["records"]}
    assert rows[12]["prediction_over_observation"] == pytest.approx(550 / 129)
    assert rows[13]["prediction_over_observation"] == pytest.approx(479 / 186)
    assert {12, 13} <= set(result["groups"]["transfer"]["ids"])
    assert not result["material_law_executed"]
    assert not result["engineering_qualified"]
    differences = {r["model_id"]: r["difference_h"] for r in result["cross_source_links"]}
    assert differences == {1: -1, 7: 0, 14: 0, 10: 0, 11: 0, 12: 0, 13: 1, 15: 0}


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "129"])
def test_invalid_life_rejected(value):
    data = json.loads(SOURCE.read_text())
    data["rows"][11][6] = value
    with pytest.raises(ValueError, match="observed_h"):
        audit(data)


def test_duplicate_and_incomplete_partition_rejected():
    data = json.loads(SOURCE.read_text())
    data["rows"].append(data["rows"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        audit(data)
    data = json.loads(SOURCE.read_text())
    data["author_transfer_table_ids"].remove(12)
    with pytest.raises(ValueError, match="Incomplete"):
        audit(data)


def test_overlap_is_not_an_independent_holdout():
    data = json.loads(SOURCE.read_text())
    data["author_transfer_table_ids"].append(6)
    with pytest.raises(ValueError, match="partition"):
        audit(data)


def test_cross_source_duplicates_rejected():
    data = json.loads(SOURCE.read_text())
    data["cross_source_rows"].append(data["cross_source_rows"][0])
    with pytest.raises(ValueError, match="cross-source"):
        audit(data)
