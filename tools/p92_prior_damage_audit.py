"""Audit published remaining-life values; does not integrate or fit a material law."""

import json
import math
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "evidence/creep/p92-prior-damage.json"


def audit(data=None):
    if data is None:
        data = json.loads(SOURCE.read_text())
    columns = data["columns"]
    records = []
    for values in data["rows"]:
        if len(values) != len(columns):
            raise ValueError("Incomplete source row")
        row = dict(zip(columns, values, strict=True))
        for name in ("observed_h", "published_prediction_h", "lath_um"):
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {name}")
        fraction = row["prior_fraction"]
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction < 1:
            raise ValueError("Invalid prior_fraction")
        if type(row["id"]) is not int or row["id"] < 1:
            raise ValueError("Invalid row identity")
        ratio = row["published_prediction_h"] / row["observed_h"]
        records.append({**row, "prediction_over_observation": ratio,
                        "relative_error": ratio - 1})
    by_id = {row["id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate source row")
    groups = {}
    seen = set()
    for group in ("calibration", "transfer"):
        ids = data[f"author_{group}_table_ids"]
        if not ids or len(set(ids)) != len(ids) or seen.intersection(ids) or not set(ids) <= by_id.keys():
            raise ValueError("Invalid author group partition")
        seen.update(ids)
        errors = [abs(by_id[i]["relative_error"]) for i in ids]
        groups[group] = {"ids": ids, "mean_absolute_relative_error": sum(errors) / len(errors),
                         "max_absolute_relative_error": max(errors)}
    if seen != by_id.keys():
        raise ValueError("Incomplete author group partition")
    links = []
    linked = set()
    experiment_ids = set()
    for model_id, experiment_id, observed in data["cross_source_rows"]:
        if model_id not in by_id or model_id in linked or experiment_id in experiment_ids:
            raise ValueError("Invalid cross-source link")
        if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed) or observed <= 0:
            raise ValueError("Invalid cross-source observed life")
        linked.add(model_id)
        experiment_ids.add(experiment_id)
        links.append(dict(model_id=model_id, experiment_id=experiment_id,
                          model_table_h=by_id[model_id]["observed_h"], experiment_table_h=observed,
                          difference_h=by_id[model_id]["observed_h"] - observed))
    return dict(schema="agentfem.research.p92-prior-damage-audit.v1", records=records,
                groups=groups, cross_source_links=links, engineering_qualified=False,
                material_law_executed=False, predictions_origin=data["prediction_origin"],
                limitations=data["limitations"])


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))
