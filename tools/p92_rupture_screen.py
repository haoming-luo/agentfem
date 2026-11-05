"""Public rupture-table reproduction, not a P92 design curve or life fit."""

import json
from pathlib import Path

from agentfem import assessments, results

SOURCE = Path(__file__).resolve().parents[1] / "evidence/creep/p92-rupture-table.json"


def screen():
    data = json.loads(SOURCE.read_text())
    records = []
    for condition in data["conditions"]:
        rows = condition["rows"]
        curve = assessments.TabulatedRuptureCurve(
            stresses=tuple(row["stress"] for row in rows),
            rupture_times=tuple(row["rupture_time"] for row in rows),
            temperature=data["temperature_K"],
            stress_unit=data["stress_unit"],
            time_unit=data["time_unit"],
            stress_measure=data["stress_measure"],
            material_condition=condition["material_condition"],
            failure_definition=data["failure_definition"],
            source=data["source"] + " Table 2; " + condition["name"],
        )
        checks = []
        for row in rows:
            # Constant supplied test stress, NOT a computed structural history.
            end = row["rupture_time"] / 2
            result = results.SimulationResult("published_rupture_node")
            result.add_history("stress", [0, end], [row["stress"]] * 2,
                               unit="MPa", abscissa_unit="h")
            result.add_history("temperature", [0, end], [data["temperature_K"]] * 2,
                               unit="K", abscissa_unit="h")
            damage = assessments.creep_damage_from_result(
                result, stress_history="stress", temperature_history="temperature",
                dwells=(assessments.DwellInterval(0, end),), rupture_time=curve,
                rupture_source=curve.source, stress_unit="MPa", time_unit="h",
                stress_measure=data["stress_measure"],
            )
            checks.append(dict(id=row["id"], supplied_rupture_hours=row["rupture_time"],
                               recovered_rupture_hours=curve(row["stress"], data["temperature_K"]),
                               half_life_time_fraction=damage.damage))
        records.append(dict(condition=condition["name"], curve=curve.as_dict(), checks=checks))
    return dict(schema="agentfem.research.p92-rupture-screen.v1", records=records,
                source_sha256=data["source_sha256"], engineering_qualified=False,
                validation_role="supplied-node reproduction and time-fraction verification",
                excluded_interrupted_test_ids=data["excluded_interrupted_test_ids"],
                limitations=["Two endpoints per material condition; no independent holdout",
                             "No replicate scatter or engineering allowable",
                             "No batch identity with the separate fatigue sources",
                             "Nominal stress is not automatically Cauchy stress",
                             "Interpolation is a declared numerical choice, not validated life"])


if __name__ == "__main__":
    print(json.dumps(screen(), indent=2, allow_nan=False))
