"""Source admission/grouping check; not P92 scatter calibration or qualification."""

import argparse
import json
from pathlib import Path

from agentfem.assessments import LifeObservation, life_survival

SOURCE = Path(__file__).resolve().parents[1] / "evidence/creep/p92-db-cohort.json"


def observations(data):
    common = dict(population=data["material_condition"],
                  batch_id="DB source pipe; heat identifier not reported", status="failure")
    rows = []
    for i, (temperature, stress, life, _) in enumerate(data["creep"], 1):
        rows.append(LifeObservation(
            specimen_id=f"DB-table5.3-row-{i}", life=life, life_unit="h",
            failure_definition="creep rupture",
            conditions={"temperature_K": temperature+273.15, "stress_MPa": stress,
                        "loading": "uniaxial tensile creep"},
            source=data["source"] + "#Table-5.3", **common))
    protocol = data["fatigue_protocol"]
    for section, table in (("fatigue", "5.4"), ("dwell", "5.5")):
        for i, (peak, dwell, period, cycles, _) in enumerate(data[section], 1):
            rows.append(LifeObservation(
                specimen_id=f"DB-table{table}-row-{i}", life=cycles, life_unit="cycles",
                failure_definition=protocol["failure_definition"],
                conditions={"temperature_K": protocol["temperature_C"]+273.15,
                            "environment": protocol["environment"],
                            "minimum_strain": protocol["minimum_strain"],
                            "peak_strain": peak/100, "dwell_s": dwell, "period_s": period,
                            "strain_rate_per_s": protocol["strain_rate_per_s"],
                            "waveform": protocol["waveform"]},
                source=data["source"] + f"#Table-{table}", **common))
    return rows


def screen():
    data = json.loads(SOURCE.read_text())
    rows = observations(data)
    groups = {}
    for row in rows:
        groups.setdefault(row.cohort_key, []).append(row)
    return dict(
        schema="agentfem.research.p92-life-observations.v1",
        source=data["source"], source_sha256=data["source_sha256"],
        validation_role="source admission and exact-condition grouping, not scatter calibration",
        specimen_identity="Source table row identifiers; specimen serial numbers unavailable",
        observation_count=len(rows), condition_groups=len(groups),
        reported_right_censored_count=0, fitted_parameters=False, engineering_qualified=False,
        limitations=["No same-condition replicate scatter or censored tests reported in these tables.",
                     "Independent specimens are assumed from reported test rows, not experimentally verified.",
                     "Do not pool different stress, temperature, strain or dwell conditions."],
        groups=[life_survival(group, independent_specimens=True,
                              noninformative_censoring=True).as_dict()
                for group in groups.values()],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(screen(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
