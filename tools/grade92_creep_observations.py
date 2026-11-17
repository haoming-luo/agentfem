"""Source-audited optimized Grade 92 observations, not a P92 material card.

This research adapter deliberately delegates censoring and cohort checks to
the public LifeObservation/life_survival API. The three ongoing tests retain
their 2018 lower bounds; no current rupture status is inferred.
"""

import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from agentfem.assessments import LifeObservation, life_survival

SOURCE = Path(__file__).resolve().parents[1] / "evidence/creep/ornl-optimized-grade92-2018.json"
COLUMNS = ["specimen", "test_id", "plate", "laboratory", "stress_MPa",
           "temperature_C", "life_h", "status"]


def observations(data=None):
    data = json.loads(SOURCE.read_text()) if data is None else data
    if data["columns"] != COLUMNS:
        raise ValueError("Source columns changed; review the transcription schema")
    rows, identities = [], set()
    for values in data["observations"]:
        record = dict(zip(COLUMNS, values, strict=True))
        specimen = record["specimen"]
        if specimen in identities:
            raise ValueError("Duplicate specimen identity in source table")
        identities.add(specimen)
        plate = record["plate"]
        pedigree = data["plates"][plate]
        rows.append(LifeObservation(
            specimen_id=specimen, life=record["life_h"], status=record["status"],
            life_unit="h", population=data["population"], batch_id=pedigree["heat"],
            failure_definition=data["failure_definition"], source=data["source"] + " Table 3",
            conditions={"stress_MPa": record["stress_MPa"],
                        "temperature_C": record["temperature_C"], "plate": plate,
                        "laboratory": record["laboratory"],
                        "processing": data["processing"][pedigree["processing"]]},
        ))
    return tuple(rows)


def screen(data=None):
    data = json.loads(SOURCE.read_text()) if data is None else data
    rows = observations(data)
    groups = defaultdict(list)
    for row in rows:
        groups[row.cohort_key].append(row)
    reports = [life_survival(group, independent_specimens=True,
                            noninformative_censoring=True).as_dict()
               for group in groups.values()]
    return dict(
        schema="agentfem.research.grade92-creep-observations.v1",
        source=data["source"], pdf_sha256=data["pdf_sha256"],
        table_sha256=sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        specimens=len(rows), failures=sum(row.status == "failure" for row in rows),
        right_censored=sum(row.status == "right_censored" for row in rows),
        groups=reports, engineering_qualified=False, bam_p92_holdout=False,
        assumptions=["Distinct specimen IDs denote independent experiments",
                     "Ongoing tests are administratively censored at report cutoff"],
        limitations=["Optimized Grade 92 processing differs from ordinary P92 and BAM material",
                     "No stress/temperature/plate/laboratory pooling or extrapolation",
                     "Singleton/all-censored groups do not identify scatter or median life",
                     "Small replicate groups do not establish design lower tails",
                     "No creep-fatigue interaction or constitutive damage law is validated"],
    )


if __name__ == "__main__":
    print(json.dumps(screen(), indent=2, allow_nan=False))
