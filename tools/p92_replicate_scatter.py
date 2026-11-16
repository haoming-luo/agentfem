"""P92 same-condition replicate survival; no pooled scatter or design life.

The source reports replicate cycle counts but not specimen serial numbers or
heat identifiers. Distinct, independent test specimens are an explicit analysis
assumption, not a fact established by this script. No failures are invented and
different temperatures/amplitudes are never pooled to inflate sample size.
"""

import json
from hashlib import sha256
from pathlib import Path

from agentfem.assessments import LifeObservation, life_survival

SOURCE = Path(__file__).resolve().parents[1] / "evidence/creep/p92-pure-fatigue.json"


def screen(data=None):
    data = json.loads(SOURCE.read_text()) if data is None else data
    groups, identities = [], set()
    for table in data["temperatures"]:
        temperature = table["temperature_C"]
        for record in table["observations"]:
            amplitude = record["amplitude"]
            identity = (temperature, amplitude)
            if identity in identities:
                raise ValueError("Repeated source condition; resolve specimen identity before pooling")
            identities.add(identity)
            rows = [LifeObservation(
                specimen_id=f"table1-{temperature}-{amplitude:g}-{i+1}",
                life=cycles, status="failure", life_unit="cycles",
                population="P92 source pipe, normalized and tempered",
                batch_id="Wang 2019 source stock; heat identifier not reported",
                failure_definition=data["failure_definition"],
                conditions={"temperature_C": temperature, "strain_amplitude": amplitude,
                            "strain_rate_per_s": .001, "dwell_s": 0.,
                            "strain_ratio": -1., "environment": "air"},
                source=data["source"] + " Table 1",
            ) for i, cycles in enumerate(record["cycles"])]
            survival = life_survival(rows, independent_specimens=True,
                                     noninformative_censoring=True).as_dict()
            groups.append(dict(temperature_C=temperature, strain_amplitude=amplitude,
                               replicates=len(rows), survival=survival))
    return dict(schema="agentfem.research.p92-replicate-survival.v1",
                source=data["source"],
                source_table_sha256=sha256(json.dumps(data, sort_keys=True,
                                           allow_nan=False).encode()).hexdigest(),
                groups=groups, independent_validation=False, engineering_qualified=False,
                assumptions=["Reported cycle counts represent distinct independent specimens",
                             "No censoring reported in this source table"],
                limitations=["Small samples: pointwise Greenwood intervals are approximate",
                             "Source failure definition remains unresolved",
                             "No parameter, model-discrepancy or component uncertainty included",
                             "Not a predictive interval, simultaneous band or design allowable",
                             "Do not pool amplitudes/temperatures or treat reused papers as extra tests"])


if __name__ == "__main__":
    print(json.dumps(screen(), indent=2, allow_nan=False))
