"""Reproduce published pure-fatigue coefficients through the public API.

No refitting and no life extrapolation. Observed life extrema are conservative
admission bounds chosen for this screen, not author-declared design limits.
Rejected amplitudes remain visible instead of widening bounds to hide them.
"""

import json
from pathlib import Path

from agentfem.constitutive import CoffinMansonBasquinCurve

SOURCE = Path(__file__).resolve().parents[1] / "evidence/creep/p92-pure-fatigue.json"


def screen():
    data = json.loads(SOURCE.read_text())
    records = []
    for table in data["temperatures"]:
        observed = [n for row in table["observations"] for n in row["cycles"]]
        curve = CoffinMansonBasquinCurve(
            young=table["young_MPa"],
            fatigue_strength_coefficient=table["sigma_f_MPa"],
            fatigue_strength_exponent=table["b"],
            fatigue_ductility_coefficient=table["epsilon_f"],
            fatigue_ductility_exponent=table["c"],
            calibrated_cycles=(min(observed), max(observed)),
            stress_unit="MPa",
            source=data["source"] + " Table 2; research reproduction",
        )
        predictions = []
        for row in table["observations"]:
            try:
                predicted = curve.cycles_to_failure(row["amplitude"])
            except ValueError as error:
                predictions.append(dict(row, admitted=False, reason=str(error)))
                continue
            predictions.append(
                dict(row, admitted=True, predicted_cycles=predicted,
                     prediction_ratios=[predicted / n for n in row["cycles"]])
            )
        records.append(dict(temperature_C=table["temperature_C"],
                            curve=curve.as_dict(), predictions=predictions))
    return dict(
        schema="agentfem.research.p92-pure-fatigue-screen.v1",
        source=data["source"], validation_role=data["validation_role"],
        failure_definition=data["failure_definition"],
        engineering_qualified=False,
        cycle_bounds_policy="Observed life extrema; screen choice, not a published allowable domain",
        records=records,
    )


if __name__ == "__main__":
    print(json.dumps(screen(), indent=2, allow_nan=False))
