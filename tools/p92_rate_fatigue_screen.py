"""Fixed published P92 coefficients; cross-rate diagnostics without refitting."""

import json
from pathlib import Path

from agentfem.constitutive import CoffinMansonBasquinCurve

SOURCE = Path(__file__).resolve().parents[1] / "evidence/creep/p92-service-aged-fatigue.json"


def screen():
    data = json.loads(SOURCE.read_text())
    records = []
    for parameters in data["curves"]:
        matching = [row for row in data["observations"] if row["rate"] == parameters["rate"]]
        lives = [row["cycles"] for row in matching]
        # The source supplies sigma_f/E, not separate elastic constants. A
        # dimensionless normalization evaluates that ratio without inventing E.
        curve = CoffinMansonBasquinCurve(
            young=1., fatigue_strength_coefficient=parameters["sigma_f_over_E"],
            fatigue_strength_exponent=parameters["b"],
            fatigue_ductility_coefficient=parameters["epsilon_f"],
            fatigue_ductility_exponent=parameters["c"],
            calibrated_cycles=(min(lives), max(lives)),
            stress_unit="normalized_by_E", source=data["source"] + " Table 4-4",
        )
        comparisons = []
        for observation in data["observations"]:
            row = dict(observation, role=("coefficient_reproduction" if observation["rate"] == parameters["rate"]
                                         else "cross_rate_diagnostic"))
            try:
                amplitude = curve.strain_amplitude(observation["cycles"])
                row.update(amplitude_at_observed_life=amplitude,
                           amplitude_relative_error=amplitude / observation["amplitude"] - 1.)
            except ValueError as error:
                row["forward_rejection"] = str(error)
            try:
                prediction = curve.cycles_to_failure(observation["amplitude"])
                row.update(predicted_cycles=prediction,
                           life_relative_error=prediction / observation["cycles"] - 1.)
            except ValueError as error:
                row["inverse_rejection"] = str(error)
            comparisons.append(row)
        records.append(dict(rate=parameters["rate"], curve=curve.as_dict(), comparisons=comparisons))
    return dict(
        schema="agentfem.research.p92-rate-fatigue-screen.v1",
        source=data["source"], source_sha256=data["source_sha256"],
        material_condition=data["material_condition"], temperature_K=data["temperature_K"],
        failure_definition=data["failure_definition"], validation_role=data["validation_role"],
        engineering_qualified=False, fitted_parameters=False, excluded=data["excluded"],
        bounds_policy="Each curve uses its own rate cohort's observed life extrema; no silent widening",
        normalization="E=1 and sigma_f=sigma_f/E are dimensionless evaluation coefficients, not elastic properties",
        records=records,
    )


if __name__ == "__main__":
    print(json.dumps(screen(), indent=2, allow_nan=False))
