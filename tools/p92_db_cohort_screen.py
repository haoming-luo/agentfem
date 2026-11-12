"""Audit a coherent P92 cohort using fixed published constants, without fitting."""

import json
from pathlib import Path

from agentfem.assessments import TabulatedRuptureCurve
from agentfem.constitutive import KachanovRabotnovCreep

SOURCE = Path(__file__).resolve().parents[1] / "evidence/creep/p92-db-cohort.json"


def rupture_curve(temperature_C, *, sparse=False):
    """Research-only uniaxial observations; no automatic alloy card selection."""
    data = json.loads(SOURCE.read_text())
    rows = [r for r in data["creep"] if r[0] == temperature_C]
    if not rows:
        raise ValueError("Only the three measured temperatures are admitted.")
    if sparse:
        # A transparent within-table sensitivity split, not a blind experiment.
        nodes = {650: (105, 120, 140), 675: (80, 100), 700: (65, 80)}
        rows = [r for r in rows if r[1] in nodes[temperature_C]]
    return TabulatedRuptureCurve(
        stresses=tuple(r[1] for r in rows), rupture_times=tuple(r[2] for r in rows),
        temperature=temperature_C + 273.15, stress_unit="MPa", time_unit="h",
        stress_measure="uniaxial_tensile", material_condition=data["material_condition"],
        failure_definition="creep specimen rupture",
        source=data["source"] + " Table 5.3" + ("; sparse within-table screen" if sparse else ""),
    )


def screen():
    data = json.loads(SOURCE.read_text())
    records = []
    for temperature, a, n, chi, b, phi in data["constants"]:
        law = KachanovRabotnovCreep(a, n, b, chi, phi)
        for t, stress, observed_h, observed_percent in data["creep"]:
            if t != temperature:
                continue
            predicted_h = law.rupture_time(stress)
            # Use a pre-endpoint fraction: evaluating precisely at a singular
            # endpoint is neither needed nor an experimental rupture prediction.
            update = law.update(stress, .9 * predicted_h)
            records.append(dict(
                temperature_C=t, stress_MPa=stress,
                observed_rupture_h=observed_h, predicted_rupture_h=predicted_h,
                life_relative_error=predicted_h / observed_h - 1,
                approximate_observed_failure_strain=observed_percent / 100,
                strain_at_90_percent_model_life=update.state.equivalent_creep_strain,
                damage_at_90_percent_model_life=update.state.damage,
                model=law.as_dict(),
            ))
    cycles = []
    for peak_percent, dwell, period, count, duration in data["fatigue"] + data["dwell"]:
        peak = peak_percent / 100
        cycles.append(dict(
            peak_strain=peak, strain_range=peak, strain_amplitude=peak / 2,
            mean_strain=peak / 2, minimum_strain=0., dwell_s=dwell,
            period_s=period, cycles=count, reported_duration_h=duration,
            calculated_duration_h=period * count / 3600,
            calculated_period_s=2 * peak / data["fatigue_protocol"]["strain_rate_per_s"] + dwell,
        ))
    interpolation = []
    for temperature in (650, 675, 700):
        curve = rupture_curve(temperature, sparse=True)
        comparisons = []
        for t, stress, observed_h, _ in data["creep"]:
            if t != temperature or stress in curve.stresses:
                continue
            predicted_h = curve(stress, curve.temperature)
            comparisons.append(dict(stress_MPa=stress, observed_h=observed_h,
                                    predicted_h=predicted_h, relative_error=predicted_h / observed_h - 1))
        interpolation.append(dict(curve=curve.as_dict(), comparisons=comparisons))
    return dict(
        schema="agentfem.research.p92-db-cohort-screen.v1",
        source=data["source"], source_sha256=data["source_sha256"],
        material_condition=data["material_condition"],
        validation_role=data["validation_role"],
        engineering_qualified=False, fitted_parameters=False,
        strain_interpretation="As printed coefficients with dimensionless strain; discrepancies retained, no percent correction",
        creep=records, cycles=cycles, limitations=data["limitations"],
        interpolation_role="Within-table omitted-node diagnostic; not a blind or independent-batch holdout",
        interpolation=interpolation,
    )


if __name__ == "__main__":
    print(json.dumps(screen(), indent=2, allow_nan=False))
