"""Synthetic postprocessing example; no material or design-code validation."""

from __future__ import annotations

import json

from agentfem import assessments, results
from agentfem.constitutive import fatigue


service_cycle = results.SimulationResult("synthetic_service_cycle")
service_cycle.add_history(
    "hotspot_stress",
    [0.0, 1.0, 2.0, 4.0, 5.0],
    [0.0, 500.0, 120.0, 80.0, 0.0],
    unit="MPa",
)
service_cycle.add_history(
    "hotspot_temperature",
    [0.0, 1.0, 2.0, 4.0, 5.0],
    [700.0, 800.0, 850.0, 840.0, 700.0],
    unit="K",
)
curve = fatigue.BasquinCurve(
    fatigue_strength_coefficient=1000.0,
    fatigue_strength_exponent=-0.1,
)
interaction = assessments.interaction_diagram(
    points=((0.0, 1.0), (0.4, 0.5), (1.0, 0.0)),
    name="synthetic interaction example",
    source="interface demonstration only; not a design-code curve",
)


def rupture_time(stress, temperature):
    """Demonstrate a project-owned bounded relation, in MPa, K and seconds."""
    if not 80. <= stress <= 120. or not 840. <= temperature <= 850.:
        raise ValueError("Outside this synthetic demonstration's declared domain.")
    return 100_000. * (120. / stress) * (850. / temperature)


creep = assessments.creep_damage_from_result(
    service_cycle,
    stress_history="hotspot_stress",
    temperature_history="hotspot_temperature",
    dwells=(
        assessments.DwellInterval(
            2.0,
            4.0,
            label="high-temperature dwell",
        ),
    ),
    rupture_time=rupture_time,
    rupture_source="synthetic bounded relation: 80..120 MPa, 840..850 K",
    stress_unit="MPa", time_unit="s", stress_measure="signed_axial",
    maximum_source_interval=2.0,
)
combined = assessments.creep_fatigue(
    fatigue=fatigue.assess_result_history(service_cycle, "hotspot_stress", curve),
    creep=creep,
    interaction=interaction,
)
combined.attach(service_cycle)

print(json.dumps(combined.as_dict(), indent=2, allow_nan=False))
