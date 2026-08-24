"""Assess declared fatigue cycles and hot dwells from named result histories."""

from __future__ import annotations

import json

from agentfem import assessments, results
from agentfem.constitutive import fatigue


service_cycle = results.SimulationResult("reviewed_service_cycle")
service_cycle.add_history(
    "hotspot_stress",
    [0.0, 1.0, 2.0, 4.0, 5.0],
    [0.0, 500.0, 120.0, 120.0, 0.0],
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
    name="reviewed project interaction",
    source="project assessment procedure revision A",
)
combined = assessments.creep_fatigue_from_result(
    service_cycle,
    fatigue_history="hotspot_stress",
    fatigue_curve=curve,
    stress_history="hotspot_stress",
    temperature_history="hotspot_temperature",
    dwells=(
        assessments.DwellInterval(
            2.0,
            4.0,
            repetitions=10.0,
            label="high-temperature dwell",
        ),
    ),
    rupture_time=lambda stress, temperature: (
        100_000.0 * (120.0 / stress) * (850.0 / temperature)
    ),
    rupture_source="project rupture relation revision A",
    interaction=interaction,
)
combined.attach(service_cycle)

print(json.dumps(combined.as_dict(), indent=2))
