"""Combine reviewed fatigue and creep consumers without hiding source data."""

from __future__ import annotations

import json

from agentfem import assessments
from agentfem.constitutive import fatigue


curve = fatigue.BasquinCurve(
    fatigue_strength_coefficient=1000.0,
    fatigue_strength_exponent=-0.1,
)
fatigue_result = fatigue.assess_history(
    [0.0, 500.0, 0.0, -500.0, 0.0],
    curve,
    source="reviewed elastic stress history",
)
creep_result = assessments.creep_time_fraction(
    (
        assessments.CreepDamageBlock(
            duration=1000.0,
            rupture_time=100_000.0,
            repetitions=10.0,
            label="high-temperature dwell",
            source="project rupture relation revision A",
        ),
    )
)
interaction = assessments.interaction_diagram(
    points=((0.0, 1.0), (0.4, 0.5), (1.0, 0.0)),
    name="reviewed project interaction",
    source="project assessment procedure revision A",
)
combined = assessments.creep_fatigue(
    fatigue=fatigue_result,
    creep=creep_result,
    interaction=interaction,
)

print(json.dumps(combined.as_dict(), indent=2))
