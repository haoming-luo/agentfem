from __future__ import annotations

import json

import pytest

from agentfem import assessments, results
from agentfem.constitutive import fatigue


def _fatigue_assessment(damage: float) -> fatigue.FatigueAssessment:
    return fatigue.FatigueAssessment(
        cycles=(fatigue.StressCycle(100.0, 0.0, 1.0),),
        damage=damage,
        repeated_history_life=float("inf") if damage == 0.0 else 1.0 / damage,
        source="verified stress history",
    )


def test_creep_time_fraction_retains_every_rupture_source():
    assessment = assessments.creep_time_fraction(
        (
            assessments.CreepDamageBlock(
                duration=1000.0,
                rupture_time=100_000.0,
                repetitions=2.0,
                label="base load dwell",
                source="reviewed rupture curve A",
            ),
            assessments.CreepDamageBlock(
                duration=20.0,
                rupture_time=10_000.0,
                repetitions=5.0,
                label="over-temperature event",
                source="reviewed rupture curve B",
            ),
        )
    )

    assert assessment.damage == pytest.approx(0.03)
    assert assessment.as_dict()["blocks"][1]["source"] == ("reviewed rupture curve B")
    with pytest.raises(ValueError, match="source"):
        assessments.CreepDamageBlock(duration=1.0, rupture_time=2.0)


def test_declared_interaction_curve_is_interpolated_without_hidden_code_data():
    diagram = assessments.interaction_diagram(
        points=((0.0, 1.0), (0.4, 0.6), (1.0, 0.0)),
        name="reviewed project interaction",
        source="project procedure CF-01 revision 2",
    )

    assert diagram.allowable_fatigue_damage(0.2) == pytest.approx(0.8)
    assert diagram.allowable_fatigue_damage(1.1) == 0.0
    assert diagram.as_dict()["source"] == "project procedure CF-01 revision 2"
    with pytest.raises(ValueError, match="decrease"):
        assessments.interaction_diagram(
            points=((0.0, 0.5), (1.0, 0.7)),
            name="invalid",
            source="test",
        )


def test_creep_fatigue_combines_existing_assessments_and_attaches_to_result():
    creep = assessments.creep_time_fraction(
        (
            assessments.CreepDamageBlock(
                duration=20.0,
                rupture_time=100.0,
                source="rupture test fit",
            ),
        )
    )
    combined = assessments.creep_fatigue(
        fatigue=_fatigue_assessment(0.7),
        creep=creep,
    )
    simulation = results.SimulationResult("creep_fatigue_assessment")
    combined.attach(simulation)

    assert combined.allowable_fatigue_damage == pytest.approx(0.8)
    assert combined.margin == pytest.approx(0.1)
    assert combined.acceptable
    assert simulation.quantity("creep_fatigue_margin") == pytest.approx(0.1)
    assert simulation.metadata["assessments"]["creep_fatigue"]["acceptable"]
    json.dumps(combined.as_dict())

    rejected = assessments.creep_fatigue(
        fatigue=_fatigue_assessment(0.81),
        creep=creep,
    )
    assert not rejected.acceptable


def test_assessments_is_an_advanced_public_module():
    import agentfem

    assert "assessments" in agentfem.public_api("advanced")
    assert agentfem.assessments.linear_interaction().name == (
        "linear damage interaction"
    )
