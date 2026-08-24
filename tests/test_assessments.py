from __future__ import annotations

import json

import numpy as np
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


def test_sequential_energy_ledger_keeps_thermal_and_mechanical_layers_distinct():
    thermal = results.SimulationResult("thermal")
    thermal.add_histories(
        [1.0, 2.0],
        {
            "thermal_content": [10.0, 14.0],
            "applied_heat_rate": [5.0, 5.0],
            "outward_heat_rate": [1.0, 1.0],
            "heat_balance_residual": [0.0, 2.0e-10],
        },
    )
    mechanical = results.SimulationResult("creep")
    mechanical.add_histories(
        [1.0, 2.0],
        {
            "internal_energy": [2.0, 3.0],
            "external_work": [2.0, 3.0 + 1.0e-9],
            "mechanical_energy_residual": [0.0, 1.0e-9],
        },
    )

    ledger = assessments.sequential_energy_ledger(thermal, mechanical)
    ledger.attach(mechanical)

    record = ledger.as_dict()
    assert record["coupling"] == "one_way_sequential"
    assert record["full_coupled_conservation_claim"] is False
    assert record["thermal"]["residual_max_abs"] == pytest.approx(2.0e-10)
    assert record["mechanical"]["residual_max_abs"] == pytest.approx(1.0e-9)
    assert "combined_residual" not in record
    assert np.isfinite(
        mechanical.quantity("sequential_energy_mechanical_residual_max_abs")
    )
    json.dumps(record)


def test_creep_fatigue_v1_extracts_declared_dwells_from_named_histories():
    simulation = results.SimulationResult("service_cycle")
    time = [0.0, 1.0, 2.0, 4.0, 5.0]
    simulation.add_history(
        "hotspot_stress",
        time,
        [0.0, 100.0, 120.0, 120.0, 0.0],
        unit="MPa",
    )
    simulation.add_history(
        "temperature",
        time,
        [700.0, 800.0, 850.0, 840.0, 700.0],
        unit="K",
    )
    curve = fatigue.BasquinCurve(
        fatigue_strength_coefficient=1000.0,
        fatigue_strength_exponent=-0.1,
    )

    combined = assessments.creep_fatigue_from_result(
        simulation,
        fatigue_history="hotspot_stress",
        fatigue_curve=curve,
        stress_history="hotspot_stress",
        temperature_history="temperature",
        dwells=(
            assessments.DwellInterval(
                2.0, 4.0, repetitions=10.0, label="rated hold"
            ),
        ),
        rupture_time=lambda stress, temperature: 100_000.0
        * (120.0 / stress)
        * np.exp((850.0 - temperature) / 100.0),
        rupture_source="reviewed project rupture fit revision A",
    )

    block = combined.creep.blocks[0]
    assert block.stress == pytest.approx(120.0)
    assert block.temperature == pytest.approx(850.0)
    assert block.duration == pytest.approx(2.0)
    assert block.damage == pytest.approx(2.0e-4)
    assert combined.fatigue.source == "service_cycle:hotspot_stress"
    assert combined.as_dict()["creep"]["blocks"][0]["source"] == (
        "reviewed project rupture fit revision A"
    )

    with pytest.raises(ValueError, match="outside history"):
        assessments.creep_blocks_from_result(
            simulation,
            stress_history="hotspot_stress",
            temperature_history="temperature",
            dwells=(assessments.DwellInterval(4.0, 6.0),),
            rupture_time=lambda stress, temperature: 1.0,
            rupture_source="test",
        )
