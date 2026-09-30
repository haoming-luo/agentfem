# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import benchmarks


def test_abaqus_explicit_finite_sliding_reference_is_a_fixed_public_protocol():
    reference = benchmarks.abaqus_explicit_finite_sliding_reference()

    assert reference.young == pytest.approx(30.0e6)
    assert reference.poisson == pytest.approx(0.3)
    assert reference.density == pytest.approx(0.284)
    assert reference.friction_coefficient == pytest.approx(0.3)
    assert reference.normal_load == pytest.approx(500.0)
    assert reference.sliding_displacement == pytest.approx(0.1)
    assert "simaver-c-finslcontdefrigid" in reference.reference
    assert "cpair_beam3d_xpl.inp" in reference.input_deck


def test_finite_sliding_assessment_accepts_balanced_coulomb_sliding():
    assessment = benchmarks.assess_finite_sliding_contact(
        contact_force_on_structure=(500.0, 0.0, 150.0),
        contact_force_on_surface=(-500.0, 0.0, -150.0),
        admissible_normal=(-1.0, 0.0, 0.0),
        young=30.0e6,
        poisson=0.3,
        density=0.284,
        applied_normal_load=500.0,
        friction_coefficient=0.3,
        sliding_displacement=0.1,
        active_point_count=6,
        sliding_point_count=6,
        invalid_point_count=0,
        facet_crossing_count=3,
        friction_dissipation=12.0,
        require_facet_crossing=True,
    )

    assert assessment.acceptable
    assert assessment.summary()["status"] == "accepted"
    assert assessment.relative_normal_balance_error == pytest.approx(0.0)
    assert assessment.relative_coulomb_cap_error == pytest.approx(0.0)
    assert assessment.relative_action_reaction_error == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("updates", "failed_quantity"),
    [
        (
            {"contact_force_on_structure": (450.0, 0.0, 135.0)},
            "normal_force_balance",
        ),
        ({"contact_force_on_structure": (500.0, 0.0, 100.0)}, "coulomb_cap"),
        (
            {"contact_force_on_surface": (-500.0, 0.0, -140.0)},
            "action_reaction",
        ),
        ({"sliding_point_count": 0}, "no_sliding_contact"),
        ({"invalid_point_count": 1}, "invalid_projection"),
        ({"facet_crossing_count": 0}, "no_facet_crossing"),
        ({"friction_dissipation": -1.0}, "negative_friction_dissipation"),
        (
            {
                "friction_coefficient": 0.2,
                "contact_force_on_structure": (500.0, 0.0, 100.0),
                "contact_force_on_surface": (-500.0, 0.0, -100.0),
            },
            "protocol_parameter:friction_coefficient",
        ),
        ({"young": 29.0e6}, "protocol_parameter:young"),
        ({"poisson": 0.29}, "protocol_parameter:poisson"),
        ({"density": 0.3}, "protocol_parameter:density"),
        (
            {"sliding_displacement": 0.05},
            "protocol_parameter:sliding_displacement",
        ),
    ],
)
def test_finite_sliding_assessment_fails_closed(updates, failed_quantity):
    inputs = {
        "contact_force_on_structure": (500.0, 0.0, 150.0),
        "contact_force_on_surface": (-500.0, 0.0, -150.0),
        "admissible_normal": (-1.0, 0.0, 0.0),
        "young": 30.0e6,
        "poisson": 0.3,
        "density": 0.284,
        "applied_normal_load": 500.0,
        "friction_coefficient": 0.3,
        "sliding_displacement": 0.1,
        "active_point_count": 6,
        "sliding_point_count": 6,
        "invalid_point_count": 0,
        "facet_crossing_count": 3,
        "friction_dissipation": 12.0,
        "require_facet_crossing": True,
    }
    inputs.update(updates)

    assessment = benchmarks.assess_finite_sliding_contact(**inputs)

    assert not assessment.acceptable, failed_quantity
    assert assessment.summary()["status"] == "failed"
    assert failed_quantity in assessment.failures


def test_finite_sliding_assessment_rejects_nonunit_normal():
    with pytest.raises(ValueError, match="unit vector"):
        benchmarks.assess_finite_sliding_contact(
            contact_force_on_structure=(500.0, 0.0, 150.0),
            contact_force_on_surface=(-500.0, 0.0, -150.0),
            admissible_normal=(-2.0, 0.0, 0.0),
            young=30.0e6,
            poisson=0.3,
            density=0.284,
            applied_normal_load=500.0,
            friction_coefficient=0.3,
            sliding_displacement=0.1,
            active_point_count=6,
            sliding_point_count=6,
            invalid_point_count=0,
            friction_dissipation=0.0,
        )


def test_finite_sliding_assessment_rejects_nonfinite_force():
    with pytest.raises(ValueError, match="finite"):
        benchmarks.assess_finite_sliding_contact(
            contact_force_on_structure=(np.nan, 0.0, 0.0),
            contact_force_on_surface=(0.0, 0.0, 0.0),
            admissible_normal=(-1.0, 0.0, 0.0),
            young=30.0e6,
            poisson=0.3,
            density=0.284,
            applied_normal_load=500.0,
            friction_coefficient=0.3,
            sliding_displacement=0.1,
            active_point_count=1,
            sliding_point_count=1,
            invalid_point_count=0,
            friction_dissipation=0.0,
        )


def test_finite_sliding_solid_bridge_runs_actual_two_stage_fem():
    bridge = benchmarks.finite_sliding_solid_protocol_bridge()

    assert bridge.acceptable, bridge.failures
    assert bridge.preload_transfer["equilibrium_accepted"]
    assert bridge.preload_relative_normal_balance_error < 1.0e-6
    assert bridge.assessment.relative_normal_balance_error < 1.0e-3
    assert bridge.assessment.relative_coulomb_cap_error < 1.0e-3
    assert bridge.assessment.relative_action_reaction_error < 1.0e-12
    assert bridge.assessment.active_point_count > 0
    assert (
        bridge.assessment.sliding_point_count
        == bridge.assessment.active_point_count
    )
    assert bridge.assessment.invalid_point_count == 0
    assert bridge.assessment.facet_crossing_count > 0
    assert bridge.assessment.require_facet_crossing is True
    assert bridge.assessment.friction_dissipation > 0.0
    assert bridge.preload_relative_energy_error < bridge.energy_tolerance
    assert bridge.sliding_relative_energy_error < bridge.energy_tolerance
    summary = bridge.summary()
    assert summary["status"] == "accepted"
    assert summary["comparison_level"] == (
        "public_protocol_bridge_not_b31_reproduction"
    )
    assert summary["surface_representation"] == (
        "triangulated_piecewise_planar"
    )


def test_finite_sliding_solid_bridge_is_rank_canonical_on_two_ranks():
    if MPI.COMM_WORLD.size != 2:
        pytest.skip("The distributed solid bridge is reviewed on two ranks.")

    bridge = benchmarks.finite_sliding_solid_protocol_bridge(comm=MPI.COMM_WORLD)
    summaries = MPI.COMM_WORLD.allgather(bridge.summary())

    assert bridge.acceptable, bridge.failures
    assert all(item == summaries[0] for item in summaries)
