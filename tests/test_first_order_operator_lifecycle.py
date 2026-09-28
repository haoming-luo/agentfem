# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Operator-lifecycle tests for first-order transient procedures."""

from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import constitutive, fields, mesh, models, studies
from agentfem import time as time_api


def _heat_step(
    operator_policy: str,
    *,
    steps: int = 3,
    update_load=None,
    comm=MPI.COMM_SELF,
):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 1),
        comm=comm,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.transient_heat_transfer(dimension=2),
        mesh=domain,
    )
    temperature = model.field(fields.temperature(domain, value=300.0))
    model.material(
        constitutive.thermoelastic(
            young=1.0e9,
            poisson=0.3,
            density=1000.0,
            thermal_expansion=1.0e-5,
            conductivity=10.0,
            specific_heat=500.0,
            reference_temperature=300.0,
        )
    )
    model.heat_source(1.0e5)
    return model.step(
        target=temperature,
        dt=0.5,
        steps=steps,
        update_load=update_load,
        operator_policy=operator_policy,
        progress=False,
    )


def test_auto_reuses_first_order_operator_across_partial_runs():
    step = _heat_step("auto", steps=4)

    step.run(until_step=2)
    partial = step.operator_lifecycle_summary()
    assert partial["selected_policy"] == "reuse"
    assert partial["matrix_assembly_count"] == 1
    assert partial["rhs_assembly_count"] == 2

    result = step.solve_result()
    final = step.operator_lifecycle_summary()
    assert final["matrix_assembly_count"] == 1
    assert final["rhs_assembly_count"] == 4
    assert final["solve_count"] == 4
    assert result.metadata["step"]["operator_lifecycle"] == final
    assert step._prepared_problem is None


def test_refresh_first_order_policy_matches_reuse():
    reused = _heat_step("reuse")
    refreshed = _heat_step("refresh_each_step")

    reused.solve()
    refreshed.solve()

    np.testing.assert_allclose(
        refreshed.current.x.array,
        reused.current.x.array,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    lifecycle = refreshed.operator_lifecycle_summary()
    assert lifecycle["matrix_assembly_count"] == 3
    assert lifecycle["rhs_assembly_count"] == 3


def test_first_order_untyped_callback_refreshes_conservatively():
    step = _heat_step("auto", update_load=lambda _time: None)

    step.solve()

    lifecycle = step.operator_lifecycle_summary()
    assert lifecycle["selected_policy"] == "refresh_each_step"
    assert lifecycle["matrix_assembly_count"] == 3
    assert lifecycle["time_inputs"]["changes_operator"] is True


def test_first_order_rhs_declaration_preserves_reuse():
    update = time_api.input_update(
        lambda _time: None,
        effects="rhs",
        name="heat_source_history",
    )
    step = _heat_step("auto", update_load=update)

    step.solve()

    lifecycle = step.operator_lifecycle_summary()
    assert lifecycle["selected_policy"] == "reuse"
    assert lifecycle["matrix_assembly_count"] == 1
    assert lifecycle["time_inputs"]["effects"] == ("right_hand_side",)


def test_first_order_operator_declaration_refreshes_and_blocks_forced_reuse():
    update = time_api.input_update(
        lambda _time: None,
        effects="operator",
        name="conductivity_history",
    )
    automatic = _heat_step("auto", steps=2, update_load=update)

    automatic.solve()

    lifecycle = automatic.operator_lifecycle_summary()
    assert lifecycle["selected_policy"] == "refresh_each_step"
    assert lifecycle["matrix_assembly_count"] == 2
    with pytest.raises(ValueError, match="AFM-TRANSIENT-OPERATOR-002"):
        _heat_step("reuse", update_load=update)


def test_first_order_reuse_fails_closed_after_operator_control_change():
    step = _heat_step("reuse", steps=2)
    step.run(until_step=1)
    step.dt *= 2.0

    with pytest.raises(RuntimeError, match="AFM-TRANSIENT-OPERATOR-001"):
        step.run()

    assert step._prepared_problem is None


def test_first_order_rejects_unknown_operator_policy():
    with pytest.raises(ValueError, match="operator_policy"):
        _heat_step("sometimes")


def test_first_order_lifecycle_is_collective_under_mpi():
    step = _heat_step("auto", steps=2, comm=MPI.COMM_WORLD)

    step.solve()

    lifecycle = step.operator_lifecycle_summary()
    records = MPI.COMM_WORLD.allgather(lifecycle)
    assert all(item == records[0] for item in records)
    assert lifecycle["matrix_assembly_count"] == 1
    assert lifecycle["rhs_assembly_count"] == 2
