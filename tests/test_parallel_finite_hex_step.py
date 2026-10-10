# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Public finite-Hex result and disk-restart MPI integration."""

from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI

from test_finite_hex_step import make_step, problem


def test_distributed_finite_step_live_fields_and_restart(tmp_path):
    directory = Path(MPI.COMM_WORLD.bcast(str(tmp_path), root=0))
    source = make_step(MPI.COMM_WORLD)
    reference = source.solve_result(
        field_variables=("S", "F", "PEEQ", "SENER"),
        output=directory / "distributed.xdmf",
    )
    partial = make_step(MPI.COMM_WORLD)
    partial.run(until_step=7)
    checkpoint = partial.save_checkpoint(directory / "restart")
    resumed = make_step(MPI.COMM_WORLD)
    resumed.load_checkpoint(checkpoint)
    result = resumed.solve_result(field_variables=("S", "F", "PEEQ", "SENER"))
    np.testing.assert_array_equal(
        source.state.u.value.x.array, resumed.state.u.value.x.array
    )
    assert resumed.history_records[-1] == pytest.approx(source.history_records[-1])
    for name in ("S", "F", "PEEQ", "SENER"):
        np.testing.assert_array_equal(
            result.fields[name].field.x.array, reference.fields[name].field.x.array
        )


def test_rank_local_missing_density_is_collective_before_step_creation():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("Requires a rank-local admission failure")
    model, u, policy = problem(
        MPI.COMM_WORLD, density=None if MPI.COMM_WORLD.rank == 1 else 2
    )
    with pytest.raises((ValueError, RuntimeError), match="density"):
        model.step(
            target=u, element_policy=policy, omega_squared_bound=1e8, dt=1e-4, steps=20
        )


def test_rank_local_unsupported_asset_is_collective_before_material_setup():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("Requires a rank-local admission failure")
    model, u, policy = problem(MPI.COMM_WORLD)
    if MPI.COMM_WORLD.rank == 1:
        model.boundary_models.append(object())
    with pytest.raises((NotImplementedError, RuntimeError), match="not admitted"):
        model.step(
            target=u, element_policy=policy, omega_squared_bound=1e8, dt=1e-4, steps=20
        )


def test_rank_local_state_schema_mismatch_is_rejected_before_field_creation():
    from agentfem import constitutive

    if MPI.COMM_WORLD.size < 2:
        pytest.skip("Requires a rank-local schema mismatch")
    model, u, policy = problem(MPI.COMM_WORLD)
    if MPI.COMM_WORLD.rank == 1:
        material = model.materials[0].item
        schema = constitutive.MaterialStateSchema(
            name="incompatible_extra_state",
            variables=(
                *material.state_schema.variables,
                constitutive.MaterialStateVariable(name="extra"),
            ),
        )
        object.__setattr__(material, "state_schema", schema)
    with pytest.raises(
        (ValueError, RuntimeError), match="field contract|differs|inconsistent"
    ):
        model.step(
            target=u, element_policy=policy, omega_squared_bound=1e8, dt=1e-4, steps=20
        )


def test_finite_step_with_boundary_owned_only_on_some_ranks():
    comm = MPI.COMM_WORLD
    model, u, policy = problem(comm)
    model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
    step = model.step(
        target=u,
        element_policy=policy,
        omega_squared_bound=1e8,
        dt=1e-4,
        steps=20,
        progress=False,
    )
    step.run()
    model, u, policy = problem(MPI.COMM_SELF)
    model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
    serial = model.step(
        target=u,
        element_policy=policy,
        omega_squared_bound=1e8,
        dt=1e-4,
        steps=20,
        progress=False,
    )
    serial.run()
    for key in ("bulk_stored_energy", "kinetic_energy", "natural_load_work"):
        assert step.history_records[-1][key] == pytest.approx(
            serial.history_records[-1][key], rel=1e-9, abs=1e-16
        )
