# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Distributed owned-cell finite response, separate from interface communication."""

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import constitutive
from agentfem.constitutive.material_driver import MaterialQuadratureResponse
from agentfem.elements._finite_uniform_hex_dolfinx import FiniteUniformHexResidual
from agentfem.elements._finite_uniform_hex_material import response_fields
from test_parallel_uniform_hex import problem


def setup(comm, counts=(4, 2, 2)):
    domain, u, _ = problem(comm, counts)
    law = constitutive.finite_strain_j2_logarithmic(
        young=100, poisson=0.3, yield_stress=1, hardening_modulus=5
    )
    response = MaterialQuadratureResponse.create(
        domain,
        law.state_schema,
        degree=1,
        stored_energy_component_names=law.stored_energy_component_names,
    )
    residual = FiniteUniformHexResidual(
        u, response, density=2, hourglass_modulus=40, hourglass_scale=0.1, chunk_size=5
    )
    return u, law, response, residual


def evaluate(residual, law):
    return residual.evaluate(
        law,
        deformation_gradient_old=np.tile(np.eye(3), (len(residual.cell_nodes), 1, 1)),
        time=0,
        time_increment=0.01,
    )


@pytest.mark.parametrize("counts", [(4, 2, 2), (1, 1, 1)])
def test_owned_finite_force_mass_energy_and_tangent_match_serial(counts):
    records = []
    for comm in (MPI.COMM_WORLD, MPI.COMM_SELF):
        u, law, response, operator = setup(comm, counts)
        xyz = u.function_space.tabulate_dof_coordinates()
        # Spatially varying finite strain includes physical and stabilization energy.
        u.x.array[:] = np.column_stack(
            (0.1 * xyz[:, 0] ** 2, 0.02 * xyz[:, 1] * xyz[:, 0], -0.01 * xyz[:, 2])
        ).ravel()
        vector, trial = evaluate(operator, law)
        direction = np.column_stack(
            (np.sin(xyz[:, 0]), xyz[:, 1] ** 2, xyz[:, 2] * xyz[:, 0])
        ).ravel()
        action = operator.tangent_action(direction)
        owned = u.function_space.dofmap.index_map.size_local
        records.append(
            (
                xyz[:owned].copy(),
                vector.array.copy().reshape(-1, 3),
                action.array.copy().reshape(-1, 3),
                operator.mass_diagonal.copy().reshape(-1, 3),
                comm.allreduce(float(trial.element_response.physical_energy.sum())),
                comm.allreduce(float(trial.element_response.hourglass_energy.sum())),
            )
        )
        vector.destroy()
        action.destroy()
    parallel, serial = records
    for i, point in enumerate(parallel[0]):
        j = np.flatnonzero(np.linalg.norm(serial[0] - point, axis=1) < 1e-12)
        assert len(j) == 1
        for field in (1, 2, 3):
            np.testing.assert_allclose(
                parallel[field][i], serial[field][j[0]], rtol=1e-11, atol=1e-13
            )
    np.testing.assert_allclose(parallel[4:], serial[4:], rtol=1e-12, atol=1e-14)
    force = parallel[1]
    current = parallel[0] + np.column_stack(
        (
            0.1 * parallel[0][:, 0] ** 2,
            0.02 * parallel[0][:, 1] * parallel[0][:, 0],
            -0.01 * parallel[0][:, 2],
        )
    )
    np.testing.assert_allclose(MPI.COMM_WORLD.allreduce(force.sum(0)), 0, atol=1e-12)
    np.testing.assert_allclose(
        MPI.COMM_WORLD.allreduce(np.cross(current, force).sum(0)), 0, atol=1e-12
    )


@pytest.mark.parametrize("failure", ["geometry", "force", "downstream"])
def test_rank_local_failure_restores_material_fields_everywhere(monkeypatch, failure):
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("Collective failure requires multiple ranks")
    u, law, response, operator = setup(MPI.COMM_WORLD)
    before = {
        name: field.x.array.copy() for name, field in response_fields(response).items()
    }

    def fail(*args, **kwargs):
        raise ValueError("injected rank-local finite response failure")

    if MPI.COMM_WORLD.rank == 1:
        if failure == "geometry":
            monkeypatch.setattr(operator.cells, "deformation_gradient", fail)
        elif failure == "force":
            monkeypatch.setattr(
                operator.cells, "_response_from_checked_displacement", fail
            )
    with pytest.raises((RuntimeError, ValueError), match="rank-local|another rank"):
        with operator.trial_evaluation(
            law,
            deformation_gradient_old=np.tile(
                np.eye(3), (len(operator.cell_nodes), 1, 1)
            ),
            time=0,
            time_increment=0.01,
        ):
            if failure == "downstream" and MPI.COMM_WORLD.rank == 1:
                fail()
    for name, field in response_fields(response).items():
        np.testing.assert_array_equal(field.x.array, before[name])
    np.testing.assert_array_equal(
        response.state.trial_state_vectors(), response.state.committed_state_vectors()
    )
    assert not operator._tangent_available


def dynamic_step(comm, counts=(4, 2, 2)):
    from agentfem import fracture, problems, state, time
    from agentfem.mechanics._finite_hex_explicit import FiniteHexExplicitResidual
    from agentfem.mechanics._finite_hex_energy import FiniteHexEnergyMonitor

    u, law, response, internal = setup(comm, counts)
    history = state.second_order_state(u)
    internal.displacement = history.u.value
    residual = FiniteHexExplicitResidual(
        internal,
        law,
        omega_squared_bound=1e6,
        maximum_negative_growth_per_increment=0.1,
    )
    residual.enable_energy(
        law.initial_array_response(len(response.state.reference_field.values))
    )
    xyz = u.function_space.tabulate_dof_coordinates()
    history.v.value.x.array[:] = np.column_stack(
        (2 * xyz[:, 0] ** 2, 0.1 * xyz[:, 1], -0.1 * xyz[:, 2])
    ).ravel()
    history.v.value.x.scatter_forward()
    ledger = fracture.DynamicEnergyLedger(
        energy=FiniteHexEnergyMonitor(residual),
        state=history,
        mass=internal.mass_diagonal,
        residual=residual,
    )
    return problems.explicit_dynamics(
        state=history,
        integrator=time.explicit.central_difference(state=history, mass=internal),
        residual=residual,
        dt=0.0002,
        steps=100,
        history_monitor=ledger,
        progress=False,
    )


@pytest.mark.parametrize("counts", [(4, 2, 2), (1, 1, 1)])
def test_finite_dynamic_plastic_path_and_energy_match_serial(counts):
    distributed = dynamic_step(MPI.COMM_WORLD, counts)
    distributed.run()
    serial = dynamic_step(MPI.COMM_SELF, counts)
    serial.run()
    function = distributed.state.u.value
    owned = function.function_space.dofmap.index_map.size_local
    xyz = function.function_space.tabulate_dof_coordinates()[:owned]
    reference_xyz = serial.state.u.value.function_space.tabulate_dof_coordinates()
    for name in ("u", "v", "a"):
        values = (
            getattr(distributed.state, name).value.x.array[: 3 * owned].reshape(-1, 3)
        )
        reference = getattr(serial.state, name).value.x.array.reshape(-1, 3)
        for point, value in zip(xyz, values):
            selected = np.flatnonzero(
                np.linalg.norm(reference_xyz - point, axis=1) < 1e-12
            )
            np.testing.assert_allclose(
                value,
                reference[selected[0]],
                rtol=1e-9,
                # Acceleration divides force cancellation by a small nodal
                # mass; use the field scale for nominally zero components.
                atol=1e-11 * max(1.0, float(np.max(np.abs(reference)))),
            )
    for key in (
        "bulk_stored_energy",
        "hourglass_energy",
        "material_dissipation",
        "kinetic_energy",
        "energy_balance_error",
    ):
        assert distributed.history_records[-1][key] == pytest.approx(
            serial.history_records[-1][key], rel=1e-9, abs=1e-13
        )
    assert distributed.history_records[-1]["material_dissipation"] > 0
    assert distributed.residual.last_spectrum == pytest.approx(
        serial.residual.last_spectrum, rel=1e-10
    )


def test_distributed_post_commit_failure_restores_energy_state(monkeypatch):
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("Collective failure requires multiple ranks")
    step = dynamic_step(MPI.COMM_WORLD)
    step.run(until_step=30)
    before = step.residual.snapshot()
    nodal = step.state.snapshot()
    original = step.residual.commit

    def fail():
        original()
        if MPI.COMM_WORLD.rank == 1:
            raise RuntimeError("injected finite post-commit failure")

    monkeypatch.setattr(step.residual, "commit", fail)
    with pytest.raises(RuntimeError, match="post-commit|another rank"):
        step.run()
    assert step.residual.snapshot() == before
    for name, values in nodal["fields"].items():
        np.testing.assert_array_equal(step.state.snapshot()["fields"][name], values)


@pytest.mark.parametrize("counts", [(4, 2, 2), (1, 1, 1)])
def test_distributed_finite_checkpoint_matches_continuous(tmp_path, counts):
    from pathlib import Path
    import json

    if MPI.COMM_WORLD.size < 2:
        pytest.skip("MPI boundary")
    directory = Path(MPI.COMM_WORLD.bcast(str(tmp_path), root=0))
    reference = dynamic_step(MPI.COMM_WORLD, counts)
    reference.run(until_step=24)
    step = dynamic_step(MPI.COMM_WORLD, counts)
    step.run(until_step=13)
    path = step.save_checkpoint(directory / "finite")
    assert json.loads(path.read_text())["schema"] == "agentfem.transient-checkpoint.v7"
    resumed = dynamic_step(MPI.COMM_WORLD, counts)
    resumed.load_checkpoint(path)
    resumed.run(until_step=24)
    assert resumed.residual.snapshot() == reference.residual.snapshot()
    assert resumed.history_records == reference.history_records


@pytest.mark.parametrize("corruption", ["checksum", "identity", "rank_swap"])
def test_corrupt_rank_auxiliary_refuses_restore_atomically(tmp_path, corruption):
    from pathlib import Path
    import json

    if MPI.COMM_WORLD.size < 2:
        pytest.skip("MPI boundary")
    directory = Path(MPI.COMM_WORLD.bcast(str(tmp_path), root=0))
    source = dynamic_step(MPI.COMM_WORLD)
    source.run(until_step=13)
    path = source.save_checkpoint(directory / "corrupt")
    if MPI.COMM_WORLD.rank == 0:
        metadata = json.loads(path.read_text())
        if corruption == "checksum":
            metadata["auxiliary_by_rank"][1]["arrays"]["sha256"] = "0" * 64
        elif corruption == "rank_swap":
            metadata["auxiliary_by_rank"][0], metadata["auxiliary_by_rank"][1] = (
                metadata["auxiliary_by_rank"][1],
                metadata["auxiliary_by_rank"][0],
            )
        else:
            tree = metadata["auxiliary_by_rank"][1]["tree"]
            residual = dict(tree["items"])["residual"]
            dict(residual["items"])["partition_identity"]["value"] = "invalid"
        path.write_text(json.dumps(metadata))
    MPI.COMM_WORLD.barrier()
    target = dynamic_step(MPI.COMM_WORLD)
    before = target.residual.snapshot()
    nodal = target.state.snapshot()
    with pytest.raises((RuntimeError, ValueError), match="checksum|identity"):
        target.load_checkpoint(path)
    assert target.residual.snapshot() == before
    assert target.completed_steps == 0
    for name, values in nodal["fields"].items():
        np.testing.assert_array_equal(target.state.snapshot()["fields"][name], values)


def test_rank_auxiliary_write_failure_preserves_published_generation(
    tmp_path, monkeypatch
):
    from pathlib import Path
    from agentfem import checkpointing

    if MPI.COMM_WORLD.size < 2:
        pytest.skip("MPI boundary")
    directory = Path(MPI.COMM_WORLD.bcast(str(tmp_path), root=0))
    source = dynamic_step(MPI.COMM_WORLD)
    source.run(until_step=13)
    path = source.save_checkpoint(directory / "atomic")
    before = path.read_bytes()
    existing = set(directory.iterdir())
    original = checkpointing.atomic_savez

    def fail(selected, **arrays):
        if MPI.COMM_WORLD.rank == 1 and str(selected).endswith(".auxiliary.npz"):
            raise OSError("injected rank auxiliary write failure")
        return original(selected, **arrays)

    monkeypatch.setattr(checkpointing, "atomic_savez", fail)
    with pytest.raises(RuntimeError, match="auxiliary write failure"):
        source.save_checkpoint(path)
    MPI.COMM_WORLD.barrier()
    assert path.read_bytes() == before
    assert set(directory.iterdir()) == existing
    restored = dynamic_step(MPI.COMM_WORLD)
    restored.load_checkpoint(path)
    assert restored.residual.snapshot() == source.residual.snapshot()


@pytest.mark.parametrize("where", ["spectrum", "accepted_energy"])
def test_rank_local_diagnostic_failure_does_not_desynchronize(monkeypatch, where):
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("MPI boundary")
    step = dynamic_step(MPI.COMM_WORLD)
    step.run(until_step=10)
    before = step.residual.snapshot()
    if MPI.COMM_WORLD.rank == 1:

        def fail(*args, **kwargs):
            raise ValueError("injected local diagnostic failure")

        if where == "spectrum":
            monkeypatch.setattr(step.residual, "_screen_trial", fail)
        else:
            monkeypatch.setattr(step.residual, "require_accepted_configuration", fail)
    with pytest.raises((RuntimeError, ValueError), match="diagnostic failure"):
        if where == "spectrum":
            step.run()
        else:
            step.history_monitor.evaluate(
                displacement=step.state.u, velocity=step.state.v
            )
    assert step.residual.snapshot() == before
