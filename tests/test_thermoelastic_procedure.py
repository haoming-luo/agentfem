# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import json
from pathlib import Path
import pytest
import ufl
from dolfinx import fem, mesh as dmesh
from mpi4py import MPI

from agentfem import amplitudes, constraints, constitutive, fields, mesh
from agentfem.time.thermoelastic import _ThermoelasticStep


def make_step(
    *, cells="tetrahedron", alpha=0.002, dilation_rate=None, temperature_rate=None
):
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (2.0, 1.0, 3.0),
        (2, 2, 2),
        cell_type=cells,
        comm=MPI.COMM_WORLD,
    )
    u, theta = fields.displacement(domain), fields.temperature(domain, value=0.0)
    material = constitutive.thermoelastic(
        young=1000.0,
        poisson=0.25,
        density=1.0,
        specific_heat=100.0,
        conductivity=1.0,
        thermal_expansion=alpha,
        reference_temperature=300.0,
    )
    bcs = []
    for axis in range(3):
        facets = dmesh.locate_entities_boundary(
            domain, 2, lambda x, axis=axis: np.isclose(x[axis], 0.0)
        )
        dofs = fem.locate_dofs_topological(u.space.sub(axis), 2, facets)
        bcs.append(fem.dirichletbc(0.0, dofs, u.space.sub(axis)))
        if dilation_rate is not None:
            length = (2.0, 1.0, 3.0)[axis]
            bcs.append(
                constraints.time_dependent_component_dirichlet(
                    u,
                    axis,
                    marker=lambda x, axis=axis, length=length: np.isclose(
                        x[axis], length
                    ),
                    amplitude=amplitudes.ramp(end_value=dilation_rate * length),
                    name=f"motion_{axis}",
                )
            )
    thermal_bcs = []
    if temperature_rate is not None:
        thermal_bcs.append(
            constraints.time_dependent_scalar_dirichlet(
                theta,
                marker=lambda x: np.any(
                    np.isclose(x, 0.0)
                    | np.isclose(x, np.array([2.0, 1.0, 3.0])[:, None]),
                    axis=0,
                ),
                amplitude=amplitudes.ramp(end_value=temperature_rate),
                name="boundary_temperature",
            )
        )
    return _ThermoelasticStep(
        u,
        theta,
        material=material,
        dt=0.1,
        steps=3,
        mechanical_bcs=bcs,
        thermal_bcs=thermal_bcs,
        heat_load=fem.Constant(domain, 10.0) * theta.test * ufl.dx,
        input_identity={
            "geometry": [2, 1, 3],
            "source": 10,
            "boundaries": "lower-face rollers, insulated",
            "initial": "zero",
            "dilation_rate": dilation_rate,
            "temperature_rate": temperature_rate,
        },
    )


@pytest.mark.parametrize("cells", ["tetrahedron", "hexahedron"])
def test_geometry_independent_procedure_matches_closed_form(cells):
    with make_step(cells=cells) as step:
        step.run()
        bulk = 1000 / (3 * (1 - 2 * 0.25))
        expected = 10 * 0.3 / (100 + 9 * bulk * 0.002**2 * 300)
        np.testing.assert_allclose(step.theta.value.x.array, expected, atol=2e-10)
        np.testing.assert_allclose(
            step.u.value.x.array.reshape(-1, 3),
            0.002 * expected * step.u.space.tabulate_dof_coordinates(),
            atol=2e-12,
        )
        result = step.result()
        assert set(result.metadata["matrix_assemblies"].values()) == {1}
        assert not hasattr(step, "reference")
        assert step.completed_steps == 3
        assert max(item["quadratic_balance_absolute"] for item in step.history) < 1e-10
        assert (
            max(item["linearized_heat_balance_absolute"] for item in step.history)
            < 2e-8
        )
        with pytest.raises(RuntimeError, match="AFM-COUPLING-004"):
            step.advance()


def test_procedure_rejects_without_advancing_or_polluting_accepted_state():
    with make_step() as step:
        step.advance()
        snapshots = [
            f.x.array.copy()
            for f in (step.u.value, step.theta.value, step.u_old, step.theta_old)
        ]

        def reject(record):
            record["time"] = -999  # acceptance observer cannot mutate history
            if step.comm.rank == 0:
                raise RuntimeError("injected rejection")

        with pytest.raises(RuntimeError, match="injected rejection"):
            step.advance(acceptance_check=reject)
        assert step.completed_steps == len(step.history) == 1
        for value, snapshot in zip(
            (step.u.value, step.theta.value, step.u_old, step.theta_old), snapshots
        ):
            np.testing.assert_array_equal(value.x.array, snapshot)
        step.run()
        assert step.completed_steps == 3


def test_returned_record_is_not_live_accepted_history():
    with make_step(alpha=0.0) as step:
        record = step.advance()
        record["residuals"].clear()
        assert step.history[0]["residuals"]
        assert step.history[0]["outer_iterations"] == 1


def test_closed_procedure_cannot_reuse_destroyed_petsc_resources():
    step = make_step()
    step.close()
    step.close()
    with pytest.raises(RuntimeError, match="AFM-COUPLING-004"):
        step.advance()


def test_joint_procedure_restart_and_corrupt_history_are_atomic(tmp_path):
    comm = MPI.COMM_WORLD
    path = comm.bcast(str(tmp_path / "coupled"), root=0)
    with make_step() as original, make_step() as restarted:
        original.advance()
        manifest = original.save_checkpoint(path)
        original.run()
        restarted.load_checkpoint(path)
        assert restarted.completed_steps == 1
        restarted.run()
        np.testing.assert_allclose(
            restarted.u.value.x.array, original.u.value.x.array, atol=1e-12
        )
        np.testing.assert_allclose(
            restarted.theta.value.x.array, original.theta.value.x.array, atol=1e-10
        )
        assert restarted.result().metadata["restart_source"]["accepted_step"] == 1
        snapshots = (
            restarted.u.value.x.array.copy(),
            restarted.theta.value.x.array.copy(),
        )
        if comm.rank == 0:
            data = json.loads(Path(manifest).read_text())
            data["history_records"][0]["outer_iterations"] = -1
            Path(manifest).write_text(json.dumps(data))
        comm.barrier()
        with pytest.raises(ValueError, match="AFM-COUPLING-005"):
            restarted.load_checkpoint(path)
        assert restarted.completed_steps == 3
        for f, snapshot in zip((restarted.u.value, restarted.theta.value), snapshots):
            np.testing.assert_array_equal(f.x.array, snapshot)


def test_registered_motion_work_and_boundary_rollback(tmp_path):
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "moving"), root=0)
    rate = 0.0001
    with make_step(dilation_rate=rate) as step:
        previous_reaction = 0.0
        for _ in range(2):
            record = step.advance()
            theta = (10 - 3 * step.blocks.beta * 300 * rate) * step.time / 100
            reaction = 6 * 3 * (2000 * rate * step.time - step.blocks.beta * theta)
            assert record["prescribed_motion_path_work"] == pytest.approx(
                0.5 * (previous_reaction + reaction) * rate * 0.1, abs=1e-11
            )
            assert record["quadratic_balance_absolute"] < 1e-10
            assert record["linearized_heat_balance_absolute"] < 2e-8
            previous_reaction = reaction
        step.save_checkpoint(path)
        accepted = [float(b.constant.value) for b in step._time_boundaries]

        def reject(_):
            if step.comm.rank == 0:
                raise RuntimeError("reject prescribed motion")

        with pytest.raises(RuntimeError, match="reject prescribed motion"):
            step.advance(acceptance_check=reject)
        assert [float(b.constant.value) for b in step._time_boundaries] == accepted
        expected = step.advance()
        with make_step(dilation_rate=rate) as resumed:
            resumed.load_checkpoint(path)
            assert [
                float(b.constant.value) for b in resumed._time_boundaries
            ] == accepted
            actual = resumed.advance()
            assert actual["prescribed_motion_path_work"] == pytest.approx(
                expected["prescribed_motion_path_work"], abs=1e-11
            )


@pytest.mark.parametrize("rate", [-0.2, 0.2])
def test_registered_temperature_boundary_has_signed_heat_reaction(rate):
    with make_step(temperature_rate=rate) as step:
        for _ in range(3):
            record = step.advance()
            assert record["prescribed_temperature_heat_input"] * rate > 0
            assert record["linearized_heat_balance_absolute"] < 2e-8
            assert record["quadratic_balance_absolute"] < 1e-10
