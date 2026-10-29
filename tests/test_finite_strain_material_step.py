"""Material-only providers through the same physical-time structural Step."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from dolfinx import mesh as dmesh
from mpi4py import MPI

from agentfem import amplitudes, constitutive as c, fields, mesh, models, steps, studies

spec = importlib.util.spec_from_file_location(
    "history_example_material",
    Path(__file__).parents[1] / "examples/history_material/material.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
HistoryMaterial = module.HistoryMaterial


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_four_file_example_runs_through_project_cli(tmp_path, mechanism, capsys):
    from agentfem import cli

    source = Path(__file__).parents[1] / "examples/history_material"
    for name in ("case.py", "material.py", "agentfem.toml", "README.md"):
        shutil.copy2(source / name, tmp_path / name)
    assert cli.main(["check", "--project", str(tmp_path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed" and report["upgrade_findings"] == []
    assert (
        cli.main(
            [
                "run",
                "--project",
                str(tmp_path),
                "--name",
                mechanism,
                "--",
                "--material",
                mechanism,
            ]
        )
        == 0
    )
    run = tmp_path / "outputs" / f"001-{mechanism}"
    record = json.loads((run / "result.json").read_text())
    execution = json.loads((run / "execution.json").read_text())
    assert execution["status"] == "completed"
    assert record["metadata"]["solve"]["accepted_increments"] == 16
    assert record["checkpoint_records"]
    assert record["metadata"]["visualization"]["primary_artifact"] == "field_output"
    for path in record["artifacts"].values():
        assert (run / path).is_file()
    tree = ET.parse(run / "results.xdmf")
    frames = tree.findall(".//Grid[@GridType='Uniform']")
    assert len(frames) == 16
    for frame in frames:
        names = {a.get("Name") for a in frame.findall("Attribute")}
        assert {"Displacement", "S_CELL", "AGE_CELL"} <= names


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_scalar_only_material_has_same_accepted_history(mechanism):
    local = HistoryMaterial(mechanism=mechanism)

    class ScalarOnly:
        name = local.name
        state_schema = local.state_schema
        tangent_convention = local.tangent_convention
        summary = staticmethod(local.summary)
        update = staticmethod(local.update)

    assert isinstance(ScalarOnly(), c.UserMaterial)
    scalar, _ = build(material=ScalarOnly())
    batch, _ = build(material=local)
    scalar.solve()
    batch.solve()
    np.testing.assert_allclose(
        scalar.solution.x.array, batch.solution.x.array, atol=1e-12
    )
    np.testing.assert_allclose(
        scalar.state.committed_state_vectors(),
        batch.state.committed_state_vectors(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        scalar.response.cauchy_stress.values,
        batch.response.cauchy_stress.values,
        atol=1e-12,
    )


def build(
    *,
    mechanism="viscoelastic",
    dimension=3,
    duration=2.0,
    material=None,
    incrementation=None,
    **options,
):
    domain = (
        dmesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
        if dimension == 3
        else dmesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    )
    study = studies.static_solid(
        dimension=dimension,
        nonlinear=True,
        **({"assumption": "plane_strain"} if dimension == 2 else {}),
    )
    model = models.create(study=study, mesh=domain)
    u = model.field(fields.displacement(domain))
    material = material or HistoryMaterial(mechanism=mechanism)
    model.material(material)
    for axis in range(dimension):
        model.fix(u, on=mesh.face(domain, axis="xyz"[axis], value=0), component=axis)
        model.fix(
            u,
            on=mesh.face(domain, axis="xyz"[axis], value=1),
            component=axis,
            value=0.12 if axis == 0 else 0.0,
        )
    options.setdefault("progress", False)
    step = model.step(
        target=u,
        duration=duration,
        incrementation=incrementation or steps.fixed(4),
        amplitude=amplitudes.tabular((0.0, 0.5, 1.0), (0.0, 1.0, 0.2)),
        **options,
    )
    return step, material


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_tangent_and_rotated_stress(mechanism):
    material = HistoryMaterial(mechanism=mechanism)
    F = np.array([[1.1, 0.05, 0], [0, 0.98, 0.01], [0, 0, 1.02]])
    point = c.MaterialPointInput(
        np.eye(3),
        F,
        0.2,
        0.2,
        [],
        np.zeros(material.state_schema.size),
        material.state_schema,
    )
    assert c.check_material_tangent(material, point).accepted
    angle = 0.9
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    original = material.update(point)
    rotated = material.update(
        replace(
            point,
            deformation_gradient_old=R @ point.deformation_gradient_old,
            deformation_gradient_new=R @ F,
        )
    )
    np.testing.assert_allclose(
        rotated.cauchy_stress, R @ original.cauchy_stress @ R.T, atol=1e-12
    )
    np.testing.assert_allclose(rotated.state_new, original.state_new, atol=1e-14)
    expected_tangent = np.einsum(
        "ia,aJbL,kb->iJkL", R, original.consistent_tangent.reshape(3, 3, 3, 3), R
    )
    np.testing.assert_allclose(
        rotated.consistent_tangent, expected_tangent.reshape(9, 9), atol=1e-11
    )


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_material_point_matches_structure(mechanism, dimension):
    step, material = build(mechanism=mechanism, dimension=dimension)
    # Construction must never integrate a fictitious physical time step.
    assert np.max(np.abs(step.state.committed_state_vectors())) == 0
    step.solve()
    oldF, state = np.eye(3), np.zeros(material.state_schema.size)
    for item in step.accepted_increments:
        F = np.diag([1 + 0.12 * step.amplitude(item["coordinate"]), 1.0, 1.0])
        response = material.update(
            c.MaterialPointInput(
                oldF, F, item["time"], item["dt"], [], state, material.state_schema
            )
        )
        oldF, state = F, response.state_new
    np.testing.assert_allclose(
        step.state.committed_state_vectors(),
        np.broadcast_to(state, step.state.committed_state_vectors().shape),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        step.response.cauchy_stress.values,
        np.broadcast_to(
            response.cauchy_stress, step.response.cauchy_stress.values.shape
        ),
        atol=1e-8,
    )
    assert step.time == 2.0
    assert step.accepted_load_factor == 1.0


def test_physical_duration_changes_viscous_response():
    fast, _ = build(duration=0.2)
    slow, _ = build(duration=2.0)
    fast.solve()
    slow.solve()
    assert not np.allclose(
        fast.response.cauchy_stress.values, slow.response.cauchy_stress.values
    )


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_restart_preserves_response_without_reintegrating(tmp_path, mechanism):
    continuous, _ = build(mechanism=mechanism)
    continuous.solve()
    partial, _ = build(mechanism=mechanism)
    partial.solve(until=0.5)
    path = partial.save_checkpoint(tmp_path / "accepted")
    resumed, material = build(mechanism=mechanism)
    original = material.update
    material.update = lambda point: (_ for _ in ()).throw(
        AssertionError("restart must not update material")
    )
    resumed.load_checkpoint(path)
    np.testing.assert_array_equal(
        resumed.response.cauchy_stress.values, partial.response.cauchy_stress.values
    )
    material.update = original
    resumed.solve()
    np.testing.assert_allclose(
        resumed.solution.x.array, continuous.solution.x.array, atol=1e-12
    )
    np.testing.assert_allclose(
        resumed.state.committed_state_vectors(),
        continuous.state.committed_state_vectors(),
        atol=1e-12,
    )
    assert resumed.accepted_increments == continuous.accepted_increments


def test_failure_restores_accepted_responses(tmp_path):
    step, material = build()
    step.solve(until=0.5)
    before = step._snapshot()
    original = material.update
    material.update = lambda point: (_ for _ in ()).throw(
        RuntimeError("forced local rejection")
    )
    with pytest.raises(RuntimeError, match="rejected increment"):
        step.solve()
    material.update = original
    np.testing.assert_array_equal(step.solution.x.array, before["U"])
    for name, values in before["response"].items():
        np.testing.assert_array_equal(step._responses[name].x.array, values)
    assert step.time == 1.0
    step.save_checkpoint(tmp_path / "still_accepted")


@pytest.mark.parametrize(
    "change, message",
    [("gradient", "disagree"), ("translation", "prescribed boundary")],
)
def test_corrupt_response_restore_is_atomic(tmp_path, change, message):
    step, _ = build()
    step.solve(until=0.5)
    path = step.save_checkpoint(tmp_path / "good")
    with np.load(path) as source:
        payload = {name: source[name].copy() for name in source.files}
    if change == "gradient":
        payload["F"] *= 1.01  # Finite, positive but inconsistent with saved U.
    else:
        payload["U"] += 0.1  # Same F, but incompatible with essential BCs.
    corrupt = tmp_path / "corrupt.npz"
    np.savez(corrupt, **payload)
    target, _ = build()
    target.solve(until=0.25)
    before = target._snapshot()
    with pytest.raises(ValueError, match=message):
        target.load_checkpoint(corrupt)
    np.testing.assert_array_equal(target.solution.x.array, before["U"])
    assert target.accepted_load_factor == 0.25


@pytest.mark.parametrize("dimension", [2, 3])
def test_result_output_mixed_fields_at_every_time(tmp_path, dimension):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    step, _ = build(dimension=dimension)
    path = tmp_path / "results.xdmf"
    result = step.solve_result(output=path)
    assert result.fields["S"].location == "quadrature_points"
    reader = vtk.vtkXdmfReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    cells = step.state.domain.topology.index_map(dimension).size_local
    for item in step.accepted_increments:
        reader.UpdateTimeStep(item["time"])
        grid = reader.GetOutputDataObject(0)
        assert grid.IsA("vtkUnstructuredGrid")
        assert grid.GetNumberOfCells() == cells
        assert grid.GetCellData().GetArray("S_CELL") is not None
        np.testing.assert_allclose(
            vtk_to_numpy(grid.GetCellData().GetArray("AGE_CELL")),
            item["time"],
            atol=1e-12,
        )
        displacement = grid.GetPointData().GetArray(step.solution.name)
        assert displacement.GetNumberOfComponents() == 3
        warp = vtk.vtkWarpVector()
        warp.SetInputData(grid)
        warp.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, step.solution.name
        )
        warp.Update()
        assert warp.GetOutput().GetNumberOfCells() == cells
        assert warp.GetOutput().GetCellData().GetArray("S_CELL") is not None
    assert result.fields["S"].processing["stress_measure"] == "cauchy"
    assert (
        result.fields["P_CELL"].processing["stress_measure"] == "first_piola_kirchhoff"
    )
    assert result.fields["AGE_CELL"].unit == "s"
    assert result.fields["RF"].processing["method"] == "assembled_equilibrium_residual"
    assert result.name == "finite_strain_material"
    assert result.metadata["solve"]["converged"]
    final = step.execution_events[-1]
    assert final.kind == "step_completed" and final.increment == final.attempt == 4
    assert final.time == 2.0


def test_material_cutback_subdivides_physical_time_without_pollution():
    class LimitedMaterial(HistoryMaterial):
        def update(self, point):
            response = super().update(point)
            return replace(
                response,
                suggested_time_scale=0.5 if point.time_increment > 0.5 + 1e-12 else 1.0,
            )

    policy = steps.automatic(initial=0.5, minimum=0.01, maximum=0.5)
    step, _ = build(material=LimitedMaterial(), incrementation=policy)
    step.solve()
    assert any(not r["accepted"] for r in step.attempted_increments)
    assert all(r["dt"] <= 0.5 + 1e-12 for r in step.accepted_increments)
    np.testing.assert_allclose(
        step.state.committed_state_vectors()[:, -1], 2.0, atol=1e-13
    )
    coordinates = tuple(r["coordinate"] for r in step.accepted_increments)
    control, _ = build(incrementation=steps.fixed(1))
    for coordinate in coordinates:
        control.solve(until=coordinate)
    np.testing.assert_allclose(
        step.state.committed_state_vectors(),
        control.state.committed_state_vectors(),
        atol=1e-11,
    )


@pytest.mark.parametrize("change", ["duration", "material"])
def test_changed_restart_identity_is_rejected(tmp_path, change):
    source, _ = build()
    source.solve(until=0.5)
    path = source.save_checkpoint(tmp_path / "checkpoint")
    target, _ = build(
        **(
            {"duration": 3.0}
            if change == "duration"
            else {"material": HistoryMaterial(shear=60.0)}
        )
    )
    with pytest.raises(ValueError, match="identity differs"):
        target.load_checkpoint(path)
    assert np.max(np.abs(target.solution.x.array)) == 0


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
@pytest.mark.parametrize("branch", [1.1, 0.9])
@pytest.mark.parametrize("dimension", [2, 3])
def test_nonuniform_dead_load_equilibrium_and_tangent(mechanism, branch, dimension):
    import dolfinx.fem.petsc as fp
    from petsc4py import PETSc

    domain = (
        dmesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2)
        if dimension == 3
        else dmesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    )
    model = models.create(
        study=studies.static_solid(
            dimension=dimension,
            nonlinear=True,
            **({"assumption": "plane_strain"} if dimension == 2 else {}),
        ),
        mesh=domain,
    )
    u = model.field(fields.displacement(domain))
    model.material(HistoryMaterial(mechanism=mechanism))
    model.fix(u, on=mesh.face(domain, axis="x", value=0.0))
    applied = (2.0, 0.3, 0.0)[:dimension]
    model.traction(applied, on=mesh.face(domain, axis="x", value=1.0))
    step = model.step(
        target=u, duration=1.0, incrementation=steps.fixed(4), progress=False
    )
    step.solve(until=0.5)
    assert np.std(step.response.cauchy_stress.values[:, 0, 0]) > 1e-3
    snapshot = step._snapshot()
    # Compare total assembled tangent against finite differences at a fixed
    # committed state, on free DOFs (nonuniform strains, nonzero shear).
    step._apply_loading(0.75)
    # Test each smooth branch, not a centered difference across max-history's
    # loading/unloading kink at precisely the committed deformation.
    base = branch * snapshot["U"]
    step.solution.x.array[:] = base
    step.solution.x.scatter_forward()
    evaluation = step._evaluate(0.75, 0.5)
    evaluation.residual.destroy()
    matrix = fp.assemble_matrix(step.tangent_form, bcs=step.bcs)
    matrix.assemble()
    direction = matrix.createVecRight()
    action = matrix.createVecLeft()
    direction.array[:] = np.random.default_rng(42).normal(size=direction.getLocalSize())
    fixed = np.unique(np.concatenate([bc.dof_indices()[0] for bc in step.bcs]))
    # Public Model constraints already expose scalar component DOF indices.
    direction.array[fixed] = 0
    matrix.mult(direction, action)
    h = 1e-7
    residuals = []
    for sign in (1, -1):
        step.solution.x.array[:] = base + sign * h * direction.array_r
        step.solution.x.scatter_forward()
        e = step._evaluate(0.75, 0.5)
        residuals.append(e.residual.array_r.copy())
        e.residual.destroy()
    derivative = -(residuals[0] - residuals[1]) / (2 * h)
    derivative[fixed] = 0
    np.testing.assert_allclose(derivative, action.array_r, atol=2e-5, rtol=2e-6)
    direction.destroy()
    action.destroy()
    matrix.destroy()
    step._restore(snapshot)
    step.solve()
    reaction = fp.assemble_vector(step.residual_form)
    reaction.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    values = reaction.array_r.reshape(-1, dimension)
    np.testing.assert_allclose(np.sum(values, axis=0), -np.asarray(applied), atol=1e-9)
    reaction.destroy()


def test_configured_output_and_continuation(tmp_path):
    import xml.etree.ElementTree as ET

    step, _ = build(dimension=2)
    # The common execution context owns model.step(output=...), not the law.
    from dataclasses import replace

    path = tmp_path / "continued.xdmf"
    step.execution_context = replace(
        step.execution_context,
        policy=replace(step.execution_context.policy, output=path),
    )
    step.solve(until=0.5)
    result = step.solve_result()
    times = [float(t.get("Value")) for t in ET.parse(path).findall(".//Time")]
    assert times == [1.5, 2.0]
    assert result.metadata["field_output"]["output_scope"] == "continuation_segment"
    assert (
        result.checkpoint_contract["capabilities"]["rank_count_portability"]
        == "unsupported"
    )
    assert len(result.artifacts) == 4  # index + shared mesh + two closed frames
    assert all(path.is_file() for path in result.artifacts.values())


def test_fixed_path_lands_on_declared_reversal_knots():
    step, _ = build(incrementation=steps.fixed(1))
    step.solve()
    assert [r["coordinate"] for r in step.accepted_increments] == [0.5, 1.0]


def test_output_failure_keeps_restartable_accepted_state(tmp_path, monkeypatch):
    from agentfem.results._live_xdmf import LiveXDMFTimeSeries

    step, _ = build()
    original = LiveXDMFTimeSeries._store_frame

    def fail_second(writer, *args):
        if writer._frame_count == 1:
            raise OSError("injected disk failure")
        return original(writer, *args)

    monkeypatch.setattr(LiveXDMFTimeSeries, "_store_frame", fail_second)
    with pytest.raises(OSError, match="disk failure"):
        step.solve_result(output=tmp_path / "partial.xdmf")
    assert (
        step.accepted_load_factor == 0.5
    )  # Accepted numerical state survives an I/O error.
    checkpoint = step.save_checkpoint(tmp_path / "after_output_failure")
    restored, _ = build()
    restored.load_checkpoint(checkpoint)
    restored.solve()
    np.testing.assert_allclose(restored.state.committed_state_vectors()[:, -1], 2.0)
    import xml.etree.ElementTree as ET

    assert len(ET.parse(tmp_path / "partial.xdmf").findall(".//Time")) == 1


@pytest.mark.parametrize(
    "option, value, error",
    [
        ("duration", 0.0, ValueError),
        ("time_unit", "", ValueError),
        ("temperature", 300.0, TypeError),
    ],
)
def test_invalid_or_unsupported_options_reject(option, value, error):
    with pytest.raises(error):
        build(**{option: value})


def test_wrong_solver_policy_and_nonraising_failure_reject():
    from agentfem import solvers

    with pytest.raises(TypeError, match="newton"):
        build(solver_options=solvers.direct_solver())
    with pytest.raises(NotImplementedError, match="always raises"):
        build(solver_options=solvers.newton(error_if_not_converged=False))


@pytest.mark.parametrize("kind", ["plane_stress", "axisymmetric"])
def test_unsupported_2d_reductions_are_not_inferred(kind):
    domain = dmesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    model = models.create(
        study=studies.static_solid(dimension=2, assumption=kind, nonlinear=True),
        mesh=domain,
    )
    u = model.field(fields.displacement(domain))
    model.material(HistoryMaterial())
    with pytest.raises(NotImplementedError, match="plane strain"):
        model.step(target=u, duration=1.0, progress=False)


@pytest.mark.parametrize("name", ["S", "Q", "DPDF", "S_CELL", "Q_CELL"])
def test_energy_component_alias_cannot_overwrite_response(name):
    material = HistoryMaterial()
    material.stored_energy_component_names = (name,)
    with pytest.raises(ValueError, match="collide"):
        build(material=material)


@pytest.mark.parametrize("alias", ["S_CELL", "AGE_CELL", "Displacement"])
def test_state_alias_cannot_shadow_primary_or_recovered_field(alias):
    material = HistoryMaterial()
    variables = material.state_schema.variables
    material.state_schema = replace(
        material.state_schema,
        variables=(
            replace(
                variables[0],
                output_name=alias,
                initial_value=variables[0].initial_values().reshape(variables[0].shape),
            ),
            variables[1],
        ),
    )
    with pytest.raises(ValueError, match="collide"):
        build(material=material)


def test_optional_energy_does_not_change_live_frame_schema(tmp_path):
    import xml.etree.ElementTree as ET

    class PartialEnergy(HistoryMaterial):
        def update(self, point):
            response = super().update(point)
            return replace(
                response,
                strain_energy_density=None
                if point.time < 1.0
                else response.strain_energy_density,
            )

    step, _ = build(material=PartialEnergy(), time_unit="s")
    result = step.solve_result(output=tmp_path / "changing-energy.xdmf")
    assert "SENER" in result.fields
    assert "SENER_CELL" not in {
        a.get("Name")
        for a in ET.parse(tmp_path / "changing-energy.xdmf").iter("Attribute")
    }
    assert result.histories["load_coordinate"].abscissa_unit == "s"


@pytest.mark.parametrize(
    "change", ["attempt_time", "missing_attempt", "nonfinite_residual"]
)
def test_corrupt_ledger_rejects_without_touching_live_state(tmp_path, change):
    import json

    step, _ = build()
    step.solve(until=0.5)
    path = step.save_checkpoint(tmp_path / "good")
    with np.load(path) as source:
        payload = {name: source[name].copy() for name in source.files}
    ledger = json.loads(str(payload["ledger"].item()))
    if change == "attempt_time":
        ledger["attempted"][0]["time"] += 0.1
    elif change == "missing_attempt":
        ledger["attempted"] = []
    else:
        ledger["attempted"][0]["residual_norm"] = float("nan")
    payload["ledger"] = json.dumps(ledger)
    np.savez(tmp_path / "bad.npz", **payload)
    target, _ = build()
    before = target._snapshot()
    with pytest.raises(ValueError, match="ledger"):
        target.load_checkpoint(tmp_path / "bad.npz")
    np.testing.assert_array_equal(target.solution.x.array, before["U"])
    assert target.time == 0 and target.restored_from is None


def test_restart_provenance_is_published(tmp_path):
    step, _ = build()
    step.solve(until=0.5)
    path = step.save_checkpoint(tmp_path / "state")
    target, _ = build()
    target.load_checkpoint(path)
    result = target.solve_result()
    assert result.metadata["restart"]["coordinate_value"] == 1.0
    assert result.metadata["restart"]["portable"] is False
    assert result.metadata["restart"]["path"] == str(path)


def test_physical_time_refinement_against_independent_relaxation_solution():
    # Independent exact convolution of tau*Qdot + Q = E(t). Along each
    # piecewise-linear stretch segment E(t) is quadratic, so no material
    # update or numerical solver is used to construct the reference.
    tau, exact_q, stretch = 0.4, 0.0, 1.0
    for endpoint in (1.12, 1.024):
        dt, speed = 1.0, endpoint - stretch
        decay = np.exp(-dt / tau)
        a, b, c2 = 0.5 * (stretch**2 - 1), stretch * speed, 0.5 * speed**2
        exact_q = (
            decay * exact_q
            + a * (1 - decay)
            + b * (dt - tau + tau * decay)
            + c2 * (dt**2 - 2 * tau * dt + 2 * tau**2 * (1 - decay))
        )
        stretch = endpoint
    E = 0.5 * (stretch**2 - 1)
    exact_stress = stretch * ((100.0 + 2 * 50.0) * E + 2 * 30.0 * (E - exact_q))
    errors = []
    for count in (8, 16, 32):
        step, _ = build(incrementation=steps.fixed(count))
        step.solve()
        errors.append(
            abs(np.mean(step.response.cauchy_stress.values[:, 0, 0]) - exact_stress)
        )
    ratios = np.asarray(errors[:-1]) / errors[1:]
    assert np.all((ratios > 1.8) & (ratios < 2.2)), (errors, ratios)


def test_pure_rotation_does_not_create_stress_or_damage():
    angle = 1.1
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    for mechanism in ("damage", "viscoelastic"):
        material = HistoryMaterial(mechanism=mechanism)
        response = material.update(
            c.MaterialPointInput(
                np.eye(3),
                R,
                0.1,
                0.1,
                [],
                np.zeros(material.state_schema.size),
                material.state_schema,
            )
        )
        np.testing.assert_allclose(response.cauchy_stress, 0.0, atol=1e-12)
        np.testing.assert_allclose(response.state_new[:-1], 0.0, atol=1e-15)


def test_stateful_provider_never_falls_back_to_linear_elastic_constants():
    from agentfem._step_provider_support import (
        supports_elasticity,
        supports_stateful_constitutive,
    )

    material = HistoryMaterial()
    material.young, material.poisson = 100.0, 0.3
    assert supports_stateful_constitutive(material)
    assert not supports_elasticity(material)


@pytest.mark.parametrize("wrapped", [False, True])
def test_follower_load_is_rejected_before_constitutive_evaluation(wrapped):
    domain = dmesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain
    )
    u = model.field(fields.displacement(domain))
    material = HistoryMaterial()
    material.update = lambda p: (_ for _ in ()).throw(
        AssertionError("must not evaluate")
    )
    model.material(material)
    model.pressure(
        10.0,
        on=mesh.face(domain, axis="x", value=1.0),
        configuration="current",
        displacement=u,
        amplitude=amplitudes.ramp() if wrapped else None,
    )
    with pytest.raises(NotImplementedError, match="FOLLOWER"):
        model.step(target=u, duration=1.0, progress=False)


def test_displacement_dependent_custom_traction_is_not_a_dead_load():
    import ufl

    domain = dmesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain
    )
    u = model.field(fields.displacement(domain))
    model.material(HistoryMaterial())
    model.traction(
        ufl.as_vector((1.0 + u.value[0], 0.0, 0.0)),
        on=mesh.face(domain, axis="x", value=1.0),
    )
    with pytest.raises(NotImplementedError, match="FOLLOWER"):
        model.step(target=u, duration=1.0, progress=False)


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, ValueError])
def test_progress_output_failure_preserves_previous_acceptance(tmp_path, error_type):
    class BrokenReporter:
        def emit(self, event):
            if event.kind == "increment_started" and event.increment == 2:
                raise error_type("progress output failed")

    step, _ = build(progress=BrokenReporter())
    with pytest.raises(error_type, match="progress output"):
        step.solve()
    assert step.accepted_load_factor == 0.25
    assert len(step.attempted_increments) == 1  # No fabricated numerical rejection.
    path = step.save_checkpoint(tmp_path / "accepted")
    restored, _ = build()
    restored.load_checkpoint(path)
    restored.solve()
    np.testing.assert_allclose(restored.state.committed_state_vectors()[:, -1], 2.0)


def test_secondary_reporting_failure_does_not_hide_material_failure():
    class BrokenFailureReporter:
        def emit(self, event):
            if event.kind == "step_failed":
                raise ValueError("secondary reporter problem")

    step, material = build(progress=BrokenFailureReporter())
    material.update = lambda point: (_ for _ in ()).throw(
        RuntimeError("primary material problem")
    )
    with pytest.raises(RuntimeError, match="primary material problem"):
        step.solve()
    assert step.accepted_load_factor == 0
    assert not step._busy and step._at_accepted


@pytest.mark.parametrize("translation", [False, True])
def test_external_displacement_mutation_cannot_become_old_state(translation):
    step, _ = build()
    step.solution.x.array[:] = (
        0.1 if translation else np.arange(len(step.solution.x.array)) * 0.001
    )
    with pytest.raises(ValueError, match="outside"):
        step.solve()
    assert step.time == 0.0


@pytest.mark.parametrize("boundary_kind", ["rigid_contact", "foundation", "follower"])
def test_capability_and_execution_agree_on_excluded_boundaries(boundary_kind):
    from agentfem import boundary_models
    from agentfem.step_providers import step_capability

    domain = dmesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain
    )
    u = model.field(fields.displacement(domain))
    model.material(HistoryMaterial())
    right = mesh.face(domain, axis="x", value=1.0)
    assert step_capability(model, target=u, options={"duration": 1.0})["supported"]
    if boundary_kind == "rigid_contact":
        model.add_boundary_model(
            boundary_models.rigid_contact_pair(
                right,
                boundary_models.rigid_body(
                    boundary_models.rigid_plane(
                        point=(1.005, 0.0, 0.0), normal=(-1.0, 0.0, 0.0)
                    )
                ),
                penalty=1000.0,
                name="tool",
            )
        )
    elif boundary_kind == "foundation":
        model.elastic_foundation(on=right, stiffness=1000.0)
    else:
        model.pressure(1.0, on=right, configuration="current", displacement=u)
    capability = step_capability(model, target=u, options={"duration": 1.0})
    assert not capability["supported"]
    assert capability["rejection_reasons"][0]["message"].startswith("AFM-USER-FINITE")
    with pytest.raises(NotImplementedError, match="AFM-USER-FINITE"):
        model.step(target=u, duration=1.0, progress=False)
    assert not model.steps
