"""Field sampling never changes the accepted constitutive time partition."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from agentfem import checkpointing, results, steps
from test_finite_strain_material_step import HistoryMaterial, build


def times(path):
    return [float(item.attrib["Value"]) for item in ET.parse(path).findall(".//Time")]


@pytest.mark.parametrize("value", [True, False, np.bool_(True), 0, -1, 2.0, "2", float("nan")])
def test_invalid_cadence_is_rejected(value):
    with pytest.raises(ValueError, match="output_every"):
        build(output_every=value)


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
@pytest.mark.parametrize("every, expected", [(1, [0.5, 1, 1.5, 2]), (3, [1.5, 2]), (None, [2]), (10, [2])])
def test_sampling_changes_only_saved_fields(tmp_path, monkeypatch, mechanism, every, expected):
    control, _ = build(mechanism=mechanism)
    control.solve()
    step, _ = build(
        mechanism=mechanism,
        output_every=every,
        checkpoint=checkpointing.every(1, directory=tmp_path / "restart"),
    )
    recoveries = []
    original = step._visual_fields

    def recover(**kwargs):
        recoveries.append((step.time, kwargs.get("include_energy", True)))
        return original(**kwargs)

    monkeypatch.setattr(step, "_visual_fields", recover)
    output = tmp_path / "results.xdmf"
    result = step.solve_result(output=output)
    assert times(output) == expected
    assert recoveries == [(time, False) for time in expected] + [(2, True)]
    assert step.accepted_increments == control.accepted_increments
    assert step.attempted_increments == control.attempted_increments
    np.testing.assert_array_equal(step.solution.x.array, control.solution.x.array)
    np.testing.assert_array_equal(step.state.committed_state_vectors(), control.state.committed_state_vectors())
    np.testing.assert_array_equal(step.reaction_field().x.array, control.reaction_field().x.array)
    assert len(step.checkpoints) == 4
    assert result.metadata["field_output"]["frame_times"] == expected
    assert result.metadata["field_output"]["frame_count"] == len(expected)
    assert result.metadata["field_output"]["output_every"] == every
    assert not result.metadata["field_output"]["continuation_boundary_published"]
    with pytest.raises(AttributeError):
        step.output_every = 9


@pytest.mark.parametrize("restart", [False, True])
def test_pause_resume_publishes_boundary_and_preserves_global_cadence(tmp_path, restart):
    first, _ = build(output_every=3)
    first_path, second_path = tmp_path / "first.xdmf", tmp_path / "second.xdmf"
    first.solve_result(output=first_path, until=0.5)
    assert times(first_path) == [1]
    if restart:
        saved = first.save_checkpoint(tmp_path / "checkpoint")
        second, _ = build(output_every=3)
        second.load_checkpoint(saved)
    else:
        second = first
    result = second.solve_result(output=second_path)
    assert times(second_path) == [1, 1.5, 2]
    assert result.metadata["field_output"]["continuation_boundary_published"]
    assert result.metadata["field_output"]["restored_boundary_published"] == restart
    joined = results.join_xdmf_segments([first_path, second_path], tmp_path / "joined.xdmf")
    assert times(joined) == [1, 1.5, 2]
    control, _ = build()
    control.solve()
    np.testing.assert_array_equal(second.state.committed_state_vectors(), control.state.committed_state_vectors())


def test_restart_can_change_output_cadence_without_changing_identity(tmp_path):
    first, _ = build(output_every=None)
    first.solve(until=0.5)
    checkpoint = first.save_checkpoint(tmp_path / "checkpoint")
    second, _ = build(output_every=np.int64(1))
    second.load_checkpoint(checkpoint)
    output = tmp_path / "results.xdmf"
    second.solve_result(output=output)
    assert times(output) == [1, 1.5, 2]
    endpoint = tmp_path / "endpoint.xdmf"
    second.solve_result(output=endpoint)
    assert times(endpoint) == [2]


def test_rejected_attempts_do_not_count_as_output_increments(tmp_path):
    class Limited(HistoryMaterial):
        def update(self, point):
            return replace(super().update(point), suggested_time_scale=0.5 if point.time_increment > 0.3 else 1.0)

    policy = steps.automatic(initial=0.5, minimum=0.001, maximum=0.5)
    step, _ = build(material=Limited(), incrementation=policy, output_every=3)
    path = tmp_path / "results.xdmf"
    step.solve_result(output=path)
    assert any(not item["accepted"] for item in step.attempted_increments)
    expected = [item["time"] for index, item in enumerate(step.accepted_increments, 1) if index % 3 == 0 or item["coordinate"] == 1]
    assert times(path) == expected
    control, _ = build(material=Limited(), incrementation=policy)
    control.solve()
    assert step.accepted_increments == control.accepted_increments
    np.testing.assert_array_equal(step.state.committed_state_vectors(), control.state.committed_state_vectors())


def test_failed_output_keeps_previous_frame_and_committed_restart(tmp_path, monkeypatch):
    from agentfem.results._live_xdmf import LiveXDMFTimeSeries

    original = LiveXDMFTimeSeries._store_frame
    count = 0

    def fail_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("injected output failure")
        return original(*args, **kwargs)

    step, _ = build(output_every=2, checkpoint=checkpointing.every(1, directory=tmp_path / "restart"))
    path = tmp_path / "results.xdmf"
    monkeypatch.setattr(LiveXDMFTimeSeries, "_store_frame", fail_second)
    with pytest.raises(OSError, match="injected output failure"):
        step.solve_result(output=path)
    assert times(path) == [1]
    assert step.accepted_load_factor == 1
    assert len(step.attempted_increments) == 4
    restored, _ = build(output_every=None)
    restored.load_checkpoint(step.checkpoints[-1].path)
    np.testing.assert_array_equal(restored.state.committed_state_vectors(), step.state.committed_state_vectors())
    monkeypatch.setattr(LiveXDMFTimeSeries, "_store_frame", original)
    restored.solve_result(output=tmp_path / "recovered.xdmf")
    assert times(tmp_path / "recovered.xdmf") == [2]


@pytest.mark.parametrize("dimension", [2, 3])
def test_vtk_reads_and_warps_every_saved_frame(tmp_path, dimension):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy

    step, _ = build(dimension=dimension, output_every=3)
    path = tmp_path / "results.xdmf"
    step.solve_result(output=path)
    reader = vtk.vtkXdmfReader()
    reader.SetFileName(str(path))
    reader.UpdateInformation()
    cells = step.state.domain.topology.index_map(dimension).size_local
    for time in times(path):
        reader.UpdateTimeStep(time)
        grid = reader.GetOutputDataObject(0)
        assert grid.IsA("vtkUnstructuredGrid")
        np.testing.assert_allclose(vtk_to_numpy(grid.GetCellData().GetArray("AGE_CELL")), time)
        warp = vtk.vtkWarpVector()
        warp.SetInputData(grid)
        warp.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, step.solution.name)
        warp.Update()
        assert warp.GetOutput().GetNumberOfCells() == cells
        for name in ("S_CELL", "P_CELL", "AGE_CELL"):
            assert warp.GetOutput().GetCellData().GetArray(name) is not None


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_nonuniform_plane_stress_all_saved_fields_match_dense_output(tmp_path, mechanism):
    vtk = pytest.importorskip("vtk")
    from vtk.util.numpy_support import vtk_to_numpy
    from test_history_plane_stress import structure

    readers = []
    states = []
    for label, every in (("dense", 1), ("sparse", 3)):
        step, _ = structure(mechanism=mechanism, nonuniform=True, output_every=every)
        path = tmp_path / f"{label}.xdmf"
        step.solve_result(output=path)
        states.append(step)
        reader = vtk.vtkXdmfReader()
        reader.SetFileName(str(path))
        reader.UpdateInformation()
        readers.append(reader)
    assert states[0].accepted_increments == states[1].accepted_increments
    np.testing.assert_array_equal(states[0]._accepted_F, states[1]._accepted_F)
    np.testing.assert_array_equal(states[0].state.committed_state_vectors(), states[1].state.committed_state_vectors())
    for time in times(tmp_path / "sparse.xdmf"):
        for reader in readers:
            reader.UpdateTimeStep(time)
        grids = [reader.GetOutputDataObject(0) for reader in readers]
        for association in ("GetPointData", "GetCellData"):
            dense, sparse = [getattr(grid, association)() for grid in grids]
            assert dense.GetNumberOfArrays() == sparse.GetNumberOfArrays()
            for index in range(dense.GetNumberOfArrays()):
                name = dense.GetArrayName(index)
                np.testing.assert_array_equal(vtk_to_numpy(dense.GetArray(name)), vtk_to_numpy(sparse.GetArray(name)))


def test_arbitrary_pause_endpoint_does_not_depend_on_output_schedule(tmp_path):
    control, _ = build()
    control.solve(until=0.37)
    sparse, _ = build(output_every=None)
    result = sparse.solve_result(output=tmp_path / "pause.xdmf", until=0.37)
    assert result.status == "partial"
    assert times(tmp_path / "pause.xdmf") == [0.74]
    assert sparse.accepted_increments == control.accepted_increments
    np.testing.assert_array_equal(sparse.state.committed_state_vectors(), control.state.committed_state_vectors())


def test_cli_output_option_is_normal_project_configuration(tmp_path, capsys):
    from agentfem import cli

    source = Path(__file__).parents[1] / "examples/history_material"
    for name in ("case.py", "material.py", "agentfem.toml", "README.md"):
        shutil.copy2(source / name, tmp_path / name)
    assert cli.main(["check", "--project", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    assert cli.main(["run", "--project", str(tmp_path), "--name", "sparse", "--", "--output-every", "4"]) == 0
    run = tmp_path / "outputs/001-sparse"
    record = json.loads((run / "result.json").read_text())
    assert times(run / "results.xdmf") == [0.5, 1, 1.5, 2]
    assert record["metadata"]["solve"]["accepted_increments"] == 16
    assert record["metadata"]["field_output"]["frame_count"] == 4
