# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Accepted co-located data through the ordinary finite-history Step."""

import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from agentfem import results, steps, units
from test_history_material_temperature import build


POINTS = (results.material_point("a", cell=0), results.material_point("b", cell=0, point=1))


@pytest.mark.parametrize("assumption", [None, "plane_strain", "plane_stress"])
def test_point_tensors_and_temperature_are_raw_accepted_data(assumption, tmp_path):
    step, _ = build(assumption=assumption, constrain=True, material_points=POINTS,
                    unit_system=units.n_mm_mpa())
    result = step.solve_result()
    identity = result.metadata["material_point_history"]
    assert identity["scientific_identity"]["point_history_units"]["base"]["length"] == "mm"
    for prefix, index in (("a", 0), ("b", 1)):
        histories = result.histories
        F = histories[prefix + ".F"].values
        E = histories[prefix + ".E_GL"].values
        np.testing.assert_allclose(E, .5 * (np.einsum("nki,nkj->nij", F, F) - np.eye(3)))
        np.testing.assert_allclose(histories[prefix + ".TEMP"].values,
                                   [step._temperature_input.at(r["time"]) for r in step.accepted_increments])
        np.testing.assert_array_equal(histories[prefix + ".S"].latest, step.response.cauchy_stress.values[index])
        np.testing.assert_array_equal(histories[prefix + ".P"].latest, step.response.first_piola_stress.values[index])
        np.testing.assert_array_equal(histories[prefix + ".F"].latest, step._accepted_F[index])
        S = histories[prefix + ".S"].values
        expected_P = np.linalg.det(F)[:, None, None] * S @ np.linalg.inv(F).transpose(0, 2, 1)
        np.testing.assert_allclose(histories[prefix + ".P"].values, expected_P, atol=1.e-12)
        np.testing.assert_allclose(histories[prefix + ".EXPOSURE"].latest, 700.)
        assert histories[prefix + ".S"].unit == units.n_mm_mpa().stress
        assert histories[prefix + ".S"].abscissa_unit == "s"
        assert len(F) == len(step.accepted_increments)
        assert histories[prefix + ".S"].abscissa[0] > 0.
    manifest = result.write_manifest(tmp_path / "result.json", include_histories=True, history_storage="npz")
    loaded = {h.name: h for h in results.read_histories(manifest)}
    np.testing.assert_array_equal(loaded["a.F"].values, result.histories["a.F"].values)


def test_restart_preserves_complete_point_history(tmp_path):
    full, _ = build(material_points=POINTS, assumption="plane_stress")
    full.solve()
    partial, _ = build(material_points=POINTS, assumption="plane_stress")
    partial.solve(until=.5)
    path = partial.save_checkpoint(tmp_path / "partial")
    resumed, _ = build(material_points=POINTS, assumption="plane_stress")
    resumed.load_checkpoint(path)
    resumed.solve()
    for expected, actual in zip(full.material_point_histories, resumed.material_point_histories, strict=True):
        assert expected.name == actual.name
        np.testing.assert_array_equal(actual.abscissa, expected.abscissa)
        np.testing.assert_array_equal(actual.values, expected.values)


@pytest.mark.parametrize("mutation", ["missing", "value", "time", "late_field_failure"])
def test_corrupt_restore_leaves_existing_history_and_fields_unchanged(tmp_path, mutation):
    source, _ = build(material_points=POINTS)
    source.solve(until=.5)
    path = source.save_checkpoint(tmp_path / "source")
    with np.load(path, allow_pickle=False) as data:
        payload = {k: data[k].copy() for k in data.files}
    if mutation == "missing":
        del payload["material_point_history"]
    elif mutation == "late_field_failure":
        payload["response_TEMP"] += 1.
    else:
        history = json.loads(payload["material_point_history"].item())
        if mutation == "value":
            history["frames"][0]["values"]["TEMP"][0] += 1.
        else:
            history["frames"][0]["time"] += .01
        payload["material_point_history"] = json.dumps(history)
    bad = tmp_path / "bad.npz"
    np.savez(bad, **payload)
    target, _ = build(material_points=POINTS)
    target.solve(until=.25)
    before = target._snapshot()
    history_before = target._point_history.payload()
    with pytest.raises(ValueError):
        target.load_checkpoint(bad)
    assert target._point_history.payload() == history_before
    assert target.accepted_load_factor == .25
    np.testing.assert_array_equal(target.solution.x.array, before["U"])
    for key, field in target._responses.items():
        np.testing.assert_array_equal(field.x.array, before["response"][key])


def test_selection_change_or_omission_rejects_restart(tmp_path):
    source, _ = build(material_points=POINTS)
    source.solve(until=.5)
    path = source.save_checkpoint(tmp_path / "state")
    for selection in ((), (POINTS[0],), (results.material_point("different", cell=0),)):
        target, _ = build(material_points=selection)
        with pytest.raises(ValueError, match="identity"):
            target.load_checkpoint(path)
    source.material_points = ()
    with pytest.raises(ValueError, match="selection"):
        source.solve()


def test_failed_commit_adds_no_history_and_retry_matches(monkeypatch):
    step, _ = build(material_points=POINTS)
    step.solve(until=.5)
    before = step._point_history.payload()
    commit = step.response.commit
    def fail():
        commit()
        raise RuntimeError("injected commit failure")
    monkeypatch.setattr(step.response, "commit", fail)
    with pytest.raises(RuntimeError, match="commit failure"):
        step.solve()
    assert step._point_history.payload() == before
    assert step.accepted_load_factor == .5
    monkeypatch.setattr(step.response, "commit", commit)
    step.solve()
    full, _ = build(material_points=POINTS)
    full.solve()
    assert step._point_history.payload() == full._point_history.payload()


def test_cutbacks_do_not_add_rejected_samples():
    step, _ = build(material_points=POINTS, limit=.21,
                    increments=steps.automatic(initial=.5, minimum=.005, maximum=.5))
    step.solve()
    assert len(step.attempted_increments) > len(step.accepted_increments)
    times = [r["time"] for r in step.accepted_increments]
    np.testing.assert_array_equal(step.material_point_histories[0].abscissa, times)


@pytest.mark.parametrize("selection", [(results.material_point("bad", cell=1000),),
                                       (results.material_point("bad", cell=0, point=1000),),
                                       (POINTS[0], POINTS[0]), ("bad",)])
def test_invalid_selection_rejected_at_construction(selection):
    with pytest.raises((ValueError, TypeError)):
        build(material_points=selection)


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_no_temperature_materials_preserve_unknown_units_and_cell_identity(mechanism):
    from test_finite_strain_material_step import build as mechanical_build
    selection = (results.material_point("first", cell=0), results.material_point("last", cell=5, point=1))
    step, _ = mechanical_build(mechanism=mechanism, material_points=selection)
    result = step.solve_result()
    sampling = step.response.cauchy_stress
    for request in selection:
        row = int(np.flatnonzero(sampling.owned_cell_keys == request.cell)[0])
        index = row * len(sampling.points) + request.point
        actual = result.histories[request.name + ".S"]
        np.testing.assert_array_equal(actual.latest, sampling.values[index])
        assert actual.unit is None
        assert request.name + ".TEMP" not in result.histories
        assert result.histories[request.name + ".F"].abscissa_unit is None


def test_failed_frame_preparation_precedes_material_commit(monkeypatch):
    step, _ = build(material_points=POINTS)
    step.solve(until=.5)
    before = step._snapshot()
    history = step._point_history.payload()
    def fail(*args, **kwargs):
        raise ValueError("injected observation error")
    monkeypatch.setattr(step._point_history, "prepare", fail)
    with pytest.raises(ValueError, match="observation error"):
        step.solve()
    assert step._point_history.payload() == history
    for key, values in step.state.snapshot().items():
        np.testing.assert_array_equal(values, before["state"][key])
    assert step.accepted_load_factor == .5


def test_cli_point_history_is_independent_of_sparse_field_output(tmp_path):
    from agentfem import cli
    source = Path(__file__).parents[1] / "examples/history_material"
    for name in ("case.py", "material.py", "agentfem.toml", "README.md"):
        shutil.copy2(source / name, tmp_path / name)
    assert cli.main(["check", "--project", str(tmp_path)]) == 0
    assert cli.main(["run", "--project", str(tmp_path), "--name", "points", "--",
                     "--material", "damage", "--temperature-cycle", "--material-point",
                     "--assumption", "plane_stress", "--output-every", "4"]) == 0
    run = tmp_path / "outputs/001-points"
    histories = {h.name: h for h in results.read_histories(run / "result.json")}
    record = json.loads((run / "result.json").read_text())
    frames = ET.parse(run / "results.xdmf").findall(".//Time")
    assert len(frames) < len(histories["sample.F"].abscissa)
    assert len(histories["sample.F"].abscissa) == record["metadata"]["solve"]["accepted_increments"]
    assert "sample.TEMP" in histories


def test_selected_recorder_matches_full_point_selection_and_checks_count():
    step, _ = build(material_points=POINTS)
    step.solve(until=.5)
    recorder = step._point_history
    values = {name: np.zeros((len(step._accepted_F), *channel["shape"]))
              for name, channel in recorder.channels.items()}
    for array in values.values():
        array[...] = np.arange(len(array)).reshape((-1,) + (1,) * (array.ndim-1))
    expected = recorder.prepare(1.5, values)
    selected = {name: array[recorder.indices] for name, array in values.items()}
    assert recorder.prepare(1.5, selected, selected=True) == expected
    selected["F"] = selected["F"][:1]
    with pytest.raises(ValueError, match="point count"):
        recorder.prepare(1.5, selected, selected=True)


def test_nonuniform_structure_preserves_each_input_cell_point():
    from mpi4py import MPI
    from agentfem import fields, mesh, models, studies
    from test_finite_strain_material_step import HistoryMaterial

    domain = mesh.rectangle((0., 0.), (1., 1.), (2, 2), cell_type="quadrilateral", comm=MPI.COMM_SELF)
    model = models.create(study=studies.static_solid(dimension=2, assumption="plane_strain", nonlinear=True),
                          mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(HistoryMaterial())
    model.clamp(u, on=mesh.face(domain, axis="x", value=0.))
    model.fix(u, on=mesh.face(domain, axis="x", value=1.), component=0, value=.08)
    selection = tuple(results.material_point(f"cell{cell}", cell=cell, point=cell % 4) for cell in range(4))
    step = model.step(target=u, duration=1., material_points=selection,
                      incrementation=steps.fixed(3), progress=False)
    expected = {item.name: [] for item in selection}
    def observe(accepted):
        field = accepted.response.cauchy_stress
        for item in selection:
            row = int(np.flatnonzero(field.owned_cell_keys == item.cell)[0])
            expected[item.name].append(field.values[row * len(field.points) + item.point].copy())
    step.solve(on_accept=observe)
    histories = {h.name: h for h in step.material_point_histories}
    for item in selection:
        np.testing.assert_array_equal(histories[item.name + ".S"].values, expected[item.name])
    assert np.ptp([history[-1][0, 0] for history in expected.values()]) > 1.e-4


def test_history_reads_are_detached():
    step, _ = build(material_points=POINTS)
    assert all(len(h.abscissa) == 0 for h in step.material_point_histories)
    step.solve(until=.5)
    original = step._point_history.payload()
    for history in step.material_point_histories:
        history.values[...] = -99.
        history.abscissa[...] = -99.
    assert step._point_history.payload() == original


@pytest.mark.parametrize("assumption", [None, "plane_strain", "plane_stress"])
def test_observation_does_not_change_solution_or_material_path(assumption):
    plain, _ = build(assumption=assumption, constrain=True)
    observed, _ = build(assumption=assumption, constrain=True, material_points=POINTS)
    plain.solve()
    observed.solve()
    np.testing.assert_array_equal(observed.solution.x.array, plain.solution.x.array)
    np.testing.assert_array_equal(observed._accepted_F, plain._accepted_F)
    for name, values in plain.state.snapshot().items():
        np.testing.assert_array_equal(observed.state.snapshot()[name], values)
    assert observed.accepted_load_factor == plain.accepted_load_factor
    assert [r["time"] for r in observed.accepted_increments] == [r["time"] for r in plain.accepted_increments]
