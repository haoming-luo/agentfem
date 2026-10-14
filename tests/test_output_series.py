"""Result sessions preserve accepted values, semantics and dependency integrity."""

from pathlib import Path
import xml.etree.ElementTree as ET
import h5py
import numpy as np
import pytest
from dolfinx import fem
from agentfem import mesh, fields, results


def fixture():
    domain = mesh.rectangle((0.0, 0.0), (1.0, 1.0), (3, 3))
    u = fem.Function(fields.displacement(domain, degree=2).space, name="Velocity")
    q = fem.Function(fem.functionspace(domain, ("DG", 0)), name="Energy")
    return u, q


def test_stream_values_semantics_and_pvd(tmp_path):
    u, q = fixture()
    r = results.SimulationResult(name="custom")
    path = tmp_path / "fields.xdmf"
    with results.output_series(
        r, path, paraview=True, coordinate_name="load_factor"
    ) as writer:
        for i in range(3):
            u.x.array[:] = i
            q.x.array[:] = 10 + i
            writer.write_accepted(coordinate=i, primary=u, fields=(q,))
    assert results.inspect_output(path)["valid"]
    assert results.inspect_output(path.with_suffix(".pvd"))["valid"]
    with h5py.File(path.with_suffix(".h5")) as h5:
        for i in range(3):
            values = h5[f"Frames/{i:04d}/Point/Velocity"][:]
            np.testing.assert_allclose(values[:, :2], i)
            np.testing.assert_allclose(h5[f"Frames/{i:04d}/Cell/Energy"][:], 10 + i)
    record = r.metadata["output_series"][0]
    assert record["warp_field"] is None
    assert record["primary_semantic_name"] == "Velocity"
    assert record["frames"] == 3
    assert not path.with_suffix(".incomplete").exists()
    path.with_suffix(".h5").unlink()
    assert not results.inspect_output(path)["valid"]
    from agentfem.results.series import validate_published_output

    with pytest.raises(ValueError, match="AFM-OUTPUT-INCOMPLETE"):
        validate_published_output(r)


def test_duplicate_coordinate_and_failure_leave_incomplete(tmp_path):
    u, _ = fixture()
    r = results.SimulationResult(name="failure")
    path = tmp_path / "fields.xdmf"
    with pytest.raises(ValueError, match="strictly increasing"):
        with results.output_series(r, path) as writer:
            writer.write_accepted(coordinate=1, primary=u)
            writer.write_accepted(coordinate=1, primary=u)
    assert r.status == "completed_with_output_errors"
    assert path.with_suffix(".incomplete").exists()


def test_empty_and_scalar_only(tmp_path):
    r = results.SimulationResult(name="scalar_only")
    r.add_quantity("energy", 1.0)
    from agentfem.results.series import validate_published_output

    validate_published_output(r)
    with pytest.raises(ValueError, match="no accepted frames"):
        with results.output_series(r, tmp_path / "empty.xdmf"):
            pass


def test_changed_mesh_and_nonfinite_rejected(tmp_path):
    u, _ = fixture()
    r = results.SimulationResult(name="changed")
    with pytest.raises(ValueError, match="changed"):
        with results.output_series(r, tmp_path / "changed.xdmf") as writer:
            writer.write_accepted(coordinate=0, primary=u)
            u.function_space.mesh.geometry.x[:] *= 2
            writer.write_accepted(coordinate=1, primary=u)


def test_scalar_and_non_displacement_result_output(tmp_path):
    u, _ = fixture()
    r = results.SimulationResult(name="velocity")
    r.add_field("Velocity", u, processing={"method": "primary_finite_element_solution"})
    artifacts = results.write_result_fields(r, tmp_path / "velocity.xdmf")
    assert artifacts.warp_field is None
    tree = ET.parse(artifacts.xdmf)
    assert tree.find('.//Attribute[@Name="Velocity"]') is not None


def test_optional_viewer_failure_preserves_scientific_data(tmp_path, monkeypatch):
    from agentfem import io
    from agentfem.results.series import validate_published_output

    def fail(*args, **kwargs):
        raise OSError("injected viewer failure")

    monkeypatch.setattr(io.ParaViewTimeSeries, "write_fields", fail)
    u, _ = fixture()
    r = results.SimulationResult(name="partial")
    path = tmp_path / "partial.xdmf"
    with pytest.warns(RuntimeWarning):
        with results.output_series(r, path, paraview=True, strict=False) as writer:
            writer.write_accepted(coordinate=0, primary=u)
    assert results.inspect_output(path)["valid"]
    assert r.status == "completed_with_output_errors"
    with pytest.warns(RuntimeWarning, match="Presentation output failed"):
        validate_published_output(r)


def test_new_segment_retains_previous_artifacts(tmp_path):
    u, _ = fixture()
    r = results.SimulationResult(name="segments")
    for i in range(2):
        with results.output_series(r, tmp_path / f"segment{i}.xdmf") as writer:
            writer.write_accepted(coordinate=i, primary=u)
    assert len(r.artifacts) == 4
    assert len(r.metadata["output_series"]) == 2


def test_publish_preserves_output_failure_status(tmp_path):
    import json
    from agentfem import project

    (tmp_path / "case.py").write_text("")
    config = project.ProjectConfig(
        root=tmp_path,
        name="publish",
        entrypoint=tmp_path / "case.py",
        output_directory=tmp_path / "out",
    )
    run = project.RunContext.create(config).prepare()
    r = results.SimulationResult(name="partial", status="completed_with_output_errors")
    run.publish(r)
    assert (
        json.loads(run.execution_path.read_text())["status"]
        == "completed_with_output_errors"
    )


def test_generator_export_does_not_materialize_history(tmp_path):
    from types import SimpleNamespace

    u, _ = fixture()

    def snapshots():
        for i in range(4):
            u.x.array[:] = i
            yield SimpleNamespace(solution=u, load_factor=i)

    path = results.write_unified_xdmf_series(
        tmp_path / "generated.xdmf",
        snapshots(),
        (() for _ in range(4)),
        deformation_scale=0,
    )
    with h5py.File(path.with_suffix(".h5")) as h5:
        for i in range(4):
            np.testing.assert_allclose(h5[f"Frames/{i:04d}/Point/U"][:, :2], i)


def test_moved_output_directory_remains_readable(tmp_path):
    import shutil

    u, _ = fixture()
    r = results.SimulationResult(name="portable")
    source = tmp_path / "source"
    with results.output_series(r, source / "fields.xdmf", paraview=True) as writer:
        writer.write_accepted(coordinate=0.0, primary=u)
    destination = tmp_path / "moved"
    shutil.move(str(source), str(destination))
    assert results.inspect_output(destination / "fields.xdmf")["valid"]
    assert results.inspect_output(destination / "fields.pvd")["valid"]


def test_dependency_shape_corruption_is_reported(tmp_path):
    u, _ = fixture()
    path = tmp_path / "shape.xdmf"
    with results.output_series(results.SimulationResult(name="shape"), path) as writer:
        writer.write_accepted(coordinate=0.0, primary=u)
    tree = ET.parse(path)
    item = tree.find(".//DataItem")
    item.set("Dimensions", "999 999")
    tree.write(path)
    report = results.inspect_output(path)
    assert not report["valid"]
    assert any("Shape mismatch" in item for item in report["errors"])


def test_failed_trial_is_not_an_accepted_frame(tmp_path):
    u, _ = fixture()
    path = tmp_path / "accepted.xdmf"
    with results.output_series(
        results.SimulationResult(name="cutback"), path
    ) as writer:
        u.x.array[:] = 0
        writer.write_accepted(coordinate=0.0, primary=u)
        previous = u.x.array.copy()
        u.x.array[:] = 999  # A failed trial belongs to the solver, not output.
        u.x.array[:] = previous
        u.x.array[:] = 0.5
        writer.write_accepted(coordinate=0.5, primary=u)
    assert results.inspect_output(path)["coordinates"] == [0.0, 0.5]
    with h5py.File(path.with_suffix(".h5")) as h5:
        assert len(h5["Frames"]) == 2
        np.testing.assert_allclose(h5["Frames/0001/Point/Velocity"][:, :2], 0.5)


def test_strict_missing_output_has_machine_readable_cli_code(tmp_path):
    from agentfem import cli
    from agentfem.results.series import validate_published_output

    r = results.SimulationResult(name="missing")
    r.metadata["output_series"] = [
        dict(
            status="completed",
            strict=True,
            scientific_artifact=str(tmp_path / "missing.xdmf"),
        )
    ]
    try:
        validate_published_output(r)
    except ValueError as error:
        record = cli._rank_error(error, rank=0)
    else:
        raise AssertionError("Missing strict output was published")
    assert record["code"] == "AFM-OUTPUT-INCOMPLETE"
    assert record["stage"] == "result_output"


def test_cli_partial_output_failure_has_nonzero_exit(tmp_path):
    from agentfem import cli

    (tmp_path / "agentfem.toml").write_text(
        '[project]\nname="partial"\nentrypoint="case.py"\n'
    )
    (tmp_path / "case.py").write_text(
        'from agentfem import project, results\nr=results.SimulationResult(name="partial",status="completed_with_output_errors")\nproject.current_run().publish(r)\n'
    )
    assert cli.main(["run", "--project", str(tmp_path), "--json"]) == 1
