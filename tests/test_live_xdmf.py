# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Published frames remain readable while the next frame is being written."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import agentfem
from agentfem.io import ResultWriter
from test_xdmf_layout import fields, check_warp


def read(path):
    meshio = pytest.importorskip("meshio")
    with meshio.xdmf.TimeSeriesReader(path) as reader:
        points, blocks = reader.read_points_cells()
        assert len(blocks) == 1 and blocks[0].type == "hexahedron"
        assert len(blocks[0].data) == 8
        records = [reader.read_data(i) for i in range(reader.num_steps)]
    return points, records


def test_live_frames_are_readable_before_close_and_share_mesh(tmp_path):
    domain, u, damage = fields()
    path = tmp_path / "results.xdmf"
    with ResultWriter(path, domain, (u, damage), storage="frame_files") as writer:
        writer.write(0.)
        assert ET.parse(path).find("Domain/Information").get("Value") == "incomplete"
        points, records = read(path)
        np.testing.assert_allclose(records[0][1]["displacement"], .1 * points)
        previous = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (tmp_path / "results-data").glob("*.h5")}
        u.x.array[:] *= 2
        writer.write(1.)
        assert len(read(path)[1]) == 2
        for p, digest in previous.items():
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
    assert ET.parse(path).find("Domain/Information") is None
    assert not (tmp_path / "results.partial.xdmf").exists()
    assert not (tmp_path / "results.h5").exists()
    import h5py
    data = tmp_path / "results-data"
    assert len(list(data.glob("*.h5"))) == 3
    with h5py.File(data / "frame-0001.h5", "r") as frame:
        assert "Mesh" not in frame
        assert "Geometry" not in frame["Frames/0001"]
    refs = ET.parse(path).findall(".//Geometry/DataItem")
    assert all(item.text.strip() == "results-data/mesh.h5:/Mesh/ReferenceGeometry" for item in refs)


def test_live_vtk_single_volume_and_warp(tmp_path):
    domain, u, damage = fields()
    path = tmp_path / "results.xdmf"
    with ResultWriter(path, domain, (u, damage), storage="frame_files") as writer:
        writer.write(0.)
        check_warp(path)


@pytest.mark.parametrize("storage", ["single_file", "frame_files"])
def test_writer_artifacts_follow_only_current_published_view(tmp_path, storage):
    domain, u, damage = fields()
    path = tmp_path / "results.xdmf"
    with ResultWriter(path, domain, (u, damage), storage=storage) as writer:
        assert writer.artifacts() == {}
        writer.write(0.)
        first = writer.artifacts()
        assert first["field_output"].is_file()
        assert len(first) == (2 if storage == "single_file" else 3)
        # Unreferenced junk is not included in the delivery inventory.
        (tmp_path / "unrelated.h5").touch()
        assert writer.artifacts() == first
    final = writer.artifacts()
    assert final["field_output"] == path
    assert all(file.is_file() for file in final.values())
    idle = ResultWriter(path, domain, (u, damage), storage=storage)
    assert idle.artifacts() == {}  # Not this writer's prior run.


def test_kill_during_next_frame_preserves_published_view(tmp_path):
    script = textwrap.dedent("""
        import os, sys, time
        from pathlib import Path
        import agentfem
        assert Path(agentfem.__file__).resolve() == Path(sys.argv[2]), agentfem.__file__
        from dolfinx import fem, mesh
        from mpi4py import MPI
        from agentfem.io import ResultWriter

        folder = Path(sys.argv[1])
        domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2,
                                      cell_type=mesh.CellType.hexahedron)
        u = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (3,))), name="displacement")
        u.interpolate(lambda x: .1*x)
        with ResultWriter(folder/"results.xdmf", domain, (u,), storage="frame_files") as writer:
            writer.write(0.)
            (folder/"ready").touch()
            deadline = time.monotonic()+35
            while not (folder/"continue").exists():
                if time.monotonic() > deadline:
                    os._exit(99)
                time.sleep(.02)
            original = writer._series._write_frame_attributes
            def interrupted(*args):
                original(*args)
                os._exit(24)  # Second HDF5 still open; neither close nor index publication.
            writer._series._write_frame_attributes = interrupted
            u.x.array[:] *= 2
            writer.write(1.)
    """)
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), str(Path(agentfem.__file__).resolve())],
        # A fresh interpreter must test the parent's package, not an older
        # globally installed writer. This also holds for installed-wheel CI.
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
             "PYTHONPATH": str(Path(agentfem.__file__).resolve().parents[1])},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 35
        while not (tmp_path / "ready").exists():
            if child.poll() is not None:
                pytest.fail(child.communicate()[0])
            if time.monotonic() > deadline:
                child.kill()
                output, _ = child.communicate(timeout=10)
                pytest.fail(f"Child did not publish its first frame.\n{output}")
            time.sleep(.02)
        assert child.poll() is None
        path = tmp_path / "results.xdmf"
        points, records = read(path)  # Independent process reads while solver is alive.
        np.testing.assert_allclose(records[0][1]["displacement"], .1*points)
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (tmp_path / "results-data").glob("*.h5")}
        (tmp_path / "continue").touch()
        output, _ = child.communicate(timeout=15)
        assert child.returncode == 24, output
        assert len(read(path)[1]) == 1
        for p, digest in before.items():
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
        assert (tmp_path / "results-data/.frame-0001.h5.tmp").exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=10)


def test_live_rejects_existing_output_and_incompatible_mode(tmp_path):
    domain, u, _ = fields()
    path = tmp_path / "results.xdmf"
    path.write_text("existing")
    with pytest.raises(FileExistsError):
        with ResultWriter(path, domain, (u,), storage="frame_files"):
            pass
    assert path.read_text() == "existing"
    with pytest.raises(ValueError, match="fresh serial"):
        ResultWriter(tmp_path / "other.xdmf", domain, mode="a", storage="frame_files")
    with pytest.raises(ValueError, match="fresh serial"):
        ResultWriter(tmp_path / "other.pvd", domain, storage="frame_files")


@pytest.mark.parametrize("storage", ["single_file", "frame_files"])
@pytest.mark.parametrize("start", [1., 1e-30, 1e15])
def test_time_index_preserves_distinct_binary64_stations(tmp_path, storage, start):
    domain, u, damage = fields()
    path = tmp_path / "results.xdmf"
    times = np.array([start, np.nextafter(start, np.inf)])
    with ResultWriter(path, domain, (u, damage), storage=storage) as writer:
        for time_value in times:
            writer.write(time_value)
    parsed = [float(item.get("Value")) for item in ET.parse(path).findall(".//Time")]
    np.testing.assert_array_equal(parsed, times)
    assert parsed[1] > parsed[0]
    np.testing.assert_array_equal([record[0] for record in read(path)[1]], times)


def test_pvd_index_preserves_distinct_binary64_stations(tmp_path):
    pytest.importorskip("pyvista")
    from types import SimpleNamespace
    from agentfem.results.output import write_deformed_vtk_series
    _, u, _ = fields()
    times = [1., np.nextafter(1., np.inf)]
    snapshots = [SimpleNamespace(solution=u, load_factor=t) for t in times]
    path, _ = write_deformed_vtk_series(tmp_path / "results.pvd", snapshots, [(), ()])
    parsed = [float(item.get("timestep")) for item in ET.parse(path).findall(".//DataSet")]
    np.testing.assert_array_equal(parsed, times)


def test_accepted_project_has_one_hundred_frames_and_one_entry(tmp_path):
    from agentfem import project, results
    from test_project_execution import context
    from test_response_material_step import make_step
    import json

    run = context(tmp_path / "project")
    step = make_step(steps=100)
    path = run.artifact("results.xdmf")
    report = project.execute_step(step, context=run,
        policy=project.RunPolicy(field_intervals=99),
        frames=results.AcceptedResponseSeries(path, storage="frame_files"))
    assert report["complete"]
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert len(metadata["stations"]) == 100
    assert metadata["stations"][0]["increment"] == 0
    assert metadata["stations"][-1]["increment"] == 100
    assert metadata["storage"] == "frame_files"
    assert Path(report["artifacts"]["process_arrays"]).is_dir()
    assert len(list(run.output_directory.glob("*.xdmf"))) == 1
    assert len(list((run.output_directory / "results-data").glob("frame-*.h5"))) == 100
