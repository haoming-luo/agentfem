# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Saved-view contracts and completed-only export publication."""

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest

from agentfem._viewer_export import validate_view
from agentfem._viewer_jobs import ExportJob


def view_record():
    return dict(schema="agentfem.view.v1", source="series.pvd", frame=0,
                preset="publication", palette="cividis", choice=["cell", "damage", None],
                range=[0., 1.], reference_geometry=False, displacement=None, scale=1., edges=False,
                title="Damage evolution", unit="", time_unit="s", camera=None)


@pytest.mark.parametrize("changes", [
    {"schema": "wrong"}, {"scale": float("nan")}, {"frame": -1}, {"frame": True},
    {"range": [1., 0.]}, {"range": [0., float("inf")]}, {"preset": "unknown"},
    {"choice": ["point", "U", "norm2"]}, {"reference_geometry": True},
    {"camera": {"position": [0, 0, 0]}}, {"title": "x" * 161},
])
def test_saved_view_rejects_invalid_contract(changes):
    with pytest.raises((ValueError, TypeError)):
        validate_view(dict(view_record(), **changes))


def test_saved_view_has_no_executable_payload():
    record = view_record()
    assert validate_view(json.loads(json.dumps(record))) == record


def test_viewer_download_asset_is_packaged():
    from importlib.resources import files
    asset = files("agentfem").joinpath("_viewer_assets/viewer.js")
    text = asset.read_text()
    assert "window.AgentFEMViewer" in text and "URL.createObjectURL" in text
    assert "url.origin !== window.location.origin" in text


def test_export_failure_never_publishes_partial_output(tmp_path, monkeypatch):
    import agentfem._viewer_jobs as jobs
    class Failed:
        returncode = 2
        def __init__(self, args, **kwargs):
            stage = Path(args[-1]).parent
            (stage / "result").mkdir()
            (stage / "result/animation.mp4").write_bytes(b"partial")
        def poll(self):
            return 2
        def wait(self, **kwargs):
            return 2
    monkeypatch.setattr(jobs.subprocess, "Popen", Failed)
    job = ExportJob(tmp_path / "exports")
    with pytest.raises(RuntimeError, match="source results are unchanged"):
        asyncio.run(job.run({"view": view_record()}, lambda *_: None))
    assert list(job.destination.iterdir()) == []
    assert job.process is None


def test_export_cancel_terminates_and_does_not_publish(tmp_path, monkeypatch):
    import agentfem._viewer_jobs as jobs
    instances = []
    class Running:
        returncode = None
        def __init__(self, *args, **kwargs):
            instances.append(self)
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = -15
        def wait(self, **kwargs):
            return self.returncode
    monkeypatch.setattr(jobs.subprocess, "Popen", Running)
    job = ExportJob(tmp_path / "exports")
    async def exercise():
        task = asyncio.create_task(job.run({"view": view_record()}, lambda *_: None))
        await asyncio.sleep(0)
        job.cancel()
        assert await task is None
    asyncio.run(exercise())
    assert instances[0].returncode == -15
    assert not list(job.destination.iterdir())


def test_export_success_publishes_bundle(tmp_path, monkeypatch):
    import agentfem._viewer_jobs as jobs
    class Completed:
        returncode = 0
        def __init__(self, args, **kwargs):
            stage = Path(args[-1]).parent
            (stage / "result").mkdir()
            (stage / "result/figure.png").write_bytes(b"finished")
            (stage / "result/view.json").write_text(json.dumps(view_record()))
        def poll(self):
            return 0
        def wait(self, **kwargs):
            return 0
    monkeypatch.setattr(jobs.subprocess, "Popen", Completed)
    job = ExportJob(tmp_path / "exports")
    target = asyncio.run(job.run({"view": view_record()}, lambda *_: None))
    assert (target / "figure.png").read_bytes() == b"finished"
    assert len(list(job.destination.iterdir())) == 1


@pytest.mark.skipif(os.environ.get("AGENTFEM_VIEWER_RENDER_TESTS") != "1", reason="Requires an offscreen VTK rendering device")
def test_real_png_and_movie_exports(tmp_path):
    pv = pytest.importorskip("pyvista")
    pytest.importorskip("imageio_ffmpeg")
    from PIL import Image
    from agentfem._viewer_export import render
    grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
    for i in range(2):
        grid.cell_data["damage"] = np.linspace(0, .5 + i*.5, grid.n_cells)
        grid.save(tmp_path / f"frame{i}.vtu")
    (tmp_path / "series.pvd").write_text('<VTKFile type="Collection"><Collection>' + ''.join(
        f'<DataSet timestep="{i}" file="frame{i}.vtu"/>' for i in range(2)) + '</Collection></VTKFile>')
    request = dict(root=str(tmp_path), view=view_record(), size=[512, 512], fps=10)
    image = render(dict(request, mode="image", transparent=True), tmp_path / "image")
    with Image.open(image) as result:
        assert result.size == (512, 512) and result.mode == "RGBA"
        assert np.asarray(result)[..., 3].min() == 0
    movie = render(dict(request, mode="process"), tmp_path / "movie")
    import imageio.v2 as imageio
    with imageio.get_reader(movie) as reader:
        assert reader.count_frames() == 8
        first, last = reader.get_data(0), reader.get_data(7)
        # The color bar persists after changing the mapper's dataset, with a
        # fixed LUT. Compression permits a small pixel difference.
        bar_first, bar_last = first[470:480, 145:360], last[470:480, 145:360]
        assert bar_first.std() > 15
        assert np.abs(bar_first.astype(float)-bar_last).mean() < 3
    record = json.loads((movie.parent / "view.json").read_text())
    assert record["export"]["saved_times"] == [0., 1.]
    assert record["export"]["interpolation"] == "none"
    # Exercise cancellation of an actual subprocess, not only a fake encoder.
    job = ExportJob(tmp_path / "cancelled")
    async def cancel_real_worker():
        task = asyncio.create_task(job.run(dict(request, mode="orbit"), lambda *_: None))
        await asyncio.sleep(.05)
        assert job.process is not None
        job.cancel()
        assert await task is None
    asyncio.run(cancel_real_worker())
    assert job.process is None and list(job.destination.iterdir()) == []
