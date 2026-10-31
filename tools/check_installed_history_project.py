# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the ordinary history-material project against an installed package.

Run with the candidate environment's Python. No source-path injection, project
solver, or rendering is used. Temporary numerical output is removed on success;
pass --work-directory to retain the complete diagnostic project on failure.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import vtk

import agentfem
from agentfem import results


def installed_identity():
    distribution = importlib.metadata.distribution("agentfem")
    expected = Path(distribution.locate_file("agentfem/__init__.py")).resolve()
    actual = Path(agentfem.__file__).resolve()
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    if (actual != expected or not expected.is_file()
            or actual.is_relative_to(Path(__file__).resolve().parents[1])
            or origin.get("dir_info", {}).get("editable", False)):
        raise RuntimeError(f"Expected an installed candidate at {expected}, imported {actual}.")
    if distribution.version != agentfem.__version__:
        raise RuntimeError("Installed metadata and imported package versions differ.")
    return {"version": agentfem.__version__, "module": str(actual)}


def fields(path):
    values = {}
    for frame in ET.parse(path).findall(".//Grid[@GridType='Uniform']"):
        time = float(frame.find("Time").get("Value"))
        for attribute in frame.findall("Attribute"):
            filename, dataset = attribute.find("DataItem").text.strip().split(":", 1)
            key = time, attribute.get("Name"), attribute.get("Center")
            if key in values:
                raise AssertionError(f"Duplicate field {key}")
            with h5py.File(path.parent / filename) as data:
                values[key] = data[dataset][:]
            if not np.isfinite(values[key]).all():
                raise AssertionError(f"Nonfinite field {key}")
    if not values:
        raise AssertionError("No field frames were published.")
    return values


def verify(root):
    identity = installed_identity()
    source = Path(__file__).resolve().parents[1] / "examples/history_material"
    project = root / "project"
    project.mkdir()
    for name in ("case.py", "material.py", "agentfem.toml", "README.md"):
        shutil.copy2(source / name, project / name)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, "-m", "agentfem._cli_entry"]

    def cli(*arguments):
        run = subprocess.run(
            [*command, *arguments], cwd=project, env=environment,
            capture_output=True, text=True, timeout=120,
        )
        if run.returncode:
            raise RuntimeError(f"CLI {arguments} failed:\n{run.stdout}\n{run.stderr}")
        return run.stdout

    checked = json.loads(cli("check", "--json"))
    if checked["status"] != "passed":
        raise AssertionError(checked)
    reports = []
    for law in ("viscoelastic", "damage"):
        paths = {}
        for kind in ("pause", "resume", "whole", "sparse"):
            name = f"{law}-{kind}"
            options = ["--material", law, "--assumption", "plane_stress"]
            if kind == "pause":
                options += ["--until", "0.5"]
            elif kind == "resume":
                options += ["--restart", str(paths["pause"] / "accepted.npz")]
            elif kind == "sparse":
                options += ["--output-every", "4"]
            cli("run", "--name", name, "--", *options)
            matches = list((project / "outputs").glob(f"*-{name}"))
            if len(matches) != 1:
                raise AssertionError(f"Ambiguous run directory for {name}: {matches}")
            paths[kind] = matches[0]
            record = json.loads((matches[0] / "result.json").read_text())
            if record["status"] != ("partial" if kind == "pause" else "completed"):
                raise AssertionError(record["status"])
            checkpoints = list((matches[0] / "checkpoints").glob("*.npz"))
            if len(checkpoints) != 2 or not all(
                p.with_suffix(".npz.checkpoint.json").is_file() for p in checkpoints
            ):
                raise AssertionError("Scheduled checkpoint retention/metadata changed.")
        expected_times = np.linspace(0.125, 2.0, 16)
        view = results.join_xdmf_segments(
            [paths["pause"] / "results.xdmf", paths["resume"] / "results.xdmf"],
            project / f"{law}-joined.xdmf", expected_times=expected_times,
        )
        dense, joined, sparse = [fields(p) for p in (
            paths["whole"] / "results.xdmf", view, paths["sparse"] / "results.xdmf"
        )]
        if dense.keys() != joined.keys() or not sparse.keys() <= dense.keys():
            raise AssertionError("Continuation or sparse field layout changed.")
        for key, value in joined.items():
            np.testing.assert_array_equal(value, dense[key])
        for key, value in sparse.items():
            np.testing.assert_array_equal(value, dense[key])
        if len({key[0] for key in sparse}) != 4:
            raise AssertionError("Sparse output did not contain four frames.")
        with np.load(paths["whole"] / "final.npz", allow_pickle=False) as whole:
            for kind in ("resume", "sparse"):
                with np.load(paths[kind] / "final.npz", allow_pickle=False) as other:
                    if set(other.files) != set(whole.files):
                        raise AssertionError("Restart state components changed.")
                    for name in whole.files:
                        np.testing.assert_array_equal(whole[name], other[name])
        reader = vtk.vtkXdmfReader()
        reader.SetFileName(str(view))
        reader.UpdateInformation()
        for time in expected_times:
            reader.UpdateTimeStep(float(time))
            grid = reader.GetOutputDataObject(0)
            if not grid.IsA("vtkUnstructuredGrid") or grid.GetNumberOfCells() != 64:
                raise AssertionError("Expected one 64-cell grid per accepted time.")
            warp = vtk.vtkWarpVector()
            warp.SetInputData(grid)
            warp.SetInputArrayToProcess(
                0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Displacement"
            )
            warp.Update()
            deformed = warp.GetOutput()
            if deformed.GetNumberOfCells() != 64 or any(
                deformed.GetCellData().GetArray(name) is None
                for name in ("S_CELL", "F_CELL", "AGE_CELL")
            ):
                raise AssertionError("Warp lost cells, stress or history fields.")
        reports.append({"law": law, "frames": 16, "sparse_frames": 4,
                        "restart_and_cadence_state_identical": True,
                        "all_common_fields_identical": True, "vtk_warp_passed": True})
    return {"status": "passed", "installed": identity, "workflows": reports,
            "scope": "same-host serial educational plane stress; not MPI or calibrated materials"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-directory", type=Path)
    args = parser.parse_args()
    if args.work_directory is None:
        with tempfile.TemporaryDirectory(prefix="agentfem-installed-history-") as directory:
            record = verify(Path(directory))
    else:
        args.work_directory.mkdir(parents=True, exist_ok=False)
        record = verify(args.work_directory.resolve())
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
