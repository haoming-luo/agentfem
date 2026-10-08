"""Geometry selection must reach every independent lifecycle solve."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import zhang_2021_macro_tangent_fd as tangent
import zhang_2021_plane_strain_restart_driver as restart


@pytest.mark.parametrize("geometry", ("figure-10a", "section-3.2.1-text"))
def test_restart_build_forwards_geometry(monkeypatch, geometry):
    def fixture(comm, *, mesh_size, geometry_source):
        assert geometry_source == geometry
        assert mesh_size == 0.3
        raise RuntimeError("fixture reached")

    monkeypatch.setattr(restart, "zhang_2021_plane_strain_composite", fixture)
    with pytest.raises(RuntimeError, match="fixture reached"):
        restart._build(None, mesh_size=0.3, increments=20, geometry_source=geometry)


@pytest.mark.parametrize("geometry", ("figure-10a", "section-3.2.1-text"))
def test_tangent_candidate_forwards_geometry(tmp_path, monkeypatch, geometry):
    def run(command, **kwargs):
        assert command[command.index("--geometry-source") + 1] == geometry
        (tmp_path / "zhang_2021_table5_plane_strain_assessment.json").write_text(
            json.dumps({"result_status": "completed", "identity_stable_during_run": True}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tangent.subprocess, "run", run)
    tangent._run_candidate(
        output=tmp_path, mesh_size=0.3, quadrature_degree=4, increments=20,
        penultimate=tangent.BASE_GRADIENT, final=tangent.BASE_GRADIENT,
        tangent=True, geometry_source=geometry,
    )


def test_restart_comparison_rejects_geometry_change():
    reference = {
        "geometry_source": "figure-10a", "accepted_increment_factors": [0.5, 1.0],
        "attempted_increment_count": 2, "global_cells": 100,
        "mesh_identity": "mesh", "constraint_identity": "constraint",
        "elastic_energy_density": 1.0,
    }
    changed = deepcopy(reference)
    changed["geometry_source"] = "section-3.2.1-text"
    assert restart._compare(reference, reference)["passed"]
    assert restart._compare(reference, changed)["failed_checks"] == ("geometry_source",)
