"""Structure-level evidence for the public 316-steel ratcheting example."""

from __future__ import annotations

from hashlib import sha256
from importlib import resources
import json
import os
from pathlib import Path

import pytest

from agentfem import benchmarks


def test_digitized_curve_keeps_figure_uncertainty_explicit():
    curve = benchmarks.simulia_316_experimental_ratcheting_curve()

    assert curve.cycle == (5, 10, 20, 40, 60, 80, 100)
    assert curve.maximum_axial_strain[-1] == pytest.approx(0.01450)
    assert curve.absolute_uncertainty > 0.0
    assert curve.as_dict()["uncertainty_basis"]


def test_digitized_curve_asset_matches_its_public_manifest():
    root = resources.files("agentfem.knowledge.external_data")
    manifest = json.loads(
        root.joinpath("simulia_316_ratcheting_figure4.json").read_text(encoding="utf-8")
    )
    payload = root.joinpath(manifest["curve_file"]).read_bytes()

    assert sha256(payload).hexdigest() == manifest["curve_file_sha256"]
    assert manifest["source"]["input_sha256"].startswith("a0d5ed2a")


def test_modified_input_cannot_claim_the_published_identity(tmp_path: Path):
    altered = tmp_path / "altered.inp"
    altered.write_text("*Heading\n** not the public deck\n", encoding="utf-8")
    assert not benchmarks.verify_simulia_ratcheting_input(altered)


def test_published_input_hash_is_verified_when_asset_is_available():
    source = Path("/tmp/ratch_axi_unsymcyclic_2.inp")
    if not source.exists():
        pytest.skip("The optional public SIMULIA input deck is not available.")
    assert benchmarks.verify_simulia_ratcheting_input(source)


def test_convergence_contract_rejects_single_level_axes():
    with pytest.raises(ValueError, match="at least two"):
        benchmarks.certify_simulia_316_shouldered_ratcheting_convergence(
            mesh_sizes=(2.5,),
        )
    with pytest.raises(ValueError, match="at least two"):
        benchmarks.certify_simulia_316_shouldered_ratcheting_convergence(
            refinements=(8,),
        )
    with pytest.raises(ValueError, match="at least two"):
        benchmarks.certify_simulia_316_shouldered_ratcheting_accuracy(
            maximum_inelastic_increments=(2.5e-4,),
        )
    with pytest.raises(ValueError, match="strictly decreasing"):
        benchmarks.certify_simulia_316_shouldered_ratcheting_accuracy(
            maximum_inelastic_increments=(2.5e-4, 5.0e-4),
        )


def test_full_reference_contract_requires_every_published_cycle():
    with pytest.raises(ValueError, match="every published cycle"):
        benchmarks.certify_simulia_316_shouldered_ratcheting_full_reference(
            cycle_count=99,
        )


@pytest.mark.skipif(
    os.environ.get("AGENTFEM_RUN_EXTERNAL_STRUCTURE") != "1",
    reason="release/nightly external structure obligation",
)
def test_shouldered_specimen_runs_as_a_real_axisymmetric_structure():
    pytest.importorskip("gmsh")
    assessment, result = benchmarks.simulia_316_shouldered_ratcheting_benchmark(
        cycle_count=5,
        refinement=4,
        mesh_size=2.5,
    )

    assert assessment.accepted
    assert assessment.final_residual_norm < 1.0e-7
    assert assessment.cell_count > 100
    assert assessment.maximum_absolute_curve_error < 1.25e-3
    assert result.metadata["external_benchmark"]["reference_curve"] == (
        "digitized_experimental_figure_with_uncertainty"
    )
    assert "maximum_center_axial_strain" in result.histories


@pytest.mark.skipif(
    os.environ.get("AGENTFEM_RUN_EXTERNAL_STRUCTURE") != "1",
    reason="release/nightly external structure obligation",
)
def test_shouldered_specimen_adaptive_path_hits_every_physical_knot():
    pytest.importorskip("gmsh")
    assessment, result = benchmarks.simulia_316_shouldered_ratcheting_benchmark(
        cycle_count=5,
        refinement=1,
        mesh_size=3.5,
        maximum_inelastic_increment=2.0e-3,
    )

    control = result.metadata["external_benchmark"]["path_control"]
    assert assessment.accepted
    assert control["kind"] == "automatic_maximum_inelastic_increment"
    assert control["all_mandatory_coordinates_reached"]
    assert control["accepted_increment_count"] >= (
        control["mandatory_coordinate_count"] - 1
    )
    assert control["maximum_accepted_plastic_increment"] <= 2.0e-3 * (1.0 + 1.0e-10)
