from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import ufl
from dolfinx import fem
import h5py
from mpi4py import MPI
import pytest

from agentfem import campaigns, datasets, mesh, results, verification
from agentfem.solvers import SolveEvent
from agentfem.results.finite_strain import HomogenizedFrame


def test_result_collects_qois_histories_artifacts_and_dataset_sample(tmp_path):
    result = results.SimulationResult(
        "cantilever",
        metadata={"solver": "reference"},
    )
    result.add_quantity("tip_displacement", -1.2e-3, unit="m")
    result.add_quantity("reaction", [10.0, -20.0], unit="N")
    result.add_history(
        "energy",
        [0.0, 0.5, 1.0],
        [0.0, 2.0, 3.0],
        unit="J",
    )
    result.add_field("displacement", artifact="cantilever.xdmf", unit="m")
    result.add_artifact("visualization", "cantilever.xdmf")

    sample = result.to_sample(
        case_id="case-1",
        inputs={"young": 200.0e9},
        outputs=("tip_displacement", "reaction"),
    )
    manifest = result.write_manifest(tmp_path / "result.json", include_histories=True)
    saved = json.loads(manifest.read_text(encoding="utf-8"))

    assert sample.outputs["tip_displacement"] == -1.2e-3
    np.testing.assert_allclose(sample.outputs["reaction"], [10.0, -20.0])
    assert sample.artifacts == {"visualization": "cantilever.xdmf"}
    assert sample.provenance["simulation_result"]["status"] == "completed"
    assert result.histories["energy"].latest == 3.0
    assert saved["schema"] == "agentfem.simulation-result"
    assert saved["history_records"][0]["sample_count"] == 3
    assert saved["field_records"][0]["live"] is False


def test_result_manifest_keeps_execution_status_separate_from_scientific_trust(
    tmp_path,
):
    result = results.SimulationResult("trustworthy")
    claim = verification.VerificationClaim.compare(
        name="reference_response",
        observable="maximum_displacement",
        actual=0.1,
        expected=0.1,
        reference="versioned analytical solution",
        relative_tolerance=1.0e-8,
    )
    result.add_verification(verification.report(claim))
    saved = json.loads(
        result.write_manifest(tmp_path / "trusted.json").read_text(encoding="utf-8")
    )

    assert result.status == "completed"
    assert result.trust_level == "verified"
    assert saved["trust_level"] == "verified"
    assert saved["verification"]["claims"][0]["status"] == "passed"


def test_written_manifest_uses_portable_paths_for_local_artifacts(tmp_path):
    result = results.SimulationResult("portable")
    artifact = tmp_path / "fields.h5"
    artifact.touch()
    result.add_artifact("fields", artifact)

    manifest = result.write_manifest(tmp_path / "result.json")
    saved = json.loads(manifest.read_text(encoding="utf-8"))

    assert saved["artifacts"] == {"fields": "fields.h5"}
    assert result.summary()["artifacts"]["fields"] == str(artifact)


def test_execution_trace_preserves_hidden_and_failed_evidence_in_manifest(tmp_path):
    simulation = results.SimulationResult("auditable")
    events = (
        SolveEvent("transient_started", "heat", total_increments=2),
        SolveEvent(
            "time_increment",
            "heat",
            increment=1,
            time=0.1,
            total_increments=2,
            display=False,
        ),
        SolveEvent(
            "step_failed",
            "heat",
            increment=2,
            residual_norm=float("inf"),
            message="synthetic failure",
        ),
    )

    results.add_execution_trace(simulation, events)
    manifest = simulation.write_manifest(
        tmp_path / "result.json",
        include_histories=True,
    )
    saved = json.loads(manifest.read_text(encoding="utf-8"))

    assert saved["metadata"]["execution"]["event_count"] == 3
    assert saved["metadata"]["execution"]["events"][1]["display"] is False
    assert saved["metadata"]["execution"]["events"][2]["residual_norm"] is None
    assert saved["history_records"][0]["sample_count"] == 1


def test_checkpoint_is_a_typed_result_asset_with_portability_boundary(tmp_path):
    state = tmp_path / "state.npz"
    state.touch()
    checkpoint = results.CheckpointRecord(
        name="accepted_05",
        path=state,
        schema="agentfem.test-checkpoint.v1",
        step_name="loading",
        coordinate_name="load_factor",
        coordinate_value=0.5,
        portable=False,
        metadata={"layout": "same mesh and dofs"},
    )
    sidecar = checkpoint.write_manifest()
    result = results.SimulationResult("restartable")
    result.add_checkpoint(checkpoint)
    manifest = result.write_manifest(tmp_path / "result.json")
    saved = json.loads(manifest.read_text(encoding="utf-8"))

    assert sidecar.is_file()
    assert saved["checkpoint_records"][0]["path"] == "state.npz"
    assert saved["checkpoint_records"][0]["portable"] is False
    assert saved["artifacts"]["checkpoint_accepted_05"] == "state.npz"


def test_result_sample_builds_a_training_dataset_without_field_serialization():
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("load", 1.0, 10.0, unit="N")
    )
    result = results.SimulationResult("beam")
    result.add_quantity("maximum_displacement", 0.25, unit="m")
    result.add_field("displacement", artifact="beam.xdmf")
    sample = result.to_sample(case_id="beam-1", inputs={"load": 5.0})
    dataset = datasets.ScientificDataset(
        parameter_space=space,
        quantities=(datasets.Quantity("maximum_displacement", unit="m"),),
        samples=(sample,),
    )

    np.testing.assert_allclose(dataset.x_matrix(), [[0.4444444444444444]])
    np.testing.assert_allclose(dataset.y_matrix(), [[0.25]])
    assert "displacement" not in sample.outputs


def test_dof_statistics_excludes_ghost_values():
    fake = SimpleNamespace(
        x=SimpleNamespace(array=np.array([-2.0, 3.0, 999.0])),
        function_space=SimpleNamespace(
            dofmap=SimpleNamespace(
                index_map=SimpleNamespace(size_local=2),
                index_map_bs=1,
            ),
            mesh=SimpleNamespace(comm=None),
        ),
    )

    stats = results.dof_statistics(fake)

    assert stats == {
        "minimum": -2.0,
        "maximum": 3.0,
        "max_abs": 3.0,
        "dof_count": 2,
    }


def test_integral_average_l2_and_dataset_ready_field_statistics():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (2.0, 1.0),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    scalar = fem.Constant(domain, 3.0)
    vector = fem.Constant(domain, (3.0, 4.0))
    dx = ufl.Measure("dx", domain=domain)

    assert results.integral(scalar, measure=dx) == 6.0
    np.testing.assert_allclose(results.average(vector, measure=dx), [3.0, 4.0])
    assert results.l2_norm(vector, measure=dx) == np.sqrt(50.0)

    V = fem.functionspace(domain, ("Lagrange", 1))
    field = fem.Function(V, name="Temperature")
    field.x.array[:] = 2.0
    result = results.SimulationResult("thermal")
    captured = result.add_dof_statistics(field, unit="K")
    assert captured["minimum"] == 2.0
    assert result.quantity("Temperature_maximum") == 2.0


def test_homogenized_history_writes_exact_npz_and_human_csv(tmp_path):
    frame = HomogenizedFrame(
        load_factor=1.0,
        deformation_gradient=np.diag([1.2, 1.0, 1.0]),
        green_lagrange_strain=np.diag([0.22, 0.0, 0.0]),
        logarithmic_strain=np.diag([np.log(1.2), 0.0, 0.0]),
        first_piola_stress=np.diag([10.0, 0.0, 0.0]),
        cauchy_stress=np.diag([12.0, 0.0, 0.0]),
        deformation_jacobian=1.2,
        strain_energy_density=1.5,
        solid_reference_fraction=0.8,
        solid_current_fraction=0.75,
        stress_consistency_error=1.0e-12,
    )

    npz = results.write_homogenized_history(tmp_path / "history.npz", [frame])
    csv = results.write_homogenized_csv(tmp_path / "history.csv", [frame])
    saved = np.load(npz)

    assert Path(npz).exists()
    assert Path(csv).exists()
    np.testing.assert_allclose(saved["first_piola_stress"][0], frame.first_piola_stress)
    assert "first_piola_stress_11" in csv.read_text(encoding="utf-8").splitlines()[0]


def test_standard_field_catalog_resolves_finite_strain_e_to_le():
    variables = results.resolve_field_variables(
        ("U", "S", "E", "EVOL", "E"),
        finite_strain=True,
    )

    assert [item.key for item in variables] == ["U", "S", "LE", "EVOL"]
    assert results.preselected_fields(
        physics="solid_mechanics",
        finite_strain=True,
    ) == ("U", "S", "LE")


def test_field_output_is_declarative_and_validated():
    request = results.field_output(
        "U",
        "S",
        "E",
        every=2,
        configuration="both",
    )

    assert request.summary()["finite_strain_aliases"] == {"E": "LE"}
    assert request.every == 2
    assert request.backend == "xdmf"


def test_field_output_intervals_are_exact_marks_not_solver_increments():
    request = results.field_output("U", "S", intervals=4)

    assert request.every is None
    assert request.required_factors() == (0.25, 0.5, 0.75, 1.0)
    assert request.summary()["intervals"] == 4


def test_output_plan_combines_field_history_diagnostics_and_presentation(tmp_path):
    field = results.field_output("U", "S", intervals=4)
    plan = results.output_plan(
        tmp_path,
        field=field,
        requests=(
            results.solver_history(),
            results.finite_strain_checks(quadrature_degree=5),
        ),
        presentation=results.presentation(
            animation=None,
            comparison=False,
        ),
        basename="job",
    )

    assert plan.every is None
    assert plan.required_factors() == (0.25, 0.5, 0.75, 1.0)
    summary = plan.summary()
    assert summary["kind"] == "output_plan"
    assert [request["kind"] for request in summary["requests"]] == [
        "solver_history",
        "finite_strain_diagnostics",
    ]


def test_solver_history_request_records_accepted_increment_evidence(tmp_path):
    increments = (
        SimpleNamespace(
            load_factor=0.25,
            start_load_factor=0.0,
            residual_norm=1.0e-9,
            iterations=4,
        ),
        SimpleNamespace(
            load_factor=1.0,
            start_load_factor=0.25,
            residual_norm=2.0e-10,
            iterations=3,
        ),
    )
    result = results.SimulationResult("nonlinear")
    context = SimpleNamespace(
        step=SimpleNamespace(
            last_solve_info=SimpleNamespace(increments=increments)
        ),
        result=result,
    )

    results.solver_history().apply(context)

    np.testing.assert_allclose(
        result.histories["increment_size"].values,
        [0.25, 0.75],
    )
    np.testing.assert_allclose(
        result.histories["newton_iterations"].values,
        [4.0, 3.0],
    )


def test_result_format_is_concise_and_does_not_dump_numeric_manifest():
    result = results.SimulationResult("readable")
    result.add_quantity("long_value", np.arange(100, dtype=float))
    result.add_artifact("fields", "results.xdmf")

    text = result.format()

    assert "Result: readable" in text
    assert "quantities: 1" in text
    assert "fields: results.xdmf" not in text
    assert "99.0" not in text


def _write_two_frame_unified_xdmf(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    Q = fem.functionspace(domain, ("DG", 0))
    snapshots = []
    cell_frames = []
    for index, factor in enumerate((0.0, 1.0)):
        displacement = fem.Function(V, name="U")
        displacement.interpolate(
            lambda x, selected=factor: np.vstack(
                (0.1 * selected * x[0], np.zeros_like(x[1]))
            )
        )
        stress = fem.Function(Q, name="MISES")
        stress.x.array[:] = 10.0 * factor
        snapshots.append(
            SimpleNamespace(
                solution=displacement,
                load_factor=factor,
            )
        )
        cell_frames.append((stress,))

    xdmf = results.write_unified_xdmf_series(
        tmp_path / "compact.xdmf",
        snapshots,
        cell_frames,
    )
    return xdmf


def test_unified_xdmf_keeps_deformed_time_series_and_fields_in_one_h5(tmp_path):
    xdmf = _write_two_frame_unified_xdmf(tmp_path)

    assert xdmf.exists()
    assert xdmf.with_suffix(".h5").exists()
    assert len(tuple(tmp_path.iterdir())) == 2
    with h5py.File(xdmf.with_suffix(".h5"), "r") as h5:
        assert h5.attrs["agentfem_schema"] == "agentfem.unified-xdmf"
        geometry0 = np.asarray(h5["Frames/0000/Geometry"])
        geometry1 = np.asarray(h5["Frames/0001/Geometry"])
        np.testing.assert_allclose(geometry1[:, 0], 1.1 * geometry0[:, 0])
        np.testing.assert_allclose(
            np.asarray(h5["Frames/0001/Cell/MISES"]),
            10.0,
        )
        assert set(h5["Frames/0001/Point"]) == {"U", "UMAG"}
        assert set(h5["Frames/0001/Cell"]) == {"MISES"}


def test_unified_xdmf_optional_pyvista_reader(tmp_path):
    pytest.importorskip("pyvista")
    xdmf = _write_two_frame_unified_xdmf(tmp_path)

    grids = results.read_unified_xdmf_series(xdmf)

    assert len(grids) == 2
    np.testing.assert_allclose(grids[1].points[:, 0], 1.1 * grids[0].points[:, 0])
    np.testing.assert_allclose(grids[1].cell_data["MISES"], 10.0)


def test_result_bulk_quantity_and_history_records_share_explicit_axes():
    result = results.SimulationResult("bulk")
    result.add_quantities({"energy": 2.0, "reaction": [1.0, 2.0]})
    result.add_histories(
        [0.0, 1.0],
        {
            "displacement": [0.0, 0.2],
            "force": [[0.0, 0.0], [10.0, 0.0]],
        },
        abscissa_name="load_factor",
        abscissa_unit=None,
    )

    assert result.quantity("energy") == 2.0
    assert result.histories["force"].value_shape == (2,)
