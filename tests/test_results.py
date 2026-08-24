from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dolfinx_mesh
import h5py
from mpi4py import MPI
import pytest

from agentfem import campaigns, datasets, fields, mesh, models, problems, results, studies, verification
from agentfem.constitutive import elasticity
from agentfem.kernel import dofs as dof_api
from agentfem.solvers import SolveEvent
from agentfem.results.finite_strain import HomogenizedFrame
from agentfem.results.output import attach_result_field_output


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
    assert sample.provenance["software_origin"]["project"] == "AgentFEM"
    assert sample.provenance["software_origin"]["initiated_by"] == "Haoming Luo"
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


def test_linear_step_result_carries_ksp_evidence_consumed_by_engineering_quality():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.linear_static(
            physics="solid_mechanics",
            dimension=2,
            assumption="plane_strain",
        ),
        mesh=domain,
        name="quality_linear_patch",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        elasticity.isotropic_elastic(
            young=1000.0,
            poisson=0.3,
            density=1.0,
        )
    )
    left = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 0.0),
        name="left",
        tag=1,
    )
    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 1.0),
        name="right",
        tag=2,
    )
    model.fix(displacement, on=left, value=0.0)
    model.traction((0.0, -1.0), on=right)

    simulation = model.step(target=displacement).solve_result()
    quality = simulation.verify("engineering")

    solve = simulation.metadata["step"]["problem"]["last_solve"]
    assert solve["kind"] == "linear_solve_info"
    assert solve["converged"] is True
    np.testing.assert_allclose(
        simulation.quantities["external_force_resultant"].value,
        [0.0, -0.2],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        simulation.quantities["reaction_force_resultant"].value,
        [0.0, 0.2],
        atol=1.0e-10,
    )
    assert simulation.quantities["relative_force_balance_error"].value < 1.0e-10
    assert simulation.metadata["static_equilibrium"]["definition"] == (
        "reaction + assembled external force"
    )
    assert quality.trust_level == "converged"
    assert quality.acceptable


def test_displacement_controlled_3d_elastic_patch_writes_standard_fields(tmp_path):
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2)
    material = elasticity.isotropic_elastic(
        young=210.0e3,
        poisson=0.3,
        density=1.0,
    )
    model = models.create(
        study=studies.linear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="three_dimensional_patch",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(material)
    strains = (1.0e-3, -2.0e-4, 3.0e-4)
    boundaries = (
        ("x", 0, strains[0]),
        ("y", 1, strains[1]),
        ("z", 2, strains[2]),
    )
    positive_faces = {}
    for axis, component, strain in boundaries:
        negative = mesh.face(
            domain,
            axis=axis,
            value=0.0,
            name=f"{axis}_negative",
            tag=2 * component + 1,
        )
        positive = mesh.face(
            domain,
            axis=axis,
            value=1.0,
            name=f"{axis}_positive",
            tag=2 * component + 2,
        )
        positive_faces[axis] = positive
        model.fix(displacement, on=negative, component=component, value=0.0)
        model.fix(displacement, on=positive, component=component, value=strain)

    step = model.step(target=displacement)
    output = tmp_path / "elastic_patch.xdmf"
    simulation = step.solve_result(output=output)

    strain_field = simulation.fields["E"].field
    stress_field = simulation.fields["S"].field
    expected_strain = np.diag(strains)
    np.testing.assert_allclose(
        strain_field.x.array.reshape((-1, 3, 3)),
        np.broadcast_to(
            expected_strain,
            strain_field.x.array.reshape((-1, 3, 3)).shape,
        ),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    expected_stress = (
        material.lambda_ * np.trace(expected_strain) * np.eye(3)
        + 2.0 * material.mu * expected_strain
    )
    np.testing.assert_allclose(
        stress_field.x.array.reshape((-1, 3, 3)),
        np.broadcast_to(
            expected_stress,
            stress_field.x.array.reshape((-1, 3, 3)).shape,
        ),
        rtol=2.0e-11,
        atol=2.0e-9,
    )
    assert results.reaction_resultant(
        step.problem,
        on=positive_faces["x"],
        component=0,
    ) == pytest.approx(expected_stress[0, 0], rel=2.0e-11, abs=2.0e-9)
    assert {"Displacement", "S", "E", "MISES"} <= set(simulation.fields)
    assert "SENER" not in simulation.fields
    assert simulation.fields["S"].processing == {
        "source_position": "constitutive_expression",
        "method": "global_l2_projection",
        "representation": "cell_average",
        "space_family": "P",
        "space_degree": 0,
        "nodal_extrapolation": False,
        "interelement_smoothing": False,
        "material_boundary_averaging": False,
    }
    assert simulation.fields["Displacement"].processing == {
        "method": "primary_finite_element_solution",
        "representation": "finite_element_dofs",
        "postprocessed": False,
    }
    assert simulation.artifacts["fields_xdmf"] == output
    assert simulation.artifacts["fields_hdf5"].is_file()
    assert simulation.metadata["field_output"] == {
        "status": "completed",
        "backend": "agentfem_unified_xdmf",
        "layout": "single_uniform_grid",
        "geometry": "reference",
        "warp_field": "U",
        "warp_field_semantic": "Displacement",
        "physical_components": 3,
        "stored_components": 3,
        "geometry_dimension": 3,
        "physical_model_dimension": 3,
        "warp_compatible": True,
        "field_aliases": {"Displacement": "U"},
    }
    grids = ET.parse(output).findall(".//Grid[@GridType='Uniform']")
    assert len(grids) == 1
    attributes = {
        item.attrib["Name"]: item.attrib["Center"]
        for item in grids[0].findall("Attribute")
    }
    assert attributes == {
        "U": "Node",
        "UMAG": "Node",
        "S": "Cell",
        "E": "Cell",
        "MISES": "Cell",
    }
    manifest = simulation.write_manifest(tmp_path / "elastic_patch.result.json")
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    field_records = {item["name"]: item for item in saved["field_records"]}
    assert field_records["S"]["processing"]["representation"] == "cell_average"
    assert field_records["S"]["processing"]["interelement_smoothing"] is False


def test_two_material_elastic_bar_has_piecewise_fields_and_boundary_reaction():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (2.0, 0.2),
        (8, 2),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.linear_static(
            physics="solid_mechanics",
            dimension=2,
            assumption="plane_stress",
        ),
        mesh=domain,
        name="two_material_bar",
    )
    displacement = model.field(fields.displacement(domain))
    regions = mesh.partition_cells(
        domain,
        soft=mesh.layer("x", upper=1.0),
        stiff=mesh.layer("x", lower=1.0),
    )
    # nu=0 isolates the exact one-dimensional series-bar solution while still
    # exercising the full two-dimensional regional assembly and projection.
    soft = elasticity.isotropic_elastic(young=1.0e3, poisson=0.0, density=1.0)
    stiff = elasticity.isotropic_elastic(young=2.0e3, poisson=0.0, density=1.0)
    model.material(soft, region=regions.soft)
    model.material(stiff, region=regions.stiff)
    left = mesh.face(domain, axis="x", value=0.0, name="left", tag=1)
    right = mesh.face(domain, axis="x", value=2.0, name="right", tag=2)
    bottom = mesh.face(domain, axis="y", value=0.0, name="bottom", tag=3)
    model.fix(displacement, on=left, component=0, value=0.0)
    model.fix(displacement, on=bottom, component=1, value=0.0)
    model.traction((10.0, 0.0), on=right)

    step = model.step(target=displacement)
    simulation = step.solve_result()
    stress = simulation.fields["S"].field
    strain = simulation.fields["E"].field

    assert results.region_average(stress[0, 0], on=regions.soft) == pytest.approx(10.0)
    assert results.region_average(stress[0, 0], on=regions.stiff) == pytest.approx(10.0)
    assert results.region_average(strain[0, 0], on=regions.soft) == pytest.approx(0.01)
    assert results.region_average(strain[0, 0], on=regions.stiff) == pytest.approx(0.005)
    assert results.reaction_resultant(
        step.problem,
        on=left,
        component=0,
    ) == pytest.approx(-2.0)

    energy_only = step.solve_result(field_variables=("SENER",))
    assert {"Displacement", "SENER"} == set(energy_only.fields)
    assert np.all(energy_only.fields["SENER"].field.x.array > 0.0)


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

    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 2.0),
        name="right",
        tag=1,
    )
    np.testing.assert_allclose(
        results.boundary_resultant(vector, on=right),
        [3.0, 4.0],
    )
    np.testing.assert_allclose(
        results.region_average(vector, on=right),
        [3.0, 4.0],
    )

    vector_space = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    vector_field = fem.Function(vector_space, name="U")
    vector_field.x.array[:] = np.tile(
        np.asarray([3.0, 4.0]),
        vector_field.x.array.size // 2,
    )
    extrema = results.field_extrema(vector_field, magnitude=True)
    assert extrema == {"minimum": 5.0, "maximum": 5.0, "magnitude": True}


def test_point_probe_and_path_sampling_feed_standard_results():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (2.0, 1.0),
        (4, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    temperature = fem.Function(V, name="T")
    temperature.interpolate(lambda x: x[0] + 2.0 * x[1])
    temperature.x.scatter_forward()

    assert results.probe(temperature, at=(0.25, 0.5)) == pytest.approx(1.25)
    np.testing.assert_allclose(
        results.sample_points(
            temperature,
            ((0.0, 0.0), (1.0, 0.5), (2.0, 1.0)),
        ),
        [0.0, 2.0, 4.0],
        atol=1.0e-14,
    )

    path = results.sample_path(
        temperature,
        start=(0.0, 0.5),
        end=(2.0, 0.5),
        count=5,
    )
    np.testing.assert_allclose(path.distance, np.linspace(0.0, 2.0, 5))
    np.testing.assert_allclose(
        path.values,
        np.linspace(1.0, 3.0, 5),
        atol=1.0e-14,
    )
    simulation = results.SimulationResult("sampled_temperature")
    history = path.add_to(
        simulation,
        name="temperature_along_centerline",
        unit="K",
        distance_unit="m",
    )
    assert history.abscissa_name == "distance"
    assert history.value_shape == ()


def test_vector_probe_and_missing_point_policy():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 2),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    displacement = fem.Function(V, name="U")
    displacement.interpolate(lambda x: np.vstack((x[0], -2.0 * x[1])))
    displacement.x.scatter_forward()

    np.testing.assert_allclose(
        results.probe(displacement, at=(0.25, 0.75)),
        [0.25, -1.5],
    )
    with pytest.raises(ValueError, match="could not locate point indices"):
        results.sample_points(displacement, ((2.0, 2.0),))
    values = results.sample_points(
        displacement,
        ((0.5, 0.5), (2.0, 2.0)),
        missing="nan",
    )
    np.testing.assert_allclose(values[0], [0.5, -1.0])
    assert np.all(np.isnan(values[1]))
    with pytest.raises(ValueError, match="geometric dimension"):
        results.sample_points(displacement, ((0.25, 0.5, 0.75, 1.0),))
    with pytest.raises(ValueError, match="non-negative"):
        results.sample_points(displacement, ((0.25, 0.5),), padding=-1.0)
    with pytest.raises(ValueError, match="at least two"):
        results.sample_path(
            displacement,
            start=(0.0, 0.0),
            end=(1.0, 1.0),
            count=1,
        )


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
    ) == ("U", "S", "LE", "MISES")
    assert results.field_variable("MISES").derived_from == ("S",)
    assert results.field_variable("SENER").derived_from == ("S", "E")
    assert results.field_output().variables == ("U", "S", "E", "MISES")


def test_small_strain_standard_fields_are_cell_average_projections():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.5),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    displacement = fields.displacement(domain).value
    displacement.interpolate(
        lambda x: np.vstack((0.01 * x[0], -0.002 * x[1]))
    )
    material = elasticity.isotropic_elastic(
        young=200.0e9,
        poisson=0.3,
        density=7800.0,
    )
    study = studies.linear_static(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_stress",
    )

    stress, strain, mises, energy = results.small_strain_cell_fields(
        displacement,
        material,
        study=study,
    )

    assert (stress.name, strain.name, mises.name, energy.name) == (
        "S",
        "E",
        "MISES",
        "SENER",
    )
    strain_values = strain.x.array.reshape((-1, 2, 2))
    np.testing.assert_allclose(
        strain_values,
        np.broadcast_to(np.diag([0.01, -0.002]), strain_values.shape),
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    stress_values = stress.x.array.reshape((-1, 2, 2))
    expected_mises = np.sqrt(
        stress_values[:, 0, 0] ** 2
        - stress_values[:, 0, 0] * stress_values[:, 1, 1]
        + stress_values[:, 1, 1] ** 2
        + 3.0 * stress_values[:, 0, 1] ** 2
    )
    np.testing.assert_allclose(mises.x.array, expected_mises, rtol=1.0e-12)
    assert np.all(energy.x.array > 0.0)


def test_projection_requires_a_mesh_for_domain_free_expression():
    with pytest.raises(ValueError, match="infer a mesh"):
        results.project(ufl.as_ufl(2.0))


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


def test_generic_history_request_uses_accepted_frame_coordinate_and_tensor_values():
    snapshots = (
        SimpleNamespace(load_factor=0.25, response=np.eye(2)),
        SimpleNamespace(load_factor=1.0, response=2.0 * np.eye(2)),
    )
    simulation = results.SimulationResult("history")
    request = results.history(
        "section_tensor",
        lambda snapshot, context: snapshot.response,
        unit="Pa",
        description="Section response on accepted frames.",
    )

    request.apply(
        SimpleNamespace(
            step=SimpleNamespace(snapshots=snapshots),
            result=simulation,
        )
    )

    history = simulation.histories["section_tensor"]
    assert history.abscissa_name == "load_factor"
    assert history.abscissa_unit is None
    np.testing.assert_allclose(history.abscissa, [0.25, 1.0])
    np.testing.assert_allclose(history.values[1], 2.0 * np.eye(2))


def test_generic_history_requires_explicit_coordinate_for_unlabelled_frames():
    request = results.history("custom", lambda snapshot, context: snapshot.value)
    context = SimpleNamespace(
        step=SimpleNamespace(snapshots=(SimpleNamespace(value=1.0),)),
        result=results.SimulationResult("history"),
    )

    with pytest.raises(ValueError, match="pass coordinate"):
        request.apply(context)


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


def test_incremental_unified_writer_keeps_all_fields_on_each_time_grid(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    Q = fem.functionspace(domain, ("DG", 0))
    displacement = fem.Function(V, name="Displacement")
    velocity = fem.Function(V, name="Velocity")
    stress = fem.Function(Q, name="MISES")
    output = tmp_path / "incremental.xdmf"

    with results.UnifiedXDMFTimeSeries(output) as writer:
        writer.write_fields(0.0, displacement, velocity, stress)
        displacement.x.array[:] = 0.1
        velocity.x.array[:] = 0.2
        stress.x.array[:] = 10.0
        writer.write_fields(1.0, displacement, velocity, stress)

    frames = ET.parse(output).findall(".//Grid[@GridType='Uniform']")
    assert len(frames) == 2
    for frame in frames:
        assert {
            item.attrib["Name"]: item.attrib["Center"]
            for item in frame.findall("Attribute")
        } == {
            "U": "Node",
            "UMAG": "Node",
            "Velocity": "Node",
            "MISES": "Cell",
        }


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
        displacement = np.asarray(h5["Frames/0001/Point/U"])
        assert displacement.shape == (geometry1.shape[0], 3)
        np.testing.assert_allclose(displacement[:, 2], 0.0)
        assert h5.attrs["primary_semantic_name"] == "Displacement"
        assert h5.attrs["primary_physical_components"] == 2
        assert h5.attrs["primary_storage_components"] == 3
        assert bool(h5.attrs["warp_compatible"]) is True

    uniform_grids = ET.parse(xdmf).findall(".//Grid[@GridType='Uniform']")
    assert len(uniform_grids) == 2
    for grid in uniform_grids:
        attributes = {
            item.attrib["Name"]: item for item in grid.findall("Attribute")
        }
        names = set(attributes)
        assert names == {"U", "UMAG", "MISES"}
        assert attributes["U"].attrib["AttributeType"] == "Vector"
        assert attributes["U"].find("DataItem").attrib["Dimensions"].endswith(" 3")


def test_two_dimensional_unified_displacement_is_directly_warpable(tmp_path):
    pytest.importorskip("pyvista")
    xdmf = _write_two_frame_unified_xdmf(tmp_path)

    grid = results.read_unified_xdmf_series(xdmf)[-1]
    reference_points = np.asarray(grid.points).copy()
    displacement = np.asarray(grid.point_data["U"])
    warped = grid.warp_by_vector("U", factor=1.0)

    assert displacement.shape[1] == 3
    np.testing.assert_allclose(displacement[:, 2], 0.0)
    np.testing.assert_allclose(warped.points, reference_points + displacement)


def test_unified_xdmf_keeps_three_dimensional_displacement_unchanged(tmp_path):
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1, 1, 1),
        comm=MPI.COMM_SELF,
        cell_type="tetrahedron",
    )
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    displacement = fem.Function(V, name="U")
    displacement.interpolate(
        lambda x: np.vstack((0.1 * x[0], 0.2 * x[1], -0.3 * x[2]))
    )

    xdmf = results.write_unified_xdmf_series(
        tmp_path / "three_dimensional.xdmf",
        (SimpleNamespace(solution=displacement, load_factor=1.0),),
        ((),),
        deformation_scale=0.0,
    )

    with h5py.File(xdmf.with_suffix(".h5"), "r") as h5:
        stored = np.asarray(h5["Frames/0000/Point/U"])
        assert stored.shape[1] == 3
        np.testing.assert_allclose(stored, displacement.x.array.reshape(-1, 3))
        assert h5.attrs["primary_physical_components"] == 3
        assert h5.attrs["primary_storage_components"] == 3


def test_two_dimensional_result_declares_warp_storage_contract(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    displacement = fem.Function(V, name="Displacement")
    simulation = results.SimulationResult("two_dimensional_warp")
    simulation.add_field(
        "Displacement",
        displacement,
        processing={"method": "primary_finite_element_solution"},
    )

    attach_result_field_output(
        simulation,
        tmp_path / "two_dimensional_warp.xdmf",
        strict=True,
    )

    contract = simulation.metadata["field_output"]
    assert contract["warp_field"] == "U"
    assert contract["warp_field_semantic"] == "Displacement"
    assert contract["field_aliases"] == {"Displacement": "U"}
    assert contract["physical_components"] == 2
    assert contract["stored_components"] == 3
    assert contract["geometry_dimension"] == 3
    assert contract["physical_model_dimension"] == 2
    assert contract["warp_compatible"] is True


def test_unified_xdmf_accepts_scalar_primary_field_on_reference_geometry(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.5),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    V = fem.functionspace(domain, ("Lagrange", 1))
    temperature = fem.Function(V, name="Temperature")
    temperature.interpolate(lambda x: 300.0 + 10.0 * x[0])

    xdmf = results.write_unified_xdmf_series(
        tmp_path / "temperature.xdmf",
        (SimpleNamespace(solution=temperature, load_factor=0.0),),
        ((),),
        deformation_scale=1.0,
    )

    with h5py.File(xdmf.with_suffix(".h5"), "r") as h5:
        assert h5.attrs["primary_field"] == "Temperature"
        assert h5.attrs["geometry_mode"] == "reference"
        assert set(h5["Frames/0000/Point"]) == {"Temperature"}
        np.testing.assert_allclose(
            h5["Frames/0000/Geometry"],
            h5["Mesh/ReferenceGeometry"],
        )
    grid = ET.parse(xdmf).find(".//Grid[@GridType='Uniform']")
    assert grid is not None
    assert {
        item.attrib["Name"]: item.attrib["Center"]
        for item in grid.findall("Attribute")
    } == {"Temperature": "Node"}


def test_unified_xdmf_places_continuous_auxiliary_field_on_same_p2_grid(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
    Q = fem.functionspace(domain, ("Lagrange", 1))
    displacement = fem.Function(V, name="U")
    temperature = fem.Function(Q, name="TEMP")
    temperature.interpolate(lambda x: x[0] + 2.0 * x[1])

    xdmf = results.write_unified_xdmf_series(
        tmp_path / "mixed_point_cell.xdmf",
        (SimpleNamespace(solution=displacement, load_factor=0.0),),
        ((temperature,),),
        deformation_scale=0.0,
    )

    with h5py.File(xdmf.with_suffix(".h5"), "r") as h5:
        assert set(h5["Frames/0000/Point"]) == {"U", "UMAG", "TEMP"}
        assert h5["Frames/0000/Point/TEMP"].shape[0] == h5.attrs["point_count"]
        geometry = np.asarray(h5["Frames/0000/Geometry"])
        np.testing.assert_allclose(
            np.asarray(h5["Frames/0000/Point/TEMP"]),
            geometry[:, 0] + 2.0 * geometry[:, 1],
            atol=1.0e-14,
        )
    grids = ET.parse(xdmf).findall(".//Grid[@GridType='Uniform']")
    assert len(grids) == 1
    attributes = {
        item.attrib["Name"]: item.attrib["Center"]
        for item in grids[0].findall("Attribute")
    }
    assert attributes["TEMP"] == "Node"


def test_analysis_output_failure_does_not_discard_converged_result(
    tmp_path,
    monkeypatch,
):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    displacement = fem.Function(V, name="U")

    class DummyProblem:
        system = None
        bcs = ()

        def solve(self):
            return displacement

        def summary(self):
            return {"kind": "dummy_problem"}

    step = problems.AnalysisStep("output_isolation", DummyProblem())

    def fail_output(*args, **kwargs):
        raise OSError("synthetic filesystem failure")

    monkeypatch.setattr(
        "agentfem.results.output.write_unified_xdmf_series",
        fail_output,
    )
    with pytest.warns(RuntimeWarning, match="field output failed"):
        simulation = step.solve_result(output=tmp_path / "result.xdmf")
    assert simulation.status == "completed_with_output_errors"
    assert simulation.metadata["field_output"]["error_type"] == "OSError"
    assert "fields_xdmf" not in simulation.artifacts
    with pytest.raises(OSError, match="synthetic filesystem failure"):
        step.solve_result(
            output=tmp_path / "strict.xdmf",
            strict_output=True,
        )


def test_tagged_boundary_is_canonical_and_boundary_evidence_is_auditable():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 2),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    imported_right = mesh.face(domain, axis="x", value=1.0, tag=102)
    tagged_right = mesh.tagged_boundary_region(
        domain,
        imported_right.facet_tags,
        tag=102,
        name="pressure_surface",
    )
    V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
    selected = dof_api.locate_component_dofs(V, 0, tagged_right)
    expected = fem.locate_dofs_topological(
        V.sub(0),
        domain.topology.dim - 1,
        imported_right.facets,
    )
    np.testing.assert_array_equal(selected, expected)

    evidence = tagged_right.audit(strict=True)
    assert evidence["global_tagged_facets"] == 2
    assert evidence["measure"] == pytest.approx(1.0)
    assert evidence["integrated_normal"] == pytest.approx((1.0, 0.0))
    assert results.region_measure(on=tagged_right) == pytest.approx(1.0)

    contradictory = mesh.tagged_boundary_region(
        domain,
        imported_right.facet_tags,
        tag=102,
        name="contradictory",
        marker=mesh.plane("x", 0.0),
    )
    mismatch = contradictory.audit()
    assert mismatch["consistent"] is False
    assert mismatch["marker_only_facets"] == 2
    assert mismatch["tag_only_facets"] == 2
    with pytest.raises(ValueError, match="tag and marker disagree"):
        contradictory.audit(strict=True)


def test_field_extrema_can_report_dg0_cell_location():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    Q = fem.functionspace(domain, ("DG", 0))
    field = fem.Function(Q, name="MISES")
    field.x.array[:] = (3.0, 8.0)

    record = results.SimulationResult("located_extrema")
    field_record = record.add_field("MISES", field, location="cells")
    extrema = results.field_extrema(field_record, location=True)

    assert extrema["minimum"] == pytest.approx(3.0)
    assert extrema["maximum"] == pytest.approx(8.0)
    assert extrema["entity_kind"] == "cell"
    assert extrema["sampling"] == "cell_values"
    assert extrema["field_representation"] == "finite_element_dofs"
    assert extrema["maximum_global_cell"] == 1
    assert len(extrema["maximum_location"]) == 2


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
