from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import (
    amplitudes,
    constitutive,
    constraints,
    fields,
    mesh,
    models,
    procedures,
    results,
    studies,
)
from agentfem.constitutive import elasticity


def _left(x):
    return np.isclose(x[0], 0.0)


def _right(x):
    return np.isclose(x[0], 1.0)


def test_smooth_step_has_clipped_values_and_zero_endpoint_slopes():
    history = amplitudes.smooth_step(
        2.0, 5.0, start_time=1.0, end_time=3.0,
    )
    assert history(0.0) == pytest.approx(2.0)
    assert history(4.0) == pytest.approx(5.0)
    assert history(2.0) == pytest.approx(3.5)
    epsilon = 1.0e-6
    assert (history(1.0 + epsilon) - history(1.0)) / epsilon < 1.0e-5
    assert (history(3.0) - history(3.0 - epsilon)) / epsilon < 1.0e-5
    assert history.summary()["metadata"]["endpoint_slopes"] == "zero"


def test_cuboid_is_the_three_dimensional_structured_mesh_factory():
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (2.0, 1.0, 0.5),
        (2, 1, 1),
        comm=MPI.COMM_SELF,
    )
    summary = mesh.summarize_mesh(domain)

    assert summary.geometric_dim == 3
    assert summary.global_cells == 2


def test_imported_mesh_facade_is_accepted_by_fields_and_model():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    imported = mesh.FEMMesh(domain)
    study = studies.static_solid(dimension=2, assumption="plane_strain")
    model = models.create(study=study, mesh=imported)
    displacement = model.field(fields.displacement(imported))

    assert displacement.value.function_space.mesh is domain
    assert model.domain is domain


def test_prescribed_component_accepts_engineering_axis_name():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, _left, name="left", tag=1)

    constraint = model.prescribe(
        displacement,
        0.0,
        component="x",
        on=left,
    )

    assert len(constraint.dirichlet) == 1
    assert constraint.dirichlet[0].name.endswith("component_0")


def test_static_solid_clamp_and_gravity_are_model_owned():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    study = studies.static_solid(
        dimension=2,
        assumption="plane_strain",
        name="gravity_static",
    )
    model = models.create(study=study, mesh=domain, name="gravity_model")
    displacement = model.field(fields.displacement(domain))
    model.material(
        elasticity.isotropic_elastic(
            young=1.0e6,
            poisson=0.3,
            density=1000.0,
            name="dense solid",
        )
    )
    left = mesh.boundary(domain, _left, name="left", tag=1)
    model.clamp(displacement, on=left)
    gravity = model.gravity((0.0, -9.81))
    capability = models.step_capability(model)
    model.check()
    simulation = model.step(target=displacement).solve_result()

    assert capability["supported"] is True
    assert capability["provider"]["name"] == "linear_static_operators"
    assert gravity.summary()["kind"] == "gravity_load"
    assert model.constraints[0].summary()["dirichlet"][0]["name"].startswith("clamped")
    assert simulation.status == "completed"
    assert displacement.max_abs() > 0.0


def test_provider_uses_scientific_builder_beneath_compatibility_facade():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        elasticity.isotropic_elastic(young=1.0e6, poisson=0.3, density=1.0)
    )
    model.clamp(
        displacement,
        on=mesh.boundary(domain, _left, name="left", tag=1),
    )
    model.linear_static_step = lambda **kwargs: pytest.fail(
        "provider called the compatibility facade"
    )

    simulation = model.step(target=displacement).solve_result()

    assert simulation.status == "completed"


def test_model_step_rejects_unknown_options_before_assembly():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        elasticity.isotropic_elastic(
            young=1.0e6,
            poisson=0.3,
            density=1.0,
        )
    )

    capability = models.step_capability(
        model,
        target=displacement,
        options={"solver_option": None},
    )

    assert capability["supported"] is False
    assert capability["provider"]["name"] == "linear_static_operators"
    assert capability["option_issues"][0]["code"] == "AFM-STEP-OPTION-001"
    with pytest.raises(TypeError, match="Did you mean 'solver_options'"):
        model.step(target=displacement, solver_option=None)


def test_surface_force_distributes_requested_resultant_over_reference_boundary():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (2.0, 0.5),
        (4, 2),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"),
        mesh=domain,
        name="end_resultant",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        elasticity.isotropic_elastic(young=1.0e6, poisson=0.3, density=1.0)
    )
    left = mesh.boundary(domain, _left, name="support", tag=1)
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 2.0), name="loaded_end", tag=2)
    model.clamp(displacement, on=left)
    load = model.surface_force((125.0, -50.0), on=right)

    applied = results.boundary_resultant(load.value, on=right)
    simulation = model.step(target=displacement).solve_result()
    balance = results.static_force_balance(model.steps[-1].problem)

    np.testing.assert_allclose(applied, [125.0, -50.0], rtol=1.0e-12, atol=1.0e-12)
    assert load.reference_measure == pytest.approx(0.5)
    assert load.summary()["distribution"] == "uniform"
    assert simulation.status == "completed"
    assert balance.relative_error < 1.0e-10


def test_axisymmetric_surface_force_uses_revolved_boundary_area():
    domain = mesh.rectangle(
        (1.0, 0.0),
        (2.0, 0.5),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    study = studies.static_solid(dimension=2, assumption="axisymmetric")
    model = models.create(study=study, mesh=domain, name="axisymmetric_end_force")
    top = mesh.boundary(
        domain,
        lambda x: np.isclose(x[1], 0.5),
        name="loaded_annulus",
        tag=1,
    )
    load = model.surface_force((0.0, 300.0), on=top)

    applied = results.boundary_resultant(load.value, on=top, study=study)

    np.testing.assert_allclose(applied, [0.0, 300.0], rtol=1.0e-12, atol=1.0e-12)
    assert load.reference_measure == pytest.approx(3.0 * np.pi)


def test_axisymmetric_validation_requires_nonnegative_radius_and_axis_regularity():
    negative_domain = mesh.rectangle(
        (-0.1, 0.0),
        (1.0, 1.0),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    study = studies.static_solid(dimension=2, assumption="axisymmetric")
    invalid = models.create(study=study, mesh=negative_domain)
    invalid.field(fields.displacement(negative_domain))
    invalid.material(
        elasticity.isotropic_elastic(young=1000.0, poisson=0.3, density=1.0)
    )
    report = invalid.validate()
    assert any(item.code == "AFM-AXISYM-001" for item in report.errors)

    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(study=study, mesh=domain)
    displacement = model.field(fields.displacement(domain))
    model.material(
        elasticity.isotropic_elastic(young=1000.0, poisson=0.3, density=1.0)
    )
    warning = model.validate()
    assert any(item.code == "AFM-AXISYM-002" for item in warning.warnings)

    axis = mesh.face(domain, axis="x", value=0.0, name="axis", tag=1)
    model.constraint(constraints.axisymmetric_axis(displacement, on=axis))
    regular = model.validate()
    assert not any(item.code == "AFM-AXISYM-002" for item in regular.issues)


def test_axisymmetric_lumped_mass_is_the_full_revolved_body_mass():
    domain = mesh.rectangle(
        (1.0, 0.0),
        (2.0, 0.5),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    study = studies.static_solid(dimension=2, assumption="axisymmetric")
    model = models.create(study=study, mesh=domain)
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        elasticity.isotropic_elastic(young=1000.0, poisson=0.3, density=2.0)
    )

    lumped = model.lumped_mass(displacement, material=material)

    physical_mass = 2.0 * np.pi * (2.0**2 - 1.0**2) * 0.5
    assert np.sum(lumped.mass) == pytest.approx(2.0 * physical_mass)


def test_axisymmetric_reference_coupling_rejects_undefined_ring_kinematics():
    domain = mesh.rectangle(
        (1.0, 0.0),
        (2.0, 0.5),
        (1, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="axisymmetric"),
        mesh=domain,
    )
    top = mesh.face(domain, axis="y", value=0.5, name="top", tag=1)

    with pytest.raises(NotImplementedError, match="ring/reference"):
        model.distributing_coupling((0.0, 1.0), on=top)
    with pytest.raises(NotImplementedError, match="ring/reference"):
        model.remote_force((0.0, 1.0), reference_point=(0.0, 0.5), on=top)


def test_steady_heat_transfer_consumes_flux_and_convection(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (6, 2),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    study = studies.steady_heat_transfer(dimension=2)
    model = models.create(study=study, mesh=domain, name="steady_heat")
    temperature = model.field(fields.temperature(domain, value=300.0))
    model.material(
        constitutive.thermoelastic(
            name="thermal solid",
            young=1.0e9,
            poisson=0.3,
            density=1000.0,
            thermal_expansion=1.0e-5,
            conductivity=10.0,
            specific_heat=500.0,
            reference_temperature=300.0,
        )
    )
    left = mesh.boundary(domain, _left, name="left", tag=1)
    right = mesh.boundary(domain, _right, name="right", tag=2)
    model.prescribed_temperature(temperature, 400.0, on=left)
    model.convection(
        on=right,
        coefficient=25.0,
        ambient_temperature=300.0,
    )
    model.check()
    # A user-supplied bulk operator must not bypass registered boundary physics.
    simulation = model.step(
        target=temperature,
        K=model.conduction(temperature),
        output=tmp_path / "steady_heat.xdmf",
    ).solve_result()

    assert simulation.status == "completed"
    assert simulation.artifacts["fields_xdmf"].is_file()
    assert 300.0 < temperature.max_value() <= 400.0 + 1.0e-8
    summary = model.boundary_models[0].summary()
    assert summary["kind"] == "thermal_convection_boundary"


def test_transient_heat_transfer_consumes_convection_boundary(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.transient_heat_transfer(dimension=2),
        mesh=domain,
        name="cooling",
    )
    temperature = model.field(fields.temperature(domain, value=400.0))
    model.material(
        constitutive.thermoelastic(
            name="thermal solid",
            young=1.0e9,
            poisson=0.3,
            density=1000.0,
            thermal_expansion=1.0e-5,
            conductivity=10.0,
            specific_heat=500.0,
            reference_temperature=300.0,
        )
    )
    right = mesh.boundary(domain, _right, name="right", tag=1)
    convection = model.convection(
        on=right,
        coefficient=25.0,
        ambient_temperature=amplitudes.ramp(
            300.0,
            320.0,
            start_time=0.0,
            end_time=2.0,
            name="ambient_ramp",
        ),
    )
    initial_mean = float(np.mean(temperature.value.x.array))
    mean_temperature = results.history(
        "mean_temperature",
        lambda accepted_step, time: np.mean(accepted_step.current.x.array),
        unit="K",
    )
    step = model.step(
        target=temperature,
        dt=1.0,
        steps=2,
        progress=False,
        output=tmp_path / "cooling.xdmf",
        history=mean_temperature,
    )
    simulation = step.solve_result(
        metadata={"case_role": "transient_contract_test"},
    )

    assert simulation.status == "completed"
    assert float(np.mean(temperature.value.x.array)) < initial_mean
    assert float(convection.ambient_temperature.value) == 320.0
    assert simulation.artifacts["fields_xdmf"].is_file()
    assert simulation.artifacts["fields_hdf5"].is_file()
    frames = ET.parse(simulation.artifacts["fields_xdmf"]).findall(
        ".//Grid[@GridType='Uniform']"
    )
    assert len(frames) == 2
    assert all(
        {attribute.attrib["Name"] for attribute in frame.findall("Attribute")}
        == {"Temperature"}
        for frame in frames
    )
    assert simulation.metadata["field_output"]["layout"] == "single_uniform_grid"
    assert simulation.metadata["case_role"] == "transient_contract_test"
    assert simulation.metadata["execution_context"]["model"] == "cooling"
    policies = simulation.metadata["execution_context"]["policies"]
    assert policies["output"] == str(tmp_path / "cooling.xdmf")
    assert policies["progress"] is False
    assert policies["history"][0]["type"] == "HistoryRequest"
    assert simulation.metadata["execution_context"]["declared_policies"] == policies
    assert (
        simulation.metadata["execution_context"]["resolved_step_record"]
        == "metadata.step"
    )
    assert simulation.fields["Temperature"].artifact.name == "cooling.xdmf"
    assert simulation.histories["mean_temperature"].unit == "K"
    assert simulation.histories["time_increment"].values.tolist() == [1.0, 1.0]


def test_history_is_a_transient_policy_not_a_silent_static_option():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        elasticity.isotropic_elastic(young=1.0e6, poisson=0.3, density=1.0)
    )
    request = results.history("dummy", lambda step, time: 0.0)

    with pytest.raises(TypeError, match="Unsupported Step option 'history'"):
        model.step(target=displacement, history=request)


def test_multimaterial_heat_combines_region_conduction_capacity_and_history():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (8, 2),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    regions = mesh.partition_cells(
        domain,
        metal=lambda x: x[0] < 0.5,
        insulation=lambda x: x[0] >= 0.5,
    )
    model = models.create(
        study=studies.transient_heat_transfer(dimension=2),
        mesh=domain,
        name="two_material_wall",
    )
    temperature = model.field(fields.temperature(domain, value=350.0))
    metal = constitutive.thermoelastic(
        name="metal",
        young=200.0e9,
        poisson=0.3,
        density=7800.0,
        thermal_expansion=1.2e-5,
        conductivity=40.0,
        specific_heat=500.0,
        reference_temperature=300.0,
    )
    insulation = constitutive.thermoelastic(
        name="insulation",
        young=1.0e9,
        poisson=0.25,
        density=1000.0,
        thermal_expansion=2.0e-5,
        conductivity=0.5,
        specific_heat=1000.0,
        reference_temperature=300.0,
    )
    model.material(metal, region=regions["metal"])
    model.material(insulation, region=regions["insulation"])
    model.prescribed_temperature(
        temperature,
        400.0,
        on=mesh.boundary(domain, _left, name="hot", tag=1),
    )
    model.prescribed_temperature(
        temperature,
        300.0,
        on=mesh.boundary(domain, _right, name="cold", tag=2),
    )

    assert model.conduction(temperature).kind == "partitioned_conduction"
    assert model.heat_capacity(temperature).kind == "partitioned_heat_capacity"
    step = model.step(target=temperature, dt=0.1, steps=2, progress=False)
    result = step.solve_result()

    assert result.status == "completed"
    assert len(step.accepted_times) == 2
    assert 300.0 - 1.0e-8 <= float(np.min(temperature.value.x.array))
    assert float(np.max(temperature.value.x.array)) <= 400.0 + 1.0e-8


def test_axisymmetric_static_solid_matches_lame_cylinder_displacement():
    inner_radius, outer_radius = 1.0, 2.0
    young, poisson, pressure = 1000.0, 0.3, 10.0
    domain = mesh.rectangle(
        (inner_radius, 0.0),
        (outer_radius, 0.2),
        (8, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="axisymmetric"),
        mesh=domain,
        name="lame_axisymmetric_cylinder",
    )
    displacement = model.field(fields.displacement(domain, degree=2))
    material = model.material(
        elasticity.isotropic_elastic(
            young=young,
            poisson=poisson,
            density=1.0,
        )
    )
    model.constraint(constraints.axisymmetric_plane_strain(displacement))
    model.pressure(
        pressure,
        on=mesh.face(
            domain,
            axis="x",
            value=inner_radius,
            name="inner_wall",
            tag=1,
        ),
    )

    report = model.validate()
    assert report.is_valid
    result = model.step(target=displacement).solve_result()

    radial_coordinates = displacement.space.tabulate_dof_coordinates()[:, 0]
    radial_values = displacement.value.x.array.reshape((-1, 2))[:, 0]
    lame_a = pressure * inner_radius**2 / (
        outer_radius**2 - inner_radius**2
    )
    lame_b = pressure * inner_radius**2 * outer_radius**2 / (
        outer_radius**2 - inner_radius**2
    )
    expected = (1.0 + poisson) / young * (
        (1.0 - 2.0 * poisson) * lame_a * radial_coordinates
        + lame_b / radial_coordinates
    )
    relative_error = np.linalg.norm(radial_values - expected) / np.linalg.norm(expected)

    assert result.status == "completed"
    assert relative_error < 2.0e-3
    assert result.fields["S"].field.ufl_shape == (3, 3)


def test_transient_heat_automatically_updates_amplitude_driven_source():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.transient_heat_transfer(dimension=2),
        mesh=domain,
        name="ramped_heating",
    )
    temperature = model.field(fields.temperature(domain, value=300.0))
    model.material(
        constitutive.thermoelastic(
            name="thermal solid",
            young=1.0e9,
            poisson=0.3,
            density=1000.0,
            thermal_expansion=1.0e-5,
            conductivity=10.0,
            specific_heat=500.0,
            reference_temperature=300.0,
        )
    )
    source = model.heat_source(
        1.0e5,
        amplitude=amplitudes.ramp(
            0.0,
            1.0,
            start_time=0.0,
            end_time=2.0,
            name="heating_ramp",
        ),
    )
    prescribed = model.prescribed_temperature(
        temperature,
        amplitudes.ramp(
            300.0,
            310.0,
            start_time=0.0,
            end_time=2.0,
            name="boundary_temperature_ramp",
        ),
        on=mesh.boundary(domain, _left, name="heated_edge", tag=1),
    )
    initial_mean = float(np.mean(temperature.value.x.array))
    step = model.step(
        target=temperature,
        dt=1.0,
        steps=2,
        progress=False,
    )
    step.solve()

    assert source.summary()["kind"] == "amplitude_load"
    assert float(source.scale.value) == 1.0
    assert float(prescribed.constant.value) == 310.0
    assert float(np.mean(temperature.value.x.array)) > initial_mean


def test_named_amplitudes_are_shared_by_loads_and_prescribed_values():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.transient_heat_transfer(dimension=2),
        mesh=domain,
        name="named_amplitudes",
    )
    temperature = model.field(fields.temperature(domain, value=300.0))
    source_history = model.amplitude(
        "source_history",
        amplitudes.ramp(0.0, 1.0),
    )
    temperature_history = model.amplitude(
        "temperature_history",
        amplitudes.ramp(300.0, 310.0),
    )
    source = model.heat_source(1.0e5, amplitude="source_history")
    prescribed = model.prescribed_temperature(
        temperature,
        "temperature_history",
        on=mesh.boundary(domain, _left, name="left", tag=1),
    )

    assert source.amplitude is source_history
    assert prescribed.amplitude is temperature_history
    assert tuple(item.name for item in model.amplitudes) == (
        "source_history",
        "temperature_history",
    )


def test_explicit_dynamics_consumes_model_constraints_and_amplitude_loads(tmp_path):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_strain",
            method="explicit",
        ),
        mesh=domain,
        name="explicit_ramped_traction",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        elasticity.isotropic_elastic(
            young=1.0e6,
            poisson=0.3,
            density=1000.0,
        )
    )
    model.clamp(
        displacement,
        on=mesh.boundary(domain, _left, name="left", tag=1),
    )
    traction = model.traction(
        (1.0e3, 0.0),
        on=mesh.boundary(domain, _right, name="right", tag=2),
        amplitude=amplitudes.ramp(
            0.0,
            1.0,
            start_time=0.0,
            end_time=2.0e-4,
        ),
    )
    step = model.step(
        target=displacement,
        dt=1.0e-4,
        steps=2,
        progress=False,
    )
    simulation = step.solve_result(output=tmp_path / "explicit.xdmf")

    assert float(traction.scale.value) == 1.0
    assert len(step.accepted_times) == 2
    assert float(np.max(displacement.value.x.array)) > 0.0
    assert simulation.artifacts["fields_xdmf"].is_file()
    assert simulation.artifacts["fields_hdf5"].is_file()
    assert {"Displacement", "Velocity", "Acceleration"}.issubset(
        simulation.fields
    )
    field_output = simulation.metadata["field_output"]
    assert field_output["warp_field"] == "U"
    assert field_output["warp_field_semantic"] == "Displacement"
    assert field_output["physical_components"] == 2
    assert field_output["stored_components"] == 3
    assert field_output["physical_model_dimension"] == 2
    assert field_output["geometry_dimension"] == 3
    assert field_output["warp_compatible"] is True
    assert field_output["recommended_visualization_artifact"].endswith(".xdmf")
    assert field_output["visualization_requires_extract_block"] is False
    assert field_output["scientific_xdmf_layout"] == "single_uniform_grid"


def test_common_dynamic_study_factory_keeps_physics_and_procedure_distinct():
    explicit = studies.dynamic_solid(
        dimension=2,
        assumption="plane_stress",
        method="explicit",
    )
    implicit = studies.dynamic_solid(
        dimension=2,
        assumption="plane_stress",
        method="generalized_alpha",
    )

    assert explicit.physics == implicit.physics == "solid_mechanics"
    assert explicit.analysis == implicit.analysis == "second_order_dynamics"
    assert explicit.preferred_procedure == "central_difference"
    assert implicit.preferred_procedure == "generalized_alpha"


def test_explicit_solution_procedure_drives_capability_and_lowering():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_stress",
            method="newmark",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        elasticity.isotropic_elastic(
            young=1.0e6,
            poisson=0.3,
            density=1000.0,
        )
    )
    selected = procedures.central_difference()

    capability = models.step_capability(
        model,
        target=displacement,
        procedure=selected,
    )
    step = model.step(
        target=displacement,
        procedure=selected,
        dt=1.0e-4,
        steps=1,
        progress=False,
    )

    assert capability["provider"]["name"] == "central_difference_explicit_dynamics"
    assert capability["procedure"]["algorithm"] == "central_difference"
    assert step.procedure is selected


def test_conflicting_method_and_solution_procedure_fail_before_lowering():
    from agentfem.step_providers import lower_step

    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_stress",
            method="explicit",
        )
    )
    with pytest.raises(ValueError, match="not conflicting numerical routes"):
        lower_step(
            model,
            analysis="second_order_dynamics",
            target=None,
            options={"method": "newmark"},
            procedure=procedures.central_difference(),
        )


def test_solution_procedure_resolution_rejects_silent_route_changes():
    with pytest.raises(ValueError, match="requires 'static'"):
        procedures.resolve(
            analysis="linear_static",
            requested=procedures.central_difference(),
        )
    with pytest.raises(ValueError, match="Unknown numerical method"):
        procedures.resolve(
            analysis="second_order_dynamics",
            requested="mystery_integrator",
        )
    selected = procedures.resolve(
        analysis="nonlinear_transient",
        requested="backward_euler_newton",
        stateful=True,
    )
    assert selected.algorithm == "backward_euler_newton"


def test_implicit_dynamics_rejects_time_dependent_supports_before_solve():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_stress",
            method="newmark",
        ),
        mesh=domain,
        name="moving_implicit_support",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        elasticity.isotropic_elastic(
            young=1.0e6,
            poisson=0.3,
            density=1000.0,
        )
    )
    model.prescribe(
        displacement,
        amplitudes.ramp(0.0, 0.01),
        component=0,
        on=mesh.boundary(domain, _left, name="left", tag=1),
    )

    report = model.validate()

    assert not report.is_valid
    assert "AFM-CONSTRAINT-002" in {item.code for item in report.errors}


def test_periodic_projection_declares_and_enforces_procedure_capabilities():
    periodic = constraints.PeriodicProjectionConstraint(
        pairs=(),
        pair_count=0,
        maximum_coordinate_mismatch=2.0e-14,
    )

    explicit = constraints.validate_solver_compatibility(
        constraints=(periodic,),
        analysis="second_order_dynamics",
        procedure="central_difference",
        comm_size=1,
    )
    implicit = constraints.validate_solver_compatibility(
        constraints=(periodic,),
        analysis="second_order_dynamics",
        procedure="newmark",
        comm_size=1,
    )
    parallel = constraints.validate_solver_compatibility(
        constraints=(periodic,),
        analysis="second_order_dynamics",
        procedure="central_difference",
        comm_size=2,
    )

    assert explicit.is_valid
    assert {item.code for item in implicit.errors} == {
        "AFM-CONSTRAINT-PROCEDURE-001"
    }
    assert {item.code for item in parallel.errors} == {
        "AFM-CONSTRAINT-PARALLEL-001"
    }
    assert periodic.summary()["capabilities"] == {
        "kind": "periodic_constraint",
        "enforcement": "nodal_pair_projection",
        "analyses": ("second_order_dynamics",),
        "procedures": ("central_difference",),
        "strict": False,
        "supports_parallel": False,
        "reaction_evidence": "unavailable",
        "work_evidence": "unavailable",
    }
    assert periodic.diagnostics()["maximum_coordinate_mismatch"] == pytest.approx(
        2.0e-14
    )


def test_constraint_balance_contract_refuses_partial_mpc_reactions():
    periodic = constraints.RectangularPeriodicMPC(
        backend=object(),
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
        axes=(0,),
        tolerance=1.0e-12,
        name="periodic_x",
    )

    contract = constraints.constraint_balance_contract((periodic,))

    assert contract["force_balance_available"] is False
    assert contract["work_balance_available"] is False
    assert contract["force_balance_gaps"] == ("periodic_x",)
    assert contract["constraints"][0]["reaction_evidence"] == (
        "provider_dual_required"
    )


def test_constraint_balance_contract_accepts_complete_provider_dual_path():
    periodic = constraints.RectangularPeriodicMPC(
        backend=object(),
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
        axes=(0,),
        tolerance=1.0e-12,
        name="periodic_x",
    )
    dual = constraints.constraint_dual(
        periodic,
        role="mpc_constraint",
        force=(12.0,),
        coordinate=(0.02,),
        resultant=(-12.0, 0.0),
        source="affine_reduced_equilibrium",
    )

    contract = constraints.constraint_balance_contract(
        (periodic,),
        provider_duals=(dual,),
    )

    assert contract["force_balance_available"] is True
    assert contract["work_balance_available"] is True
    assert contract["force_balance_gaps"] == ()
    assert dual.work_sample().summary() == {
        "name": "periodic_x",
        "role": "mpc_constraint",
        "force": [12.0],
        "displacement": [0.02],
    }


def test_constraint_provider_publishes_its_own_converged_dual_evidence():
    class ProviderOwnedConstraint:
        name = "weak_support"

        def capabilities(self):
            return constraints.ConstraintCapabilities(
                kind="weak_constraint",
                enforcement="nitsche",
                reaction_evidence="provider_dual_required",
                work_evidence="provider_dual_path_required",
            )

        def dual_evidence(self, problem):
            assert problem == "converged_problem"
            return constraints.constraint_dual(
                self,
                role="weak_constraint",
                force=(8.0,),
                coordinate=(0.025,),
                resultant=(-8.0, 0.0),
                source="nitsche_boundary_traction",
            )

    selected = ProviderOwnedConstraint()
    duals = constraints.collect_provider_duals(
        (selected,),
        "converged_problem",
    )
    contract = constraints.constraint_balance_contract(
        (selected,),
        provider_duals=duals,
    )

    assert len(duals) == 1
    assert duals[0].source == "nitsche_boundary_traction"
    assert contract["force_balance_available"] is True
    assert contract["work_balance_available"] is True


def test_constraint_dual_collection_rejects_wrong_owner_and_duplicates():
    class WrongOwner:
        name = "declared"

        def dual_evidence(self, _problem):
            return constraints.constraint_dual(
                "another_constraint",
                force=(1.0,),
                coordinate=(0.0,),
            )

    with pytest.raises(ValueError, match="does not match"):
        constraints.collect_provider_duals((WrongOwner(),), object())

    periodic = constraints.RectangularPeriodicMPC(
        backend=object(),
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
        axes=(0,),
        tolerance=1.0e-12,
        name="periodic_x",
    )
    duplicate = constraints.constraint_dual(
        periodic,
        force=(1.0,),
        coordinate=(0.0,),
        resultant=(0.0, 0.0),
    )
    with pytest.raises(ValueError, match="must be unique"):
        constraints.collect_provider_duals(
            (periodic,),
            object(),
            extra=(duplicate, duplicate),
        )

    wrong_role = constraints.constraint_dual(
        periodic,
        role="contact_constraint",
        force=(1.0,),
        coordinate=(0.0,),
        resultant=(0.0, 0.0),
    )
    with pytest.raises(ValueError, match="roles do not match"):
        constraints.collect_provider_duals(
            (periodic,),
            object(),
            extra=(wrong_role,),
        )


def test_constraint_balance_contract_rejects_partial_or_unmatched_duals():
    periodic = constraints.RectangularPeriodicMPC(
        backend=object(),
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
        axes=(0,),
        tolerance=1.0e-12,
        name="periodic_x",
    )
    force_only = constraints.constraint_dual(
        periodic,
        force=(12.0,),
        resultant=(-12.0, 0.0),
        source="force_only_provider",
    )
    contract = constraints.constraint_balance_contract(
        (periodic,),
        provider_duals=(force_only,),
    )
    assert contract["force_balance_available"] is True
    assert contract["work_balance_available"] is False

    generalized_only = constraints.constraint_dual(
        periodic,
        force=(12.0,),
        coordinate=(0.02,),
        source="generalized_force_without_spatial_resultant",
    )
    contract = constraints.constraint_balance_contract(
        (periodic,),
        provider_duals=(generalized_only,),
    )
    assert contract["force_balance_available"] is False
    assert contract["work_balance_available"] is True

    unrelated = constraints.constraint_dual(
        "another_constraint",
        force=(1.0,),
        coordinate=(0.0,),
    )
    with pytest.raises(ValueError, match="does not match"):
        constraints.constraint_balance_contract(
            (periodic,),
            provider_duals=(unrelated,),
        )

    duplicate_name = constraints.PeriodicConstraintSpec(
        slave_marker=lambda x: x,
        master_marker=lambda x: x,
        map_slave_to_master=lambda x: x,
        name="periodic_x",
    )
    with pytest.raises(ValueError, match="names must be unique"):
        constraints.constraint_balance_contract((periodic, duplicate_name))


def test_model_rejects_periodic_projection_for_newmark_before_problem_build():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_stress",
            method="newmark",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        elasticity.isotropic_elastic(
            young=1.0e6,
            poisson=0.3,
            density=1000.0,
        )
    )
    bottom = mesh.face(domain, axis="y", value=0.0, name="bottom", tag=1)
    top = mesh.face(domain, axis="y", value=0.2, name="top", tag=2)
    model.periodic(
        displacement,
        master=bottom,
        slave=top,
        match_axis="x",
        method="projection",
    )

    with pytest.raises(
        ValueError,
        match="AFM-CONSTRAINT-PROCEDURE-001",
    ):
        model.step(target=displacement, dt=1.0e-4, steps=2)
