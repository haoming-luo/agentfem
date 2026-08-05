from __future__ import annotations

import numpy as np
from mpi4py import MPI

from agentfem import amplitudes, constitutive, fields, mesh, models, studies
from agentfem.constitutive import elasticity


def _left(x):
    return np.isclose(x[0], 0.0)


def _right(x):
    return np.isclose(x[0], 1.0)


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


def test_steady_heat_transfer_consumes_flux_and_convection():
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
    ).solve_result()

    assert simulation.status == "completed"
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
    step = model.step(
        target=temperature,
        dt=1.0,
        steps=2,
        progress=False,
    )
    simulation = step.solve_result(output=tmp_path / "cooling.xdmf")

    assert simulation.status == "completed"
    assert float(np.mean(temperature.value.x.array)) < initial_mean
    assert float(convection.ambient_temperature.value) == 320.0
    assert simulation.artifacts["fields_xdmf"].is_file()
    assert simulation.artifacts["fields_hdf5"].is_file()
    assert simulation.fields["Temperature"].artifact.name == "cooling.xdmf"
    assert simulation.histories["time_increment"].values.tolist() == [1.0, 1.0]


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


def test_model_validation_rejects_declared_but_unimplemented_study_combination():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="axisymmetric"),
        mesh=domain,
        name="unsupported_axisymmetric",
    )
    model.field(fields.displacement(domain))
    model.material(
        elasticity.isotropic_elastic(
            young=1.0e6,
            poisson=0.3,
            density=1000.0,
        )
    )

    report = model.validate()

    assert not report.is_valid
    capability_issue = next(
        item for item in report.errors if item.code == "AFM-STUDY-002"
    )
    assert capability_issue.context["capability"]["supported"] is False


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
