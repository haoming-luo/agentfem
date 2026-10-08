# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import (
    constitutive,
    eigenstrains,
    fields,
    mesh,
    models,
    state,
    steps,
    studies,
)


def _domain():
    return mesh.rectangle(
        (0.0, 0.0), (1.0, 0.2), (4, 2), comm=MPI.COMM_WORLD, cell_type="quadrilateral"
    )


def test_same_length_different_mesh_rejected_without_mutation():
    a, b = _domain(), _domain()
    source, target = (
        fields.temperature(a, value=400.0),
        fields.temperature(b, value=300.0),
    )
    stage = steps.engineering_step("transfer")
    stage.predefine(target, source, method="interpolate")
    with pytest.raises(ValueError, match="AFM-TRANSFER-005"):
        stage.apply_predefined_fields()
    np.testing.assert_array_equal(target.value.x.array, 300.0)


def test_copy_requires_identical_space_and_interpolation_is_explicit():
    domain = _domain()
    source = fields.temperature(domain)
    source.value.interpolate(lambda x: 300.0 + 40.0 * x[0])
    target = fields.temperature(domain, degree=2, value=100.0)
    stage = steps.engineering_step("transfer")
    stage.predefine(target, source)
    with pytest.raises(ValueError, match="same function space"):
        stage.apply_predefined_fields()
    np.testing.assert_array_equal(target.value.x.array, 100.0)
    stage.predefine(
        target, source, method="interpolate", source_time=1.0, target_time=1.0
    )
    stage.apply_predefined_fields()
    coordinates = target.space.tabulate_dof_coordinates()
    np.testing.assert_allclose(target.value.x.array, 300.0 + 40.0 * coordinates[:, 0])
    assert stage.summary()["field_transfers"][0]["time_status"] == "declared_equal"


@pytest.mark.parametrize(
    "source_time,target_time", [(1.0, 2.0), (1.0, None), (np.nan, np.nan)]
)
def test_time_mismatch_is_rejected(source_time, target_time):
    with pytest.raises(ValueError, match="AFM-TRANSFER-004"):
        steps.engineering_step("bad-time").predefine(
            object(), 1.0, source_time=source_time, target_time=target_time
        )


def test_batch_assignment_is_atomic_on_rank_local_failure():
    domain = _domain()
    a, b = (
        fields.temperature(domain, value=10.0),
        fields.temperature(domain, value=20.0),
    )
    stage = steps.engineering_step("atomic")
    stage.predefine(a, 50.0, name="a")
    stage.predefine(b, np.nan if domain.comm.rank == 0 else 60.0, name="b")
    with pytest.raises(ValueError, match="AFM-TRANSFER-007"):
        stage.apply_predefined_fields()
    np.testing.assert_array_equal(a.value.x.array, 10.0)
    np.testing.assert_array_equal(b.value.x.array, 20.0)


def test_downstream_failure_rolls_back_on_all_ranks():
    domain = _domain()
    source = fields.temperature(domain, value=400.0)
    target = fields.temperature(domain, value=300.0)
    u = fields.displacement(domain)
    stage = steps.engineering_step("mechanical")
    stage.predefine(target, source, method="interpolate")
    with pytest.raises(RuntimeError):
        with stage.field_transaction(displacement=u):
            stage.apply_predefined_fields()
            u.value.x.array[:] = 3.0
            if domain.comm.rank == 0:
                raise RuntimeError("deliberate downstream rejection")
    np.testing.assert_array_equal(target.value.x.array, 300.0)
    np.testing.assert_array_equal(u.value.x.array, 0.0)
    np.testing.assert_array_equal(source.value.x.array, 400.0)


def _material():
    return constitutive.thermoelastic(
        young=1.0e6,
        poisson=0.25,
        density=1.0,
        thermal_expansion=1.0e-5,
        conductivity=1.0,
        specific_heat=1.0,
        reference_temperature=300.0,
    )


@pytest.mark.parametrize("restrained", [False, True])
@pytest.mark.parametrize("gradient", [0.0, 100.0])
def test_solved_heat_to_mechanical_response_and_result_evidence(restrained, gradient):
    domain = _domain()
    material = _material()
    heat = models.create(study=studies.steady_heat_transfer(dimension=2), mesh=domain)
    temperature = heat.field(fields.temperature(domain, value=300.0))
    heat.material(material)
    left = mesh.face(domain, axis="x", value=0.0, name="left", tag=1)
    right = mesh.face(domain, axis="x", value=1.0, name="right", tag=2)
    heat.prescribed_temperature(temperature, on=left, value=400.0)
    heat.prescribed_temperature(temperature, on=right, value=400.0 + gradient)
    heat.step(target=temperature).solve_result()
    temperature_x = temperature.space.tabulate_dof_coordinates()[:, 0]
    np.testing.assert_allclose(
        temperature.value.x.array, 400.0 + gradient * temperature_x, atol=1.0e-10
    )
    solid = models.create(
        study=studies.static_solid(
            dimension=2, assumption="plane_strain" if restrained else "plane_stress"
        ),
        mesh=domain,
    )
    u = solid.field(fields.displacement(domain, degree=2))
    solid.material(material)
    copied_temperature = fields.temperature(domain, value=300.0)
    solid.eigenstrain(eigenstrains.thermal(copied_temperature))
    if restrained:
        solid.fix(u, location=lambda x: np.full(x.shape[1], True))
    else:
        bottom = mesh.face(domain, axis="y", value=0.0, name="bottom", tag=3)
        solid.fix(u, on=bottom, component=1)
        solid.pin(u, at=(0.0, 0.0), components=0, tolerance=1.0e-9)
    stage = solid.stage("thermal-to-solid")
    stage.predefine(copied_temperature, temperature, method="interpolate")
    with stage.field_transaction(displacement=u):
        result = solid.step(target=u, configuration=stage).solve_result()
    stress = result.fields["S"].field.x.array.reshape(-1, 2, 2)
    if restrained:
        stress_x = result.fields["S"].field.function_space.tabulate_dof_coordinates()[
            :, 0
        ]
        expected = (
            -(2.0 * material.mu + 3.0 * material.lambda_)
            * 1.0e-5
            * (100.0 + gradient * stress_x)
        )
        np.testing.assert_allclose(stress[:, 0, 0], expected, rtol=1.0e-10)
        np.testing.assert_allclose(stress[:, 1, 1], expected, rtol=1.0e-10)
        np.testing.assert_allclose(u.value.x.array, 0.0, atol=1.0e-12)
    else:
        coordinates = u.space.tabulate_dof_coordinates()
        x, y = coordinates[:, 0], coordinates[:, 1]
        expected = 1.0e-5 * np.column_stack(
            (100.0 * x + 0.5 * gradient * (x * x - y * y), 100.0 * y + gradient * x * y)
        )
        np.testing.assert_allclose(
            u.value.x.array.reshape(-1, 2), expected, atol=1.0e-11
        )
        np.testing.assert_allclose(stress, 0.0, atol=1.0e-6)
    evidence = result.metadata["engineering_step"]["field_transfers"][0]
    assert evidence["mesh_relation"] == "same_runtime_mesh"
    assert evidence["scope"] == "snapshot_not_live_binding"
    assert evidence["time_status"] == "unspecified"


def test_failed_lowering_restores_predefined_fields():
    domain = _domain()
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"), mesh=domain
    )
    u = model.field(fields.displacement(domain))
    model.material(_material())
    t = fields.temperature(domain, value=300.0)
    stage = model.stage("failed-build")
    stage.predefine(t, 400.0)
    with pytest.raises((ValueError, TypeError)):
        model.step(target=u, configuration=stage, definitely_invalid_option=True)
    np.testing.assert_array_equal(t.value.x.array, 300.0)


def test_transaction_rejects_nonfinite_success_and_keeps_accepted_fields():
    target = fields.temperature(_domain(), value=300.0)
    with pytest.raises((ValueError, RuntimeError)):
        with state.field_transaction(temperature=target):
            target.value.x.array[:] = np.nan
    np.testing.assert_array_equal(target.value.x.array, 300.0)


def test_restored_heat_field_can_be_transferred_at_its_accepted_time(tmp_path):
    from test_first_order_operator_lifecycle import _heat_step
    from pathlib import Path
    from dolfinx import fem

    comm = MPI.COMM_WORLD
    path = Path(comm.bcast(str(tmp_path / "heat-checkpoint"), root=0))
    source = _heat_step("auto", steps=4, comm=comm)
    restored = _heat_step("auto", steps=4, comm=comm)
    try:
        source.run(until_step=2)
        source.save_checkpoint(path)
        restored.load_checkpoint(path)
        restored.run()
        source.run()
        np.testing.assert_allclose(restored.current.x.array, source.current.x.array)
        destination = fem.Function(restored.current.function_space)
        stage = steps.engineering_step("restart-transfer")
        accepted_time = restored.completed_steps * restored.dt
        stage.predefine(
            destination,
            restored.current,
            source_time=accepted_time,
            target_time=accepted_time,
        )
        stage.apply_predefined_fields()
        np.testing.assert_array_equal(destination.x.array, restored.current.x.array)
        assert stage.transfer_evidence[0]["source_time"] == accepted_time
    finally:
        source.close()
        restored.close()


def test_component_mismatch_rejected():
    domain = _domain()
    t = fields.temperature(domain, value=300.0)
    stage = steps.engineering_step("shape")
    stage.predefine(t, fields.displacement(domain), method="interpolate")
    with pytest.raises(ValueError, match="AFM-TRANSFER-006"):
        stage.apply_predefined_fields()
    np.testing.assert_array_equal(t.value.x.array, 300.0)


def test_new_stage_rebuilds_temperature_dependent_stiffness():
    from agentfem import materials

    domain = _domain()
    material = constitutive.temperature_dependent_thermoelastic(
        young=materials.temperature_property(
            [300.0, 500.0], [1.0e6, 2.0e6], extrapolation="constant"
        ),
        poisson=0.25,
        density=1.0,
        thermal_expansion=1.0e-5,
        conductivity=1.0,
        specific_heat=1.0,
        reference_temperature=300.0,
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"), mesh=domain
    )
    u = model.field(fields.displacement(domain))
    model.material(material)
    model.fix(u, location=lambda x: np.full(x.shape[1], True))
    temperature = fields.temperature(domain, value=300.0)
    model.eigenstrain(eigenstrains.thermal(temperature))
    stage = model.stage("changing-temperature")
    for selected, young in [(400.0, 1.5e6), (500.0, 2.0e6)]:
        stage.predefine(temperature, selected)
        with stage.field_transaction(displacement=u):
            result = model.step(target=u, configuration=stage).solve_result()
        stress = result.fields["S"].field.x.array.reshape(-1, 2, 2)
        expected = -young / (1.0 - 2.0 * 0.25) * 1.0e-5 * (selected - 300.0)
        np.testing.assert_allclose(stress[:, 0, 0], expected, rtol=1.0e-10)


def test_snapshot_assignments_do_not_cascade_or_alias():
    from dolfinx import fem

    a = fields.temperature(_domain(), value=10.0)
    b = fem.Function(a.space)
    b.x.array[:] = 20.0
    stage = steps.engineering_step("swap")
    stage.predefine(a, b, name="a")
    stage.predefine(b, a, name="b")
    stage.apply_predefined_fields()
    np.testing.assert_array_equal(a.value.x.array, 20.0)
    np.testing.assert_array_equal(b.x.array, 10.0)


def test_multimaterial_solved_heat_uses_regional_temperature_dependent_stress():
    """Two conductivities and moduli: compare cell stresses to exact integrals.

    Full restraint isolates regional constitutive/transfer semantics; this is
    not a free bilayer bending benchmark. E(T)*(T-Tref) is quadratic, so DG0
    must match its cell average, not merely its value at the cell centre.
    """
    from agentfem import materials

    domain = _domain()
    regions = mesh.partition_cells(
        domain,
        left=mesh.layer("x", upper=0.5),
        right=mesh.layer("x", lower=0.5),
    )
    heat = models.create(study=studies.steady_heat_transfer(dimension=2), mesh=domain)
    solid = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"), mesh=domain
    )
    temperature = heat.field(fields.temperature(domain))
    displacement = solid.field(fields.displacement(domain, degree=2))
    for region, conductivity, base, alpha in (
        (regions.left, 1.0, 1.0e6, 1.0e-5),
        (regions.right, 2.0, 3.0e6, 2.0e-5),
    ):
        material = constitutive.temperature_dependent_thermoelastic(
            young=materials.temperature_property(
                [300.0, 500.0], [base, 2.0 * base], extrapolation="constant"
            ),
            poisson=0.25, density=1.0, thermal_expansion=alpha,
            conductivity=conductivity, specific_heat=1.0, reference_temperature=300.0,
        )
        heat.material(material, region=region)
        solid.material(material, region=region)
    for x, value in ((0.0, 350.0), (1.0, 450.0)):
        heat.prescribed_temperature(
            temperature, on=mesh.face(domain, axis="x", value=x), value=value
        )
    heat_result = heat.step(target=temperature).solve_result()
    x = temperature.space.tabulate_dof_coordinates()[:, 0]
    expected_temperature = np.where(
        x <= 0.5, 350.0 + (400.0 / 3.0) * x,
        350.0 + 200.0 / 3.0 + (200.0 / 3.0) * (x - 0.5),
    )
    np.testing.assert_allclose(temperature.value.x.array, expected_temperature, atol=1e-10)
    target = fields.temperature(domain, degree=2, value=300.0)
    solid.eigenstrain(eigenstrains.thermal(target))
    solid.fix(displacement, location=lambda x: np.full(x.shape[1], True))
    stage = solid.stage("partitioned-thermal-to-solid")
    stage.predefine(target, temperature, method="interpolate")
    with stage.field_transaction(displacement=displacement):
        result = solid.step(target=displacement, configuration=stage).solve_result()
    stress_field = result.fields["S"].field
    centers = stress_field.function_space.tabulate_dof_coordinates()[:, 0]
    left = centers < 0.5
    slope = np.where(left, 400.0 / 3.0, 200.0 / 3.0)
    delta = np.where(left, 50.0 + slope * centers,
                     50.0 + 200.0 / 3.0 + slope * (centers - 0.5))
    average_delta_squared = delta**2 + slope**2 * 0.25**2 / 12.0
    expected_stress = -np.where(left, 1e6, 3e6) * np.where(left, 1e-5, 2e-5) / 0.5 * (
        delta + average_delta_squared / 200.0
    )
    stress = stress_field.x.array.reshape(-1, 2, 2)
    np.testing.assert_allclose(stress[:, 0, 0], expected_stress, rtol=1e-10)
    np.testing.assert_allclose(stress[:, 1, 1], expected_stress, rtol=1e-10)
    assert result.fields["S"].processing["material_boundary_averaging"] is False
    assert len(result.fields["S"].processing["material_partition"]) == 2
    assert heat_result.status == "completed"


def test_rejected_downstream_solve_retry_preserves_heat_and_accepted_evidence(tmp_path):
    """Restart heat once, reject a solved structural trial, retry without heat work."""
    from pathlib import Path
    from copy import deepcopy
    from test_first_order_operator_lifecycle import _heat_step

    comm = MPI.COMM_WORLD
    path = Path(comm.bcast(str(tmp_path / "accepted-heat"), root=0))
    source = _heat_step("auto", steps=4, comm=comm)
    restored = _heat_step("auto", steps=4, comm=comm)
    try:
        source.run()
        source.save_checkpoint(path)
        restored.load_checkpoint(path)
        domain = restored.current.function_space.mesh
        model = models.create(
            study=studies.static_solid(dimension=2, assumption="plane_strain"), mesh=domain
        )
        u = model.field(fields.displacement(domain))
        model.material(_material())
        model.fix(u, location=lambda x: np.full(x.shape[1], True))
        target = fields.temperature(domain, value=300.0)
        model.eigenstrain(eigenstrains.thermal(target))
        stage = model.stage("retry-solid")
        stage.predefine(target, 300.0)
        stage.apply_predefined_fields()
        accepted_evidence = deepcopy(stage.transfer_evidence)
        accepted_heat = restored.current.x.array.copy()
        counters = deepcopy(restored.operator_lifecycle_summary())
        accepted_time = restored.completed_steps * restored.dt
        stage.predefine(target, restored.current, method="interpolate",
                        source_time=accepted_time, target_time=accepted_time)
        rejected_result = None
        with pytest.raises(RuntimeError):
            with stage.field_transaction(displacement=u):
                rejected_result = model.step(target=u, configuration=stage).solve_result()
                if comm.rank == 0:
                    raise RuntimeError("deliberate downstream acceptance rejection")
        assert stage.transfer_evidence == accepted_evidence
        np.testing.assert_array_equal(target.value.x.array, 300.0)
        np.testing.assert_array_equal(u.value.x.array, 0.0)
        with stage.field_transaction(displacement=u):
            result = model.step(target=u, configuration=stage).solve_result()
        np.testing.assert_array_equal(restored.current.x.array, accepted_heat)
        assert restored.operator_lifecycle_summary() == counters
        stress = result.fields["S"].field.x.array.reshape(-1, 2, 2)
        # Uniform heating: Q/(rho*c) * (4*dt) = 0.4 K.
        np.testing.assert_allclose(stress[:, 0, 0], -8.0, atol=1e-8)
        np.testing.assert_allclose(
            result.fields["S"].field.x.array, rejected_result.fields["S"].field.x.array
        )
        evidence = deepcopy(result.metadata["engineering_step"])
        stage.predefine(target, 500.0)
        stage.apply_predefined_fields()
        assert result.metadata["engineering_step"] == evidence
        assert evidence["field_transfers"][0]["source_time"] == accepted_time
    finally:
        source.close()
        restored.close()
