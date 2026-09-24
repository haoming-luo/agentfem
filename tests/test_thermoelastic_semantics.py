# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Thermoelastic eigenstrain, partition and support semantics."""

from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import (
    constitutive,
    eigenstrains,
    fields,
    mesh,
    models,
    results,
    studies,
    units,
)


def _thermoelastic(*, name, young=1.0e6, alpha=1.0e-5):
    return constitutive.thermoelastic(
        name=name,
        young=young,
        poisson=0.25,
        density=1.0,
        thermal_expansion=alpha,
        conductivity=1.0,
        specific_heat=1.0,
        reference_temperature=300.0,
    )


def _free_expansion_model(*, partitioned: bool):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (8, 2),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"),
        mesh=domain,
        name="free_thermal_expansion",
        units=units.si(),
    )
    displacement = model.field(fields.displacement(domain, degree=2))
    if partitioned:
        regions = mesh.partition_cells(
            domain,
            left=mesh.layer("x", upper=0.5),
            right=mesh.layer("x", lower=0.5),
        )
        model.material(_thermoelastic(name="left", young=1.0e6), region=regions.left)
        model.material(_thermoelastic(name="right", young=2.0e6), region=regions.right)
    else:
        model.material(_thermoelastic(name="solid"))
    left = mesh.face(domain, axis="x", value=0.0, name="left", tag=1)
    model.fix(displacement, on=left, component=0)
    model.pin(displacement, at=(0.0, 0.0), components=1, tolerance=1.0e-9)
    temperature = fields.temperature(domain, value=400.0)
    model.eigenstrain(eigenstrains.thermal(temperature))
    return model, displacement


@pytest.mark.parametrize("partitioned", (False, True))
def test_free_expansion_has_zero_physical_stress_and_explicit_strain_split(
    partitioned,
):
    model, displacement = _free_expansion_model(partitioned=partitioned)

    simulation = model.step(target=displacement).solve_result()

    coordinates = displacement.space.tabulate_dof_coordinates()
    values = displacement.value.x.array.reshape((-1, 2))
    expected = 1.0e-3 * coordinates[:, :2]
    np.testing.assert_allclose(values, expected, rtol=2.0e-8, atol=2.0e-11)
    assert {"S", "E_TOTAL", "E_EIGEN", "E_MECH", "MISES"}.issubset(simulation.fields)
    assert np.max(np.abs(simulation.fields["S"].field.x.array)) < 1.0e-6
    np.testing.assert_allclose(
        simulation.fields["E_EIGEN"].field.x.array,
        simulation.fields["E_TOTAL"].field.x.array,
        rtol=2.0e-8,
        atol=2.0e-11,
    )
    assert np.max(np.abs(simulation.fields["E_MECH"].field.x.array)) < 1.0e-11
    processing = simulation.fields["S"].processing
    assert simulation.fields["Displacement"].unit == "m"
    assert simulation.fields["S"].unit == "kg/(m*s^2)"
    assert simulation.fields["E_MECH"].unit == "1"
    assert processing["material_boundary_averaging"] is False
    assert len(processing["material_partition"]) == (2 if partitioned else 1)
    if partitioned:
        assert {
            item["region"] for item in processing["material_partition"]
        } == {"left", "right"}
    assert processing["eigenstrain_sources"][0]["source"] == "thermal"


def test_multimaterial_thermal_operator_is_one_partitioned_operator():
    model, displacement = _free_expansion_model(partitioned=True)
    temperature = model.eigenstrains[0].temperature

    operator = model.thermal_expansion(displacement, temperature)

    assert operator.kind == "partitioned_eigenstrain"
    assert len(model.eigenstrains) == 1
    assert len(operator.parts) == 2
    assert {part["operator"]["metadata"]["material"] for part in operator.parts} == {
        "left",
        "right",
    }


def test_thermal_source_rejects_material_without_expansion_contract_pre_solve():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        constitutive.isotropic_elastic(
            young=1.0e6,
            poisson=0.25,
            density=1.0,
        )
    )
    model.eigenstrain(eigenstrains.thermal(fields.temperature(domain, value=400.0)))

    with pytest.raises(TypeError, match="thermal_expansion.*reference_temperature"):
        model.step(target=displacement)


def test_material_partition_overlap_and_gap_fail_before_assembly():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (4, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    left = mesh.cells(domain, name="left", where=mesh.layer("x", upper=0.7), tag=1)
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"),
        mesh=domain,
    )
    model.field(fields.displacement(domain))
    model.material(_thermoelastic(name="a"), region=left)
    model.material(_thermoelastic(name="b"), region=left)

    report = model.validate()
    codes = {item.code for item in report.issues}

    assert "AFM-MATERIAL-004" in codes
    assert "AFM-MATERIAL-005" in codes


def test_rigid_mode_audit_is_diagnostic_and_pin_roller_recipe_is_stable():
    model, displacement = _free_expansion_model(partitioned=False)

    audit = model.rigid_mode_audit(displacement)

    assert audit.total_modes == 3
    assert audit.remaining_modes == 0
    assert audit.stable


def test_empty_field_variable_tuple_suppresses_only_derived_fields():
    model, displacement = _free_expansion_model(partitioned=False)

    simulation = model.step(target=displacement).solve_result(field_variables=())

    assert set(simulation.fields) == {"Displacement"}
    assert "S" not in simulation.fields


def test_result_catalog_keeps_one_physical_stress_and_explicit_strain_names():
    assert results.field_variable("S").name == "CauchyStress"
    assert results.field_variable("E_TOTAL").name == "TotalInfinitesimalStrain"
    assert results.field_variable("E_EIGEN").name == "Eigenstrain"
    assert results.field_variable("E_MECH").derived_from == (
        "E_TOTAL",
        "E_EIGEN",
    )


def test_plane_strain_fully_constrained_heating_uses_three_dimensional_trace():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.2),
        (2, 1),
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    material = _thermoelastic(name="solid", young=1.0e6, alpha=1.0e-5)
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"),
        mesh=domain,
        name="constrained_plane_strain_heating",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(material)
    model.fix(
        displacement,
        location=lambda x: np.full(x.shape[1], True, dtype=bool),
    )
    model.eigenstrain(eigenstrains.thermal(fields.temperature(domain, value=400.0)))

    simulation = model.step(target=displacement).solve_result()

    alpha_delta = 1.0e-3
    expected_normal_stress = -(
        2.0 * material.mu + 3.0 * material.lambda_
    ) * alpha_delta
    stress = simulation.fields["S"].field.x.array.reshape((-1, 2, 2))
    np.testing.assert_allclose(stress[:, 0, 0], expected_normal_stress, rtol=1.0e-11)
    np.testing.assert_allclose(stress[:, 1, 1], expected_normal_stress, rtol=1.0e-11)
    assert np.max(np.abs(stress[:, 0, 1])) < 1.0e-12
    assert np.max(np.abs(simulation.fields["E_TOTAL"].field.x.array)) < 1.0e-13
    np.testing.assert_allclose(
        simulation.fields["E_MECH"].field.x.array,
        -simulation.fields["E_EIGEN"].field.x.array,
        rtol=0.0,
        atol=1.0e-13,
    )


def test_three_dimensional_free_expansion_uses_the_same_public_contract():
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 0.2, 0.1),
        (2, 1, 1),
        comm=MPI.COMM_SELF,
        cell_type="hexahedron",
    )
    model = models.create(
        study=studies.static_solid(dimension=3),
        mesh=domain,
        name="free_thermal_expansion_3d",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(_thermoelastic(name="solid"))
    for axis, component in (("x", 0), ("y", 1), ("z", 2)):
        model.fix(
            displacement,
            on=mesh.face(domain, axis=axis, value=0.0),
            component=component,
        )
    model.eigenstrain(eigenstrains.thermal(fields.temperature(domain, value=400.0)))

    simulation = model.step(target=displacement).solve_result()

    coordinates = displacement.space.tabulate_dof_coordinates()
    values = displacement.value.x.array.reshape((-1, 3))
    np.testing.assert_allclose(values, 1.0e-3 * coordinates, rtol=2.0e-8, atol=2.0e-11)
    assert np.max(np.abs(simulation.fields["S"].field.x.array)) < 1.0e-6
    assert simulation.fields["S"].processing["material_boundary_averaging"] is False
