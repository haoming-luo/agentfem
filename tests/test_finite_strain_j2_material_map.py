from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import (
    constitutive,
    fields,
    mesh,
    models,
    results,
    solvers,
    steps,
    studies,
)
from agentfem.step_providers import step_capability

from periodic_cube_fixture import periodic_unit_cube


def _materials():
    return (
        constitutive.finite_strain_j2_logarithmic(
            young=200_000.0,
            poisson=0.3,
            yield_stress=180.0,
            hardening_modulus=2_000.0,
        ),
        constitutive.finite_strain_j2_logarithmic(
            young=120_000.0,
            poisson=0.3,
            yield_stress=120.0,
            hardening_modulus=1_000.0,
        ),
    )


def _regional_map(domain):
    regions = mesh.partition_cells(
        domain,
        left=lambda x: x[0] <= 0.5,
        right=lambda x: x[0] > 0.5,
    )
    left, right = _materials()
    assignments = (
        SimpleNamespace(item=left, region=regions.left),
        SimpleNamespace(item=right, region=regions.right),
    )
    return constitutive.QuadratureMaterialMap.from_assignments(
        domain,
        assignments,
        material_type=constitutive.FiniteStrainJ2Logarithmic,
    )


def _point(material, gradient, state):
    return constitutive.MaterialPointInput(
        deformation_gradient_old=np.eye(3),
        deformation_gradient_new=gradient,
        time=1.0,
        time_increment=1.0,
        properties=(),
        state_old=state,
        state_schema=material.state_schema,
    )


def test_material_driver_dispatches_finite_strain_j2_by_cell_region():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1)
    material_map = _regional_map(domain)
    schema = material_map.require_common_state_schema()
    assert (
        material_map.require_common_tangent_convention().stress_measure
        == "first_piola"
    )
    state = constitutive.MaterialQuadratureState.create(domain, schema, degree=2)
    lateral = 1.0 / np.sqrt(1.02)
    gradient = np.diag((1.02, lateral, lateral))
    committed = state.committed_state_vectors().copy()

    response = constitutive.update_material_points(
        material_map,
        state,
        deformation_gradient_old=np.eye(3),
        deformation_gradient_new=gradient,
        time=1.0,
        time_increment=1.0,
    )

    points_per_cell = len(state.reference_field.points)
    for index in range(response.point_count):
        selected = material_map.material_for_point(
            index,
            points_per_cell=points_per_cell,
        )
        expected = selected.update(_point(selected, gradient, committed[index]))
        np.testing.assert_allclose(
            response.cauchy_stress[index],
            expected.cauchy_stress,
            rtol=2.0e-12,
            atol=2.0e-10,
        )
        np.testing.assert_allclose(
            response.state_new[index],
            expected.state_new,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
    np.testing.assert_array_equal(state.committed_state_vectors(), committed)
    assert set(material_map.cell_regions) == {1, 2}

    state.rollback()
    with pytest.raises(RuntimeError, match="own their parameters"):
        constitutive.update_material_points(
            material_map,
            state,
            deformation_gradient_old=np.eye(3),
            deformation_gradient_new=gradient,
            time=1.0,
            time_increment=1.0,
            properties=(1.0,),
        )
    np.testing.assert_array_equal(state.committed_state_vectors(), committed)
    np.testing.assert_array_equal(state.trial_state_vectors(), committed)


def test_regional_material_protocol_mismatch_fails_before_integration():
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    first, second = _materials()
    cell_count = domain.topology.index_map(domain.topology.dim).size_local
    regions = np.ones(cell_count, dtype=np.int64)

    incompatible_schema = constitutive.MaterialStateSchema(
        "different_finite_strain_state",
        (constitutive.MaterialStateVariable("damage"),),
    )
    schema_map = constitutive.QuadratureMaterialMap(
        domain,
        {
            1: first,
            2: SimpleNamespace(
                state_schema=incompatible_schema,
                tangent_convention=second.tangent_convention,
            ),
        },
        np.where(np.arange(cell_count) % 2 == 0, 1, 2),
    )
    with pytest.raises(ValueError, match="one complete state schema"):
        schema_map.require_common_state_schema()

    tangent_map = constitutive.QuadratureMaterialMap(
        domain,
        {
            1: first,
            2: SimpleNamespace(
                state_schema=second.state_schema,
                tangent_convention=constitutive.MaterialTangentConvention.abaqus_umat(),
            ),
        },
        np.where(np.arange(cell_count) % 2 == 0, 1, 2),
    )
    with pytest.raises(ValueError, match="one complete tangent convention"):
        tangent_map.require_common_tangent_convention()

    energy_map = constitutive.QuadratureMaterialMap(
        domain,
        {
            1: first,
            2: SimpleNamespace(
                state_schema=second.state_schema,
                tangent_convention=second.tangent_convention,
                stored_energy_component_names=("ELENER",),
            ),
        },
        np.where(np.arange(cell_count) % 2 == 0, 1, 2),
    )
    with pytest.raises(ValueError, match="stored-energy component contract"):
        energy_map.require_common_stored_energy_component_names()


def test_public_two_phase_periodic_j2_uses_standard_output_lifecycle(tmp_path):
    fixture = periodic_unit_cube(MPI.COMM_SELF, stretch=1.01)
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="two_phase_finite_strain_j2_periodic_cell",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    regions = mesh.partition_cells(
        fixture.domain,
        left=lambda x: x[0] <= 0.5,
        right=lambda x: x[0] > 0.5,
    )
    left, right = _materials()
    model.material(left, region=regions.left)
    model.material(right, region=regions.right)
    periodicity = model.constraint(fixture.constraint(displacement))
    output = results.output_plan(
        tmp_path,
        field=results.field_output(
            "U",
            "P",
            "S",
            "MISES",
            "SENER",
            "ELENER",
            "HARDENER",
            "PEEQ",
            "PDENER",
            every=2,
            configuration="reference",
            backend="xdmf",
        ),
        requests=(results.periodic_cell_history(periodicity),),
        presentation=None,
        basename="two_phase_periodic_j2",
    )
    capability = step_capability(
        model,
        target=displacement,
        options={"constraints": periodicity},
    )
    assert capability["supported"]
    assert capability["provider"]["name"] == "finite_strain_j2_affine_static"

    step = model.step(
        target=displacement,
        constraints=periodicity,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=25,
            line_search="backtracking",
        ),
        output=output,
        progress=False,
    )
    result = step.solve_result()

    assert isinstance(step.material, constitutive.QuadratureMaterialMap)
    assert step.execution_context.material is step.material
    assert step.last_solve_info.completed_step
    assert result.metadata["output_plan"]["status"] == "completed"
    assert result.artifacts["field_history"].is_file()
    assert {
        "F",
        "P",
        "S",
        "MISES",
        "SENER",
        "ELENER",
        "HARDENER",
        "FP",
        "PEEQ",
        "PDENER",
    } <= set(result.fields)
    np.testing.assert_allclose(
        step.response.strain_energy_density.values,
        step.response.stored_energy_density_components["ELENER"].values
        + step.response.stored_energy_density_components["HARDENER"].values,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    xdmf_text = result.artifacts["field_history"].read_text(encoding="utf-8")
    assert "ELENER" in xdmf_text
    assert "HARDENER" in xdmf_text
    assert "homogenized_elastic_energy_density" in result.histories
    assert "homogenized_hardening_energy_density" in result.histories
    assert "homogenized_plastic_dissipation_density" in result.histories
    assert np.all(
        np.diff(
            result.histories[
                "homogenized_plastic_dissipation_density"
            ].values
        )
        >= -2.0e-10
    )
    np.testing.assert_allclose(
        result.histories["homogenized_strain_energy_density"].values,
        result.histories["homogenized_elastic_energy_density"].values
        + result.histories["homogenized_hardening_energy_density"].values,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert result.quantity("homogenized_strain_energy_density") == pytest.approx(
        result.quantity("homogenized_elastic_energy_density")
        + result.quantity("homogenized_hardening_energy_density")
    )
    assert result.quantity("maximum_hill_mandel_relative_error") < 1.0e-7

    points_per_cell = len(step.response.state.reference_field.points)
    point_regions = np.repeat(step.material.cell_regions, points_per_cell)
    peeq = step.response.state.committed_state_vectors()[:, -2]
    pdener = step.response.state.committed_state_vectors()[:, -1]
    stress = step.response.cauchy_stress.values
    mises = np.sqrt(
        1.5
        * np.sum(
            (stress - np.trace(stress, axis1=1, axis2=2)[:, None, None] / 3.0 * np.eye(3))
            ** 2,
            axis=(1, 2),
        )
    )
    assert not np.isclose(np.mean(peeq[point_regions == 1]), np.mean(peeq[point_regions == 2]))
    assert not np.isclose(
        np.mean(pdener[point_regions == 1]),
        np.mean(pdener[point_regions == 2]),
    )
    assert not np.isclose(np.mean(mises[point_regions == 1]), np.mean(mises[point_regions == 2]))
