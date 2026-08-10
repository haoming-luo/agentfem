from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import ufl
from mpi4py import MPI

from agentfem import (
    boundary_models,
    constitutive,
    constraints,
    coordinates,
    fields,
    loads,
    mesh,
    models,
    results,
    steps,
    studies,
)
from agentfem.step_providers import step_capability
from agentfem.mesh import abaqus


def test_mixed_hybrid_unknown_has_one_constant_pressure_value_per_cell():
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1),
        comm=MPI.COMM_SELF, cell_type="tetrahedron",
    )
    unknown = fields.displacement_pressure(domain)

    assert unknown.displacement_degree == 2
    assert unknown.pressure_degree == 0
    assert unknown.summary()["pressure_unknowns_per_cell"] == 1
    assert unknown.space.num_sub_spaces == 2


def test_mixed_material_is_selected_by_the_unified_step_provider():
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1),
        comm=MPI.COMM_SELF, cell_type="tetrahedron",
    )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain,
    )
    unknown = model.field(fields.displacement_pressure(domain))
    material = model.material(
        constitutive.mixed_neo_hookean(young=1.0e6, poisson=0.499),
    )

    capability = step_capability(
        model, target=unknown, options={"material": material},
    )

    assert capability["supported"]
    assert capability["provider"]["name"] == "mixed_neo_hookean_constant_pressure"


def test_mixed_neo_hookean_direct_moduli_match_quadratic_volumetric_energy():
    material = constitutive.mixed_neo_hookean(
        shear_modulus=2.5,
        bulk_modulus=2.5e4,
    )
    F = np.diag((1.20, 0.93, 0.91))
    J = float(np.linalg.det(F))
    invariant = float(np.trace(F.T @ F))
    expected = (
        0.5 * 2.5 * (J ** (-2.0 / 3.0) * invariant - 3.0)
        + 0.5 * 2.5e4 * (J - 1.0) ** 2
    )

    assert material.mu == pytest.approx(2.5)
    assert material.bulk_modulus == pytest.approx(2.5e4)
    assert material.as_dict()["C10"] == pytest.approx(1.25)
    assert material.as_dict()["D1"] == pytest.approx(8.0e-5)
    assert material.as_dict()["abaqus_C10"] == pytest.approx(1.25)
    assert material.as_dict()["abaqus_D1"] == pytest.approx(8.0e-5)
    assert constitutive.mixed_condensed_energy_value(F, material) == pytest.approx(
        expected
    )


def test_mixed_neo_hookean_rejects_ambiguous_parameterization():
    with pytest.raises(ValueError, match="exactly one complete pair"):
        constitutive.mixed_neo_hookean(
            young=1000.0,
            poisson=0.49,
            shear_modulus=1.0,
            bulk_modulus=1.0e4,
        )
    with pytest.raises(ValueError, match="declared together"):
        constitutive.mixed_neo_hookean(shear_modulus=1.0)


def test_mixed_hybrid_zero_state_solves_with_subspace_constraints():
    domain = mesh.cuboid((0, 0, 0), (1, 1, 1), (1, 1, 1), comm=MPI.COMM_SELF, cell_type="tetrahedron")
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
    unknown = model.field(fields.displacement_pressure(domain))
    material = model.material(constitutive.mixed_neo_hookean(young=1.0e6, poisson=0.499))
    exterior = mesh.boundary(domain, lambda x: np.full(x.shape[1], True), name="all")
    model.fix(unknown.displacement, on=exterior)

    problem = model.step(target=unknown, material=material, increments=1, progress=False)
    result = problem.solve_result()

    assert problem.last_solve_info.converged
    assert np.max(np.abs(unknown.value.x.array)) == pytest.approx(0.0)
    assert tuple(result.fields) == ("U", "PRESSURE")


def test_mixed_hybrid_affine_periodic_reduction_keeps_pressure_independent(tmp_path):
    domain = mesh.cuboid(
        (0, 0, 0), (1, 1, 1), (1, 1, 1),
        comm=MPI.COMM_SELF, cell_type="tetrahedron",
    )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain,
    )
    unknown = model.field(fields.displacement_pressure(domain))
    material = model.material(
        constitutive.mixed_neo_hookean(young=1.0e6, poisson=0.499),
    )
    nodes = abaqus.AbaqusNodeTable(
        labels=np.asarray((1, 2, 3, 4)),
        coordinates=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
             (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        ),
    )
    periodicity = constraints.abaqus_periodic_cell(
        unknown,
        nodes=nodes,
        equations=abaqus.AbaqusEquationSet(()),
        deformation_gradient=np.eye(3),
        anchor_node=1,
        reference_nodes=(2, 3, 4),
    )
    output = results.output_plan(
        tmp_path,
        field=results.field_output(
            "U", "P", "PRESSURE", "S", "J",
            configuration="reference",
            backend="xdmf",
        ),
        requests=(results.periodic_cell_history(periodicity),),
        presentation=None,
        basename="mixed_periodic",
    )

    reduction = periodicity.reduction()
    _, pressure_maps = unknown.space.sub(1).collapse()
    pressure_parent_dofs = set(
        np.asarray(pressure_maps[0], dtype=int).tolist()
    )
    assert pressure_parent_dofs <= set(
        reduction.independent_full_dofs.astype(int).tolist()
    )

    problem = model.step(
        target=unknown,
        material=material,
        constraints=periodicity,
        increments=1,
        output=output,
        progress=False,
    )
    result = problem.solve_result()
    result = output.finalize(
        model=model,
        step=problem,
        result=result,
        target=unknown.collapsed_displacement(name="U"),
        material=material,
    )

    assert problem.last_solve_info.converged
    assert all(
        item.checks["minimum_quadrature_J"] > 0.0
        for item in problem.last_solve_info.increments
    )
    assert set(result.fields) == {"U", "PRESSURE", "P", "S", "J"}
    assert np.max(np.abs(unknown.value.x.array)) == pytest.approx(0.0)
    assert (tmp_path / "mixed_periodic.xdmf").exists()
    assert "homogenized_first_piola_stress" in result.histories


def test_abaqus_reference_controls_can_leave_one_macro_component_free():
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1),
        comm=MPI.COMM_SELF, cell_type="tetrahedron",
    )
    displacement = fields.displacement(domain)
    nodes = abaqus.AbaqusNodeTable(
        labels=np.asarray((1, 2, 3, 4)),
        coordinates=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
    )
    periodicity = constraints.abaqus_periodic_cell(
        displacement,
        nodes=nodes,
        equations=abaqus.AbaqusEquationSet(()),
        anchor_node=1,
        reference_nodes=(2, 3, 4),
        control_displacements=(
            (0.2, 0.0, 0.0),
            (0.0, None, 0.0),
            (0.0, 0.0, None),
        ),
    )

    reduction = periodicity.reduction()
    assert periodicity.has_free_macro_dofs
    assert len(periodicity.prescribed_control_dofs) == 10
    assert reduction.reduced_size == displacement.value.x.array.size - 10
    assert periodicity.prescribed_values_at(0.5)[(2, 0)] == pytest.approx(0.1)
    assert (3, 1) not in periodicity.prescribed_control_dofs

    displacement.value.interpolate(
        lambda x: np.vstack((0.2 * x[0], -0.15 * x[1], np.zeros_like(x[2])))
    )
    measured = periodicity.measured_deformation_gradient(displacement)
    np.testing.assert_allclose(
        measured,
        np.diag((1.2, 0.85, 1.0)),
        rtol=0.0,
        atol=1.0e-14,
    )
    summary = periodicity.summary()
    assert summary["free_macro_dofs"] == [
        {"node": 3, "component": 2},
        {"node": 4, "component": 3},
    ]
    diagnostics = results.finite_strain_diagnostics(
        displacement,
        constraint=periodicity,
    )
    np.testing.assert_array_equal(
        diagnostics["macro_control_prescribed_mask"],
        ((1.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
    )
    assert all(
        np.all(np.isfinite(np.asarray(value, dtype=float)))
        for value in diagnostics.values()
    )


def test_distributing_coupling_preserves_force_and_moment_resultants():
    domain = mesh.rectangle(
        (0.0, 0.0), (2.0, 1.0), (2, 2),
        comm=MPI.COMM_SELF, cell_type="triangle",
    )
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 2.0), name="right")
    load = loads.distributing_coupling(
        (3.0, 4.0), moment=2.0, reference_point=(2.0, 0.5), on=right,
    )

    integrated = results.boundary_resultant(load.traction, on=right)
    check = results.free_body_resultant(
        boundary_tractions=((load.traction, right),), about=(2.0, 0.5),
    )

    np.testing.assert_allclose(integrated, [3.0, 4.0], atol=1.0e-11)
    assert check.moment == pytest.approx(2.0, abs=1.0e-11)


def test_remote_force_consumes_local_components_and_named_reference_point():
    domain = mesh.rectangle(
        (0.0, 0.0), (2.0, 1.0), (2, 2),
        comm=MPI.COMM_SELF, cell_type="triangle",
    )
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 2.0), name="right")
    local = coordinates.cartesian(x=(0.0, 1.0), y=(-1.0, 0.0), name="fixture")
    point = coordinates.reference_point((2.0, 0.5), name="RP-1")

    load = loads.remote_force(
        (3.0, 4.0), moment=2.0, reference_point=point,
        system=local, on=right,
    )
    integrated = results.boundary_resultant(load.traction, on=right)

    np.testing.assert_allclose(integrated, [-4.0, 3.0], atol=1.0e-11)
    assert load.reference_point == pytest.approx((2.0, 0.5))
    assert load.summary()["reference_name"] == "RP-1"
    assert load.summary()["coordinate_system"] == "fixture"


def test_remote_displacement_is_a_ramped_rigid_boundary_motion():
    domain = mesh.rectangle(
        (0.0, 0.0), (2.0, 1.0), (2, 2),
        comm=MPI.COMM_SELF, cell_type="triangle",
    )
    displacement = fields.displacement(domain)
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 2.0), name="right")
    point = coordinates.reference_point((2.0, 0.5), name="RP-1")
    constraint = constraints.remote_displacement(
        displacement,
        reference_point=point,
        on=right,
        translation=(1.0, 2.0),
        rotation=0.25,
    )
    reference = constraint.reference_values.copy()

    path = constraints.prescribed_value_path((constraint,))
    path.update(0.4)

    np.testing.assert_allclose(constraint.value.x.array, 0.4 * reference)
    assert path.summary()["field_values"] == 1
    assert constraint.summary()["reference_point"] == "RP-1"


def test_elastic_foundation_is_a_mechanical_boundary_matrix():
    domain = mesh.rectangle(
        (0.0, 0.0), (1.0, 1.0), (1, 1),
        comm=MPI.COMM_SELF, cell_type="triangle",
    )
    bottom = mesh.boundary(domain, lambda x: np.isclose(x[1], 0.0), name="bottom")
    displacement = fields.displacement(domain)
    foundation = boundary_models.elastic_foundation(
        on=bottom, stiffness=10.0, mode="normal",
    )

    operator = foundation.operator(displacement)

    assert operator.role == "matrix"
    assert operator.family == "mechanical_boundary"


def test_centrifugal_and_hydrostatic_loads_have_physical_resultants():
    rotating = mesh.rectangle(
        (0.0, 0.0), (1.0, 1.0), (2, 2),
        comm=MPI.COMM_SELF, cell_type="triangle",
    )
    centrifugal = loads.centrifugal(
        3.0, density=2.0, center=(0.0, 0.0), domain=rotating,
    )
    resultant = results.integral(
        centrifugal.value, measure=ufl.dx, comm=MPI.COMM_SELF,
    )
    np.testing.assert_allclose(resultant, [9.0, 9.0], atol=1.0e-12)

    tank = mesh.rectangle(
        (0.0, -1.0), (1.0, 0.0), (2, 2),
        comm=MPI.COMM_SELF, cell_type="triangle",
    )
    wall = mesh.boundary(tank, lambda x: np.isclose(x[0], 0.0), name="wall")
    hydrostatic = loads.hydrostatic_pressure(
        density=2.0, gravity=(0.0, -10.0), reference_point=(0.0, 0.0), on=wall,
    )
    np.testing.assert_allclose(
        results.boundary_resultant(hydrostatic.traction, on=wall),
        [10.0, 0.0], atol=1.0e-12,
    )


def test_engineering_steps_inherit_and_deactivate_named_assets():
    first = steps.engineering_step("preload")
    second = steps.engineering_step("service", previous=first)
    gravity = SimpleNamespace(name="gravity")
    pressure = SimpleNamespace(name="pressure")
    first.activate_load(gravity)
    second.activate_load(pressure)
    second.deactivate_load("gravity")

    assert first.resolve_loads((pressure,)) == (gravity,)
    assert second.resolve_loads(()) == (pressure,)
    assert second.summary()["previous"] == "preload"


def test_section_resultant_returns_force_and_moment():
    domain = mesh.rectangle(
        (0.0, 0.0), (2.0, 1.0), (2, 2),
        comm=MPI.COMM_SELF, cell_type="triangle",
    )
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 2.0), name="right")
    stress = ufl.as_matrix(((2.0, 0.0), (0.0, 1.0)))

    resultant = results.section_resultant(stress, on=right, about=(2.0, 0.5))

    np.testing.assert_allclose(resultant.force, [2.0, 0.0], atol=1.0e-12)
    assert resultant.moment == pytest.approx(0.0, abs=1.0e-12)
