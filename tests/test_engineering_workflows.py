from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import ufl
from mpi4py import MPI

from agentfem import boundary_models, constitutive, fields, loads, mesh, models, results, steps, studies
from agentfem.step_providers import step_capability


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
    assert tuple(result.fields) == ("U", "P")


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
