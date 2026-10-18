"""Exact narrow-strip and current-trial checks for bounded planar contact."""

from __future__ import annotations

import json

import basix.ufl
from dolfinx import fem, mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np
import pytest

from agentfem import (
    boundary_models, constitutive, fields, mesh, models, results, solvers, steps,
    studies,
)
from agentfem.boundary_models.contact_active_partition import (
    affine_active_triangle_quadrature,
)
from agentfem.boundary_models.dolfinx_implicit_facet_contact import (
    dolfinx_implicit_rigid_contact,
)
from agentfem.step_providers import step_capability


@pytest.mark.parametrize("gaps", ((-0.05, 0.95, -0.05), (-0.2, 0.8, 0.8)))
def test_affine_partition_positive_area_and_potential_derivatives(gaps):
    values = np.asarray(gaps)
    shapes, weights = affine_active_triangle_quadrature(values)
    assert np.all(weights > 0.0)
    assert np.all(shapes @ values <= 1e-14)
    def potential(q):
        current_shapes, current_weights = affine_active_triangle_quadrature(q)
        return 0.5 * np.sum(current_weights * (current_shapes @ q) ** 2)

    def force(q):
        current_shapes, current_weights = affine_active_triangle_quadrature(q)
        return current_shapes.T @ (current_weights * (current_shapes @ q))

    residual = shapes.T @ (weights * (shapes @ values))
    tangent = shapes.T @ (weights[:, None] * shapes)
    direction = np.asarray((0.2, -0.3, 0.7))
    h = 1e-7
    assert (potential(values + h * direction) - potential(values - h * direction)) / (2 * h) == pytest.approx(residual @ direction)
    np.testing.assert_allclose(
        tangent @ direction,
        (force(values + h * direction) - force(values - h * direction)) / (2 * h),
        rtol=1e-8, atol=1e-10,
    )


@pytest.mark.parametrize("width", (1e-4, 0.01, 0.05, 0.2))
def test_affine_partition_matches_independent_narrow_strip_formula(width):
    # g=x-width on the unit reference triangle; whole-face degree-two
    # quadrature samples no active point for the narrower cases.
    gaps = np.asarray((-width, 1 - width, -width))
    shapes, weights = affine_active_triangle_quadrature(gaps)
    energy = 0.5 * np.sum(weights * (shapes @ gaps) ** 2)
    assert np.sum(weights) == pytest.approx(width - width**2 / 2, rel=1e-11, abs=0.0)
    assert energy == pytest.approx(
        width**3 / 6 - width**4 / 24, rel=1e-10, abs=0.0,
    )


def _narrow_contact(*, integration="gauss", moving=False):
    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, 1, 1, 1,
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    space = fem.functionspace(
        domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,)),
    )
    displacement = fem.Function(space)
    displacement.x.array.reshape((-1, 3))[:, 0] = space.tabulate_dof_coordinates()[:, 1]
    facets = dolfinx_mesh.locate_entities_boundary(
        domain, 2, lambda x: np.isclose(x[0], 0.0),
    )
    tags = dolfinx_mesh.meshtags(
        domain, 2, facets, np.full(len(facets), 7, dtype=np.int32),
    )
    slave = mesh.tagged_boundary_region(domain, tags, tag=7, name="slave")
    motion = (
        boundary_models.prescribed_rigid_motion_schedule(
            boundary_models.prescribed_rigid_motion(translation=(0.01, 0, 0)),
            start_time=0, end_time=1,
        ) if moving else None
    )
    surface = boundary_models.rigid_plane(
        point=(0.04 if moving else 0.05, 0, 0), normal=(1, 0, 0),
    )
    pair = boundary_models.rigid_contact_pair(
        slave,
        boundary_models.rigid_body(surface, motion_schedule=motion),
        penalty=17.0,
    )
    operator = dolfinx_implicit_rigid_contact(
        pair, displacement, contact_integration=integration,
    )
    if moving:
        operator.update_motion(1.0)
    return displacement, operator


@pytest.mark.parametrize("moving", (False, True))
def test_trial_planar_partition_recovers_strip_missed_by_whole_face_rule(moving):
    displacement, clipped = _narrow_contact(
        integration="planar_active_partition", moving=moving,
    )
    _, ordinary = _narrow_contact(moving=moving)
    value = clipped.evaluate()
    assert ordinary.evaluate().potential_energy == 0.0
    assert value.potential_energy == pytest.approx(17.0 * 0.05**3 / 6, rel=1e-12)
    assert value.assembly.contact_force_on_structure[0] == pytest.approx(
        17.0 * 0.05**2 / 2, rel=1e-12,
    )
    assert clipped.summary()["contact_integration"] == "planar_active_partition"
    base = displacement.x.array.copy()
    direction = np.random.default_rng(16).normal(size=base.shape)
    h = 1e-7
    displacement.x.array[:] = base + h * direction
    plus = clipped.evaluate()
    displacement.x.array[:] = base - h * direction
    minus = clipped.evaluate()
    displacement.x.array[:] = base
    np.testing.assert_allclose(
        value.tangent_action(direction.reshape(-1, 3)),
        (plus.nodal_structural_residual - minus.nodal_structural_residual) / (2 * h),
        rtol=1e-7, atol=1e-8,
    )


def test_trial_partition_rejects_nonplanar_tool():
    displacement, ordinary = _narrow_contact()
    sphere = boundary_models.rigid_sphere(center=(0, 0, 0), radius=1.0)
    pair = boundary_models.rigid_contact_pair(
        ordinary.pair.slave_boundary,
        boundary_models.rigid_body(sphere), penalty=17.0,
    )
    with pytest.raises(NotImplementedError, match="AFM-CONTACT-PARTITION"):
        dolfinx_implicit_rigid_contact(
            pair, displacement, contact_integration="planar_active_partition",
        )


def _ordinary_step(integration="planar_active_partition", *, tilted=False):
    domain = dolfinx_mesh.create_unit_cube(
        MPI.COMM_SELF, 1, 1, 1,
        cell_type=dolfinx_mesh.CellType.tetrahedron,
    )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right")
    model.clamp(displacement, on=left)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000, poisson=0.0, yield_stress=100, hardening_modulus=100,
    ))
    motion = boundary_models.prescribed_rigid_motion_schedule(
        boundary_models.prescribed_rigid_motion(translation=(-0.02, 0, 0)),
        start_time=0, end_time=1,
    )
    pair = boundary_models.rigid_contact_pair(
        right,
        boundary_models.rigid_body(
            boundary_models.rigid_plane(
                point=(1.01, 0, 0),
                normal=tuple(
                    np.asarray((-1, 0.8 if tilted else 0, 0), dtype=float)
                    / np.sqrt(1.0 + (0.8 if tilted else 0.0) ** 2)
                ),
            ),
            motion_schedule=motion,
        ), penalty=200,
    )
    model.add_boundary_model(pair)
    options = {"material": material, "contact_integration": integration}
    assert step_capability(model, target=displacement, options=options)["supported"]
    step = model.step(
        target=displacement, **options,
        incrementation=steps.fixed(2),
        solver_options=solvers.newton(
            relative_tolerance=1e-8, absolute_tolerance=1e-9,
            maximum_iterations=20,
        ), progress=False,
    )
    return model, displacement, material, step


def test_ordinary_j2_step_admits_bounded_partition_and_rejects_other_tools():
    model, displacement, material, step = _ordinary_step()
    result = step.solve_result()
    assert result.status == "completed"
    assert step.contact_provider.summary()["contact_integration"] == "planar_active_partition"
    assert step.contact_provider.evaluate().potential_energy > 0
    rejected = step_capability(
        model, target=displacement,
        options={"material": material, "contact_integration": "planar_active_partition",
                 "contact_quadrature_degree": 5},
    )
    assert not rejected["supported"]
    assert any(
        "AFM-CONTACT-PARTITION" in item["message"]
        for item in rejected["rejection_reasons"]
    )


def test_partition_checkpoint_identifies_rule_and_restores_accepted_state(tmp_path):
    _, _, _, selected = _ordinary_step()
    selected.solve()
    saved = selected.save_checkpoint(tmp_path / "partitioned")
    _, _, _, same = _ordinary_step()
    same.load_checkpoint(saved)
    np.testing.assert_allclose(same.solution.x.array, selected.solution.x.array)
    assert same.accepted_load_factor == pytest.approx(selected.accepted_load_factor)
    _, _, _, ordinary = _ordinary_step("gauss")
    with pytest.raises(ValueError, match="identity"):
        ordinary.load_checkpoint(saved)
    assert ordinary.accepted_load_factor == 0.0
    assert np.max(np.abs(ordinary.solution.x.array)) == 0.0


def test_partitioned_narrow_contact_enters_ordinary_j2_equilibrium():
    _, _, _, clipped = _ordinary_step(tilted=True)
    _, _, _, ordinary = _ordinary_step("gauss", tilted=True)
    assert clipped.solve_result().status == "completed"
    assert ordinary.solve_result().status == "completed"
    assert clipped.contact_provider.evaluate().active_points > 0
    assert ordinary.contact_provider.evaluate().active_points == 0
    assert clipped.contact_provider.evaluate().potential_energy > 0
    assert ordinary.contact_provider.evaluate().potential_energy == 0.0
    assert np.max(np.abs(clipped.solution.x.array - ordinary.solution.x.array)) > 1e-10


def test_partitioned_output_marks_point_ids_as_snapshot_local(tmp_path):
    _, _, _, step = _ordinary_step(tilted=True)
    assert step.solve_result().status == "completed"
    paths = results.export_accepted_contact_points(step, tmp_path / "contact")
    metadata = json.loads(paths["contact_point_metadata"].read_text())
    assert metadata["pairs"][0]["point_identity_scope"] == "accepted_snapshot_only"
    with np.load(paths["contact_points"]) as data:
        assert len(data["point_ids"]) == metadata["pairs"][0]["point_count"]


def test_partitioned_contact_interrupted_restore_matches_continuous_path(tmp_path):
    _, _, _, uninterrupted = _ordinary_step(tilted=True)
    uninterrupted.solve()
    _, _, _, partial = _ordinary_step(tilted=True)
    partial.solve(until=0.5)
    saved = partial.save_checkpoint(tmp_path / "partial_partitioned")
    _, _, _, restored = _ordinary_step(tilted=True)
    restored.load_checkpoint(saved)
    restored.solve()
    np.testing.assert_allclose(
        restored.solution.x.array, uninterrupted.solution.x.array,
        rtol=1e-12, atol=1e-12,
    )
    assert restored.energy_history[-1].tool_work == pytest.approx(
        uninterrupted.energy_history[-1].tool_work,
    )
