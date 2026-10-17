"""Private planar friction Operator checks; public Step admission is separate."""

from __future__ import annotations

import basix.ufl
import json
from dolfinx import fem, mesh
from mpi4py import MPI
import numpy as np
import pytest

from agentfem import boundary_models, constitutive, fields, models, solvers, steps, studies
from agentfem import mesh as agent_mesh
from agentfem.boundary_models.dolfinx_implicit_facet_contact import (
    dolfinx_implicit_rigid_contact,
)


def _operator():
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    space = fem.functionspace(
        domain,
        basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,)),
    )
    displacement = fem.Function(space)
    facets = mesh.locate_entities_boundary(
        domain, 2, lambda x: np.isclose(x[0], 0.0),
    )
    tags = mesh.meshtags(
        domain, 2, facets, np.full(len(facets), 7, dtype=np.int32),
    )
    slave = agent_mesh.tagged_boundary_region(domain, tags, tag=7, name="slave")
    pair = boundary_models.rigid_contact_pair(
        slave,
        boundary_models.rigid_body(boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
        )),
        penalty=200.0, friction_coefficient=0.4,
        tangential_penalty=100.0,
    )
    return displacement, dolfinx_implicit_rigid_contact(pair, displacement)


@pytest.mark.parametrize("tangential", [0.02, 0.10])
def test_fixed_plane_friction_backend_stick_slip_tangent(tangential):
    displacement, operator = _operator()
    accepted = displacement.x.array.copy()
    displacement.x.array.reshape(-1, 3)[:, 0] = 0.10
    displacement.x.array.reshape(-1, 3)[:, 1] = tangential
    base = displacement.x.array.copy()
    contribution = operator.evaluate()
    assert contribution.active_points > 0
    assert contribution.friction_assembly is not None
    assert contribution.friction_tangent is not None
    assert operator.friction_state.accepted is None
    if tangential < 0.05:
        assert contribution.friction_tangent.sticking_points > 0
    else:
        assert contribution.friction_tangent.sliding_points > 0
    direction = np.random.default_rng(102).normal(size=base.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-7
    displacement.x.array[:] = base + epsilon * direction
    plus = operator.evaluate().nodal_structural_residual
    displacement.x.array[:] = base - epsilon * direction
    minus = operator.evaluate().nodal_structural_residual
    displacement.x.array[:] = base
    np.testing.assert_allclose(
        contribution.tangent_action(direction.reshape(-1, 3)),
        (plus - minus) / (2.0 * epsilon),
        rtol=2.0e-5, atol=2.0e-6,
    )
    assert np.allclose(displacement.x.array, base)
    assert operator.friction_state.accepted is None
    assert not np.array_equal(accepted, base)


def test_fixed_plane_friction_accept_restore_and_corrupt_state_rejection():
    displacement, operator = _operator()
    displacement.x.array.reshape(-1, 3)[:, 0] = 0.10
    displacement.x.array.reshape(-1, 3)[:, 1] = 0.10
    initial = operator.snapshot_accepted_state()
    first = operator.evaluate()
    assert first.friction_assembly.dissipation_increment > 0.0
    assert operator.friction_state.accepted is None
    operator.commit_accepted()
    accepted = operator.snapshot_accepted_state()
    assert accepted != initial
    assert json.loads(json.dumps(accepted)) == accepted
    assert operator.friction_state.accepted is not None
    assert operator.evaluate().friction_assembly.dissipation_increment == pytest.approx(0.0)
    displacement.x.array.reshape(-1, 3)[:, 1] += 0.04
    assert operator.evaluate().friction_assembly.dissipation_increment > 0.0
    operator.restore_accepted_state(accepted)
    assert operator.snapshot_accepted_state() == accepted
    damaged = json.loads(json.dumps(accepted))
    damaged["point_ids"][0] = -1
    with pytest.raises(ValueError, match="point IDs"):
        operator.restore_accepted_state(damaged)
    assert operator.snapshot_accepted_state() == accepted


def _frictional_j2_step(*, tangential_load: float, linear_solver=None):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
    displacement = model.field(fields.displacement(domain))
    right = agent_mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="clamp")
    left = agent_mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="slave")
    model.clamp(displacement, on=right)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0, yield_stress=1.0e9, hardening_modulus=100.0,
    ))
    pair = boundary_models.rigid_contact_pair(
        left,
        boundary_models.rigid_body(boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
        )),
        penalty=500.0, friction_coefficient=0.4, tangential_penalty=100.0,
        name="fixed_frictional_plane",
    )
    model.add_boundary_model(pair)
    model.traction((100.0, tangential_load, 0.0), on=left)
    step = model.step(
        target=displacement, material=material, incrementation=steps.fixed(2),
        solver_options=solvers.newton(
            relative_tolerance=1e-8, absolute_tolerance=1e-9,
            maximum_iterations=30, line_search="backtracking",
            linear_solver=(solvers.direct_solver() if linear_solver is None else linear_solver),
        ),
        progress=False,
    )
    return step


def test_public_friction_rejects_symmetry_only_linear_solver():
    with pytest.raises(ValueError, match="AFM-IMPLICIT-FRICTION-LINEAR"):
        _frictional_j2_step(
            tangential_load=15.0,
            linear_solver=solvers.LinearSolverOptions(ksp_type="cg", pc_type="jacobi"),
        )


@pytest.mark.parametrize(
    ("shear_displacement", "expected_shear_force", "expected_dissipation"),
    [(0.02, -2.0, 0.0), (0.20, -10.0, 1.0)],
)
def test_prescribed_friction_patch_matches_independent_force_and_energy(
    shear_displacement, expected_shear_force, expected_dissipation,
):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1)
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
    displacement = model.field(fields.displacement(domain))
    right = agent_mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="clamp")
    left = agent_mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="slave")
    model.clamp(displacement, on=right)
    model.fix(displacement, on=left, component=0, value=0.10)
    model.fix(displacement, on=left, component=1, value=shear_displacement)
    model.fix(displacement, on=left, component=2, value=0.0)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000.0, poisson=0.0, yield_stress=1.0e9, hardening_modulus=100.0,
    ))
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        left,
        boundary_models.rigid_body(boundary_models.rigid_plane(
            point=(0.05, 0.0, 0.0), normal=(-1.0, 0.0, 0.0),
        )),
        penalty=500.0, friction_coefficient=0.4,
        tangential_penalty=100.0, name="patch_plane",
    ))
    step = model.step(
        target=displacement, material=material,
        incrementation=steps.fixed(1), progress=False,
    )
    result = step.solve_result()
    force = np.asarray(result.metadata["constraint_duals"][0]["force"])
    np.testing.assert_allclose(
        force, (-25.0, expected_shear_force, 0.0),
        rtol=1e-12, atol=1e-10,
    )
    strong_reaction = np.sum(
        result.fields["RF"].field.x.array.reshape((-1, 3)), axis=0
    )
    np.testing.assert_allclose(
        strong_reaction + force, 0.0, rtol=0.0, atol=1e-8,
    )
    assert step.energy_history[-1].friction_dissipation == pytest.approx(
        expected_dissipation, rel=1e-12, abs=1e-10,
    )
    assert step.energy_history[-1].contact_potential == pytest.approx(
        0.625 + 0.5 * expected_shear_force ** 2 / 100.0,
        rel=1e-12, abs=1e-10,
    )


@pytest.mark.parametrize("tangential_load", [1.0, 100.0])
def test_public_fixed_plane_j2_friction_accepts_and_restarts(tmp_path, tangential_load):
    reference = _frictional_j2_step(tangential_load=tangential_load)
    result = reference.solve_result()
    assert len(reference.energy_history) == 2
    assert reference.contact_provider.current_point_diagnostics()["active_point_count"] > 0
    assert reference.contact_provider.friction_state.accepted is not None
    assert reference.energy_history[-1].friction_dissipation >= 0.0
    if tangential_load == 100.0:
        assert reference.energy_history[-1].friction_dissipation > 0.0
    assert reference.energy_history[-1].internal_energy == pytest.approx(
        reference.energy_history[-1].stored_energy
        + reference.energy_history[-1].plastic_dissipation
        + reference.energy_history[-1].contact_potential
        + reference.energy_history[-1].friction_dissipation
        + reference.energy_history[-1].friction_separation_release
    )

    partial = _frictional_j2_step(tangential_load=tangential_load)
    partial.solve(until=0.5)
    path = partial.save_checkpoint(tmp_path / "friction.npz")
    restarted = _frictional_j2_step(tangential_load=tangential_load)
    restarted.load_checkpoint(path)
    restarted.solve()
    np.testing.assert_allclose(
        restarted.solution.x.array, reference.solution.x.array,
        rtol=1e-8, atol=1e-9,
    )
    assert restarted.energy_history[-1].friction_dissipation == pytest.approx(
        reference.energy_history[-1].friction_dissipation, rel=1e-8, abs=1e-9,
    )
    assert "friction_dissipation" in result.histories
    assert "friction_separation_release" in result.histories
    assert result.metadata["mechanical_energy_ledger"]["friction_separation_release"] == (
        "reported_numerical_loss_not_coulomb_dissipation"
    )


def test_public_friction_failure_rolls_back_joint_state(monkeypatch):
    step = _frictional_j2_step(tangential_load=100.0)
    step.solve(until=0.5)
    accepted_solution = step.solution.x.array.copy()
    accepted_friction = step.contact_provider.snapshot_accepted_state()
    accepted_energy = tuple(step.energy_history)
    commit = step.contact_provider.commit_accepted

    def fail_after_contact_commit():
        commit()
        raise RuntimeError("injected post-contact-commit failure")

    monkeypatch.setattr(step.contact_provider, "commit_accepted", fail_after_contact_commit)
    with pytest.raises(RuntimeError, match="injected post-contact-commit failure"):
        step.solve()
    assert step.accepted_load_factor == pytest.approx(0.5)
    np.testing.assert_array_equal(step.solution.x.array, accepted_solution)
    assert step.contact_provider.snapshot_accepted_state() == accepted_friction
    assert tuple(step.energy_history) == accepted_energy


def test_public_friction_checkpoint_corruption_is_atomic(tmp_path):
    partial = _frictional_j2_step(tangential_load=100.0)
    partial.solve(until=0.5)
    path = partial.save_checkpoint(tmp_path / "friction.npz")
    with np.load(path, allow_pickle=False) as archive:
        damaged = {name: archive[name].copy() for name in archive.files}
    state = json.loads(str(damaged["contact_state"]))
    state["point_ids"][0] = -1
    damaged["contact_state"] = json.dumps(state)
    corrupt = tmp_path / "corrupt_friction.npz"
    np.savez_compressed(corrupt, **damaged)
    rejected = _frictional_j2_step(tangential_load=100.0)
    initial_contact = rejected.contact_provider.snapshot_accepted_state()
    with pytest.raises(ValueError, match="point IDs"):
        rejected.load_checkpoint(corrupt)
    assert rejected.accepted_load_factor == 0.0
    assert not rejected.energy_history
    assert rejected.contact_provider.snapshot_accepted_state() == initial_contact
    state = json.loads(str(damaged["contact_state"]))
    state["point_ids"][0] = partial.contact_provider.adapter.trace.point_ids[0].item()
    state["kinematics"]["slave_positions"][0][1] += 0.05
    damaged["contact_state"] = json.dumps(state)
    inconsistent = tmp_path / "inconsistent_friction.npz"
    np.savez_compressed(inconsistent, **damaged)
    with pytest.raises(ValueError, match="accepted kinematics differ"):
        rejected.load_checkpoint(inconsistent)
    assert rejected.accepted_load_factor == 0.0
    assert rejected.contact_provider.snapshot_accepted_state() == initial_contact
