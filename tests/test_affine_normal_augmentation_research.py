# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded AL/MPC discrete references and ordinary Step lifecycle checks.

The patch excludes periodically fixed corners. We first inspect the actual
gap map, then compare the converged outer iteration against a separate exact
elimination of the same active nodal obstacle. Public Step lifecycle tests then
exercise the bounded admission. This is not an independent material law or
continuum pressure proof.
"""

import numpy as np
import pytest
from types import SimpleNamespace
from dolfinx import fem
from mpi4py import MPI

from agentfem import boundary_models, constitutive, constraints, fields, mesh, models, solvers, steps, studies
from agentfem._affine_equilibrium import solve_serial_affine_equilibrium
from agentfem.boundary_models._augmented_plane_trial_provider import DolfinxAugmentedPlaneTrialProvider
from agentfem.boundary_models.contact_affine_gap_research import affine_plane_gap_system
from agentfem.constraints.affine import AffineReduction
from periodic_cube_fixture import periodic_unit_cube


def _patch(cells=4, penalty=10000., *, increments=1, acceleration="none", moving=False,
           maximum_augmentations=60, public=False, whole_face=False, cyclic=False):
    fixture = periodic_unit_cube(MPI.COMM_SELF, cells=cells)
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=fixture.domain)
    u = model.field(fields.displacement(fixture.domain))
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000., poisson=.3, yield_stress=2., hardening_modulus=100.,
    ))
    if cyclic:
        gradients = [np.diag((s, 1 / np.sqrt(s), 1 / np.sqrt(s))) for s in (1., 1.02, .99, 1.02)]
        periodicity = model.constraint(constraints.abaqus_periodic_cell(
            u, nodes=fixture.nodes, equations=fixture.equations,
            deformation_gradient_path=constraints.deformation_gradient_path((0., .25, .75, 1.), gradients),
            anchor_node=fixture.anchor_node, reference_nodes=fixture.reference_nodes,
        ))
    else:
        periodicity = model.constraint(fixture.constraint(u))
    patch = mesh.boundary(
        fixture.domain,
        lambda x: (np.isclose(x[0], 1.) if whole_face else
                   np.isclose(x[0], 1.) & (x[1] >= .25-1e-12)
                   & (x[1] <= .75+1e-12) & (x[2] >= .25-1e-12)
                   & (x[2] <= .75+1e-12)),
        name="interior_periodic_face_patch",
    )
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        patch, boundary_models.rigid_body(
            boundary_models.rigid_plane(point=(1.005, 0., 0.), normal=(-1., 0., 0.)),
            motion_schedule=(boundary_models.prescribed_rigid_motion_schedule(
                boundary_models.prescribed_rigid_motion(
                    translation=(-.002, 0., 0.), reference_point=(1.005, 0., 0.),
                ), start_time=0., end_time=1.,
            ) if moving else None),
        ), penalty=penalty, name="interior_plane",
    ))
    policy = steps.normal_augmentation(
        penetration_tolerance=1e-8, complementarity_tolerance=1e-7,
        maximum_augmentations=maximum_augmentations, length_unit="m", pressure_unit="Pa",
        acceleration=acceleration,
    )
    step = model.step(
        target=u, material=material, constraints=periodicity,
        incrementation=steps.fixed(increments), progress=False,
        **({"normal_augmentation": policy} if public else {}),
        solver_options=solvers.newton(relative_tolerance=1e-9, absolute_tolerance=1e-10,
                                     maximum_iterations=60, line_search="backtracking"),
    )
    provider = (step.contact_provider if public else DolfinxAugmentedPlaneTrialProvider(
        step.contact_provider, policy=policy,
    ))
    return step, provider


def _research_path(**options):
    """Exercise ordinary model.step() after the independent discrete checks."""
    step, _ = _patch(increments=4, public=True, **options)
    return step


def _solve(step, reduction, matrix, values, operators):
    solvers._assign_reconstructed(step.solution, reduction, values)
    return solve_serial_affine_equilibrium(
        fem.form(step.residual_form), fem.form(step.jacobian_form),
        step.solution, reduction, matrix, values,
        options=step.solver_options.for_affine_reduction(), operators=operators,
        transaction=step.state_transaction, start_factor=0., target_factor=1.,
        reporter=None, step_name="research_only", step_number=1,
        increment_number=1, attempt_number=1,
    )


@pytest.mark.parametrize("cells,penalty", [(4, 10000.), (4, 40000.), (8, 20000.)])
def test_compatible_periodic_patch_al_matches_exact_active_constraint_elimination(cells, penalty):
    step, provider = _patch(cells=cells, penalty=penalty)
    step.state_transaction.initialize()
    committed = step.response.state.committed_state_vectors().copy()
    reduction = step.constraint.reduction(1.)
    matrix = reduction.matrix(MPI.COMM_SELF)
    dense = matrix.copy()
    dense.convert("dense")
    try:
        transformation = dense.getDenseArray().copy()
    finally:
        dense.destroy()
    gap_map = affine_plane_gap_system(
        provider.adapter.trace, provider.adapter.reference_nodal_positions,
        transformation, reduction.offset, provider.surface,
    )
    point_count = (cells // 2 + 1) ** 2
    assert len(gap_map.offset) == point_count
    assert np.linalg.matrix_rank(gap_map.matrix) == point_count
    assert not gap_map.stationary_penetration(gap_tolerance=1e-10)["incompatible_point_ids"]
    initial = step.constraint.initial_reduced_values(
        reduction, step.constraint.deformation_gradient_at(1.),
    )
    values = initial.copy()
    ready = False
    try:
        for _ in range(provider.normal_multiplier_state.maximum_augmentations + 1):
            info = _solve(step, reduction, matrix, values, (provider,))
            assert info.converged, info
            values = info.reduced_values
            # Confirm total equilibrium, not only the contact law.
            norm = solvers._reduced_residual_norm(
                fem.form(step.residual_form), matrix, (provider,),
            )
            ready, diagnostics = provider.finish_frozen_equilibrium(
                residual_norm=norm, residual_tolerance=info.threshold,
            )
            np.testing.assert_array_equal(step.response.state.committed_state_vectors(), committed)
            assert step.state_transaction.accepted_factor == 0.
            if ready:
                break
            if provider.normal_multiplier_state.augmentation_count == (
                provider.normal_multiplier_state.maximum_augmentations
            ):
                break
            provider.augment()
        assert ready, diagnostics
        augmented_u = step.solution.x.array.copy()
        augmented_state = step.response.state.trial_state_vectors().copy()
        response = provider.normal_response()
        assert np.all(response.pressure > 0.)

        # This fixture's gap rows select distinct reduced Cartesian DOFs.
        # Eliminate them exactly, without augmented or penalty forces.
        columns = []
        qbar = np.zeros(reduction.reduced_size)
        for row, offset in zip(gap_map.matrix, gap_map.offset, strict=True):
            nonzero = np.flatnonzero(row)
            assert len(nonzero) == 1
            column = int(nonzero[0])
            columns.append(column)
            qbar[column] = -offset / row[column]
        assert len(set(columns)) == len(columns)
        free = np.setdiff1d(np.arange(reduction.reduced_size), columns)
        from scipy.sparse import csr_matrix
        hard_matrix = csr_matrix(transformation[:, free])
        hard = AffineReduction(
            hard_matrix.indptr, hard_matrix.indices, hard_matrix.data,
            reduction.offset + transformation @ qbar,
            reduction.independent_full_dofs[free],
        )
        hard_T = hard.matrix(MPI.COMM_SELF)
        try:
            reference = _solve(step, hard, hard_T, initial[free], ())
            assert reference.converged
            np.testing.assert_allclose(step.solution.x.array, augmented_u, atol=5e-8, rtol=1e-6)
            np.testing.assert_allclose(step.response.state.trial_state_vectors(),
                                       augmented_state, atol=2e-7, rtol=2e-6)
            body, _ = solvers._assemble_affine_full_residual(fem.form(step.residual_form), ())
            try:
                reduced_body = transformation.T @ body.array_r
            finally:
                body.destroy()
            weighted_pressure, *_ = np.linalg.lstsq(gap_map.matrix.T, reduced_body, rcond=None)
            pressure = weighted_pressure / gap_map.weights
            np.testing.assert_allclose(pressure, response.pressure, rtol=2e-5, atol=1e-5)
            # State column 9 is equivalent plastic strain for this J2 law,
            # not an identity deformation-gradient component.
            assert np.max(augmented_state[:, 9]) > 1e-3
            assert step.constraint.mismatch(1.) < 1e-12
        finally:
            hard_T.destroy()
    finally:
        matrix.destroy()
        step.state_transaction.rollback_increment(accepted_factor=0.)


@pytest.mark.parametrize("moving,acceleration", [(False, "none"), (True, "aitken")])
def test_affine_outer_path_restart_and_single_material_commit(tmp_path, monkeypatch,
                                                             moving, acceleration):
    full = _research_path(moving=moving, acceleration=acceleration)
    original = full.state_transaction.commit_increment
    commits = []
    expected_work = []

    def commit(**kwargs):
        commits.append(kwargs["target_factor"])
        original(**kwargs)

    monkeypatch.setattr(full.state_transaction, "commit_increment", commit)
    original_contact_commit = full.contact_provider.commit_accepted

    def contact_commit():
        expected_work.append(full.contact_provider.normal_work_increment())
        original_contact_commit()

    monkeypatch.setattr(full.contact_provider, "commit_accepted", contact_commit)
    result = full.solve_result()
    assert "contact_potential" not in result.quantities
    assert "interior_plane_normal_relative_work" in result.histories
    assert not result.metadata["normal_contact_work:interior_plane"]["automatic_energy_verification"]
    assert commits == [.25, .5, .75, 1.]
    assert any(i.checks["normal_augmentations"] > 0 for i in full.accepted_increments)
    partial = _research_path(moving=moving, acceleration=acceleration)
    partial.solve(until=.5)
    saved = partial.save_checkpoint(tmp_path / "private_affine_al")
    resumed = _research_path(moving=moving, acceleration=acceleration)
    resumed.load_checkpoint(saved)
    resumed.solve()
    np.testing.assert_allclose(resumed.solution.x.array, full.solution.x.array, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(
        resumed.response.state.committed_state_vectors(),
        full.response.state.committed_state_vectors(), atol=1e-12, rtol=1e-10,
    )
    np.testing.assert_allclose(
        resumed.contact_provider.normal_multiplier_state.accepted_multiplier,
        full.contact_provider.normal_multiplier_state.accepted_multiplier,
        atol=1e-10, rtol=1e-9,
    )
    assert [i.checks for i in resumed.accepted_increments] == [i.checks for i in full.accepted_increments]
    key = "normal_contact_work:interior_plane"
    rows = full.accepted_history_recorders[key].records
    for left, right, work in zip(rows[:-1], rows[1:], expected_work, strict=True):
        assert right["work"] - left["work"] == pytest.approx(work.relative_work, abs=1e-14)
        assert right["absolute_bound"] - left["absolute_bound"] == pytest.approx(work.absolute_work_bound)
    assert resumed.accepted_history_recorders[key].checkpoint_state() == (
        full.accepted_history_recorders[key].checkpoint_state()
    )
    if moving:
        assert resumed.operator_dual_history.complete(accepted_factor=1.)
        assert resumed.operator_dual_history.work("interior_plane") == pytest.approx(
            full.operator_dual_history.work("interior_plane"),
        )


@pytest.mark.parametrize("failure", ["outer_limit", "after_contact_commit"])
def test_affine_outer_path_failure_restores_joint_state(monkeypatch, failure):
    step = _research_path()
    step.solve(until=.5)
    before_u = step.solution.x.array.copy()
    before_state = step.response.state.committed_state_vectors().copy()
    before_pressure = step.contact_provider.normal_multiplier_state.accepted_multiplier.copy()
    before_checks = [i.checks for i in step.accepted_increments]
    before_work = step.accepted_history_recorders["normal_contact_work:interior_plane"].checkpoint_state()
    if failure == "outer_limit":
        original = step.contact_provider.finish_frozen_equilibrium

        def never_ready(**kwargs):
            _, diagnostics = original(**kwargs)
            return False, diagnostics

        monkeypatch.setattr(step.contact_provider, "finish_frozen_equilibrium", never_ready)
    else:
        original = step.contact_provider.commit_accepted

        def commit_then_fail():
            original()
            raise RuntimeError("injected after contact commit")

        monkeypatch.setattr(step.contact_provider, "commit_accepted", commit_then_fail)
    with pytest.raises(RuntimeError):
        step.solve(until=.75)
    np.testing.assert_array_equal(step.solution.x.array, before_u)
    np.testing.assert_array_equal(step.response.state.committed_state_vectors(), before_state)
    np.testing.assert_array_equal(
        step.contact_provider.normal_multiplier_state.accepted_multiplier, before_pressure,
    )
    assert step.accepted_load_factor == .5
    assert [i.checks for i in step.accepted_increments] == before_checks
    assert step.contact_provider.normal_multiplier_state.augmentation_count == 0
    assert not step.contact_provider._outer_in_progress
    assert step.accepted_history_recorders["normal_contact_work:interior_plane"].checkpoint_state() == before_work


def test_public_affine_augmentation_rejects_prescribed_corner_normal_motion():
    with pytest.raises(NotImplementedError, match="AFM-CONTACT-AL-MPC-ROWS"):
        _patch(public=True, whole_face=True)


def test_sparse_normal_row_guard_is_sufficient_not_a_general_rank_test():
    from scipy.sparse import csr_matrix
    from agentfem.boundary_models._augmented_plane_trial_provider import validate_affine_plane_normal_rows
    trace = SimpleNamespace(_direct_nodes=np.array([0, 1]), point_ids=np.array([10, 20]))

    def reduction(first, second):
        matrix = csr_matrix(np.array([first, [0., 0.], [0., 0.],
                                      second, [0., 0.], [0., 0.]]))
        return AffineReduction(matrix.indptr, matrix.indices, matrix.data,
                               np.zeros(6), np.array([0, 3]))

    assert validate_affine_plane_normal_rows(trace, (1., 0., 0.), reduction([1, 0], [0, 1]))[
        "point_count"
    ] == 2
    # Independent but overlapping rows are outside the sufficient admission.
    # That rejection is not an infeasibility theorem.
    with pytest.raises(NotImplementedError, match="overlapping"):
        validate_affine_plane_normal_rows(trace, (1., 0., 0.), reduction([1, 1], [0, 1]))
    with pytest.raises(NotImplementedError, match="prescribed"):
        validate_affine_plane_normal_rows(trace, (1., 0., 0.), reduction([0, 0], [0, 1]))


def test_affine_outer_automatic_retry_rolls_back_multiplier_before_retry(monkeypatch):
    step = _research_path(acceleration="aitken")
    step.incrementation = steps.automatic(initial=.5, minimum=.01, maximum=.5)
    original = step._equilibrium_procedure.solve
    from dataclasses import replace
    rejected = False

    def reject_first(*args, **kwargs):
        nonlocal rejected
        response = original(*args, **kwargs)
        if not rejected and kwargs["target_factor"] == .5:
            rejected = True
            return replace(response, converged=False, message="injected converged trial rejection")
        return response

    monkeypatch.setattr(step._equilibrium_procedure, "solve", reject_first)
    step.solve()
    assert rejected
    assert not step.attempted_increments[0].converged
    assert step.accepted_load_factor == 1.
    assert step.contact_provider.normal_multiplier_state.augmentation_count == 0
    assert len(step.accepted_history_recorders["normal_contact_work:interior_plane"].records) == (
        len(step.accepted_increments) + 1
    )


@pytest.mark.parametrize("corruption", ["pressure", "work_bound", "count", "nonnumeric"])
def test_normal_work_checkpoint_corruption_is_atomically_rejected(tmp_path, corruption):
    import json
    step = _research_path()
    step.solve(until=.5)
    saved = step.save_checkpoint(tmp_path / "corrupt_normal_work")
    payload = json.loads(saved.read_text())
    key = "normal_contact_work:interior_plane"
    record = payload["accepted_observer_state"][key]
    if corruption == "pressure":
        record["last_sample"]["normal_pressures"][0] += 1.
    elif corruption == "work_bound":
        record["records"][-1]["absolute_bound"] = -1.
    elif corruption == "nonnumeric":
        record["last_sample"]["signed_gaps"][0] = "not a number"
    else:
        record["records"].pop(1)
    # No hash covers the checkpoint's own JSON metadata; the State owner must
    # validate deliberately corrupted scalar fields during restore.
    saved.write_text(json.dumps(payload))
    target = _research_path()
    before = target.response.state.committed_state_vectors().copy()
    with pytest.raises(ValueError, match="Normal-work"):
        target.load_checkpoint(saved)
    assert target.accepted_load_factor == 0.
    np.testing.assert_array_equal(target.solution.x.array, 0.)
    np.testing.assert_array_equal(target.response.state.committed_state_vectors(), before)
    assert not target.accepted_history_recorders[key].records


def test_affine_cyclic_open_recontact_and_two_successive_restarts(tmp_path):
    def create():
        return _patch(public=True, increments=16, cyclic=True, acceleration="aitken")[0]

    full = create()
    full.solve()
    partial = create()
    partial.solve(until=.25)
    assert np.max(partial.contact_provider.normal_response().pressure) > 1.
    first = partial.save_checkpoint(tmp_path / "loaded")
    unloaded = create()
    unloaded.load_checkpoint(first)
    unloaded.solve(until=.75)
    np.testing.assert_allclose(unloaded.contact_provider.normal_response().pressure, 0., atol=1e-12)
    second = unloaded.save_checkpoint(tmp_path / "unloaded")
    resumed = create()
    resumed.load_checkpoint(second)
    resumed.solve()
    assert np.max(resumed.contact_provider.normal_response().pressure) > 1.
    np.testing.assert_allclose(resumed.solution.x.array, full.solution.x.array, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(resumed.response.state.committed_state_vectors(),
                               full.response.state.committed_state_vectors(), atol=1e-12, rtol=1e-10)
    key = "normal_contact_work:interior_plane"
    assert resumed.accepted_history_recorders[key].checkpoint_state() == (
        full.accepted_history_recorders[key].checkpoint_state()
    )
    assert [i.checks for i in resumed.accepted_increments] == [i.checks for i in full.accepted_increments]


def test_displacement_frame_thinning_does_not_thin_accepted_normal_work():
    dense = _research_path()
    dense.output_every = 1
    dense.solve()
    sparse = _research_path()
    sparse.output_every = None
    sparse.solve()
    assert len(dense.snapshots) == 5
    assert len(sparse.snapshots) == 2
    key = "normal_contact_work:interior_plane"
    assert dense.accepted_history_recorders[key].checkpoint_state() == (
        sparse.accepted_history_recorders[key].checkpoint_state()
    )
    np.testing.assert_array_equal(dense.solution.x.array, sparse.solution.x.array)
