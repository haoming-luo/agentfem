"""Bounded AL increment admission using the existing standard-J2 Procedure."""

from dataclasses import replace
import json

import numpy as np
import pytest

from agentfem import amplitudes, boundary_models, results, steps
from agentfem.step_providers import step_capability
from test_finite_j2_facet_contact import _patch


def _problem(*, penalty=500.0, maximum_augmentations=60, **options):
    _, displacement, _, problem = _patch(
        triangulated=False, initial_penetration_limit=0.0,
        penalty=penalty,
        normal_augmentation=steps.normal_augmentation(
            penetration_tolerance=1e-7, complementarity_tolerance=1e-6,
            maximum_augmentations=maximum_augmentations,
            length_unit="m", pressure_unit="Pa",
        ), **options,
    )
    return displacement, problem


def _state(problem):
    return problem.response.state.committed_state_vectors().copy()


def test_plastic_normal_augmentation_commits_material_once_per_accepted_increment(monkeypatch):
    displacement, problem = _problem(increments=3)
    transaction = problem.state_transaction
    original_update = transaction.refresh_trial
    original_commit = transaction.commit_increment
    boundaries = []
    inner_states = []

    def refresh(**kwargs):
        inner_states.append((kwargs['start_factor'], _state(problem)))
        return original_update(**kwargs)

    def commit(**kwargs):
        boundaries.append((kwargs['start_factor'], kwargs['target_factor']))
        return original_commit(**kwargs)

    monkeypatch.setattr(transaction, 'refresh_trial', refresh)
    monkeypatch.setattr(transaction, 'commit_increment', commit)
    result = problem.solve_result()
    assert result.status == 'completed'
    assert len(boundaries) == 3
    for start in {item[0] for item in inner_states}:
        arrays = [array for factor, array in inner_states if factor == start]
        for array in arrays[1:]:
            np.testing.assert_array_equal(array, arrays[0])
    assert all(info.normal_augmentations > 0 for info in problem.accepted_increments)
    assert np.max(_state(problem)[:, -2]) > 0
    assert results.probe(displacement, at=(1., .5, .5))[0] == pytest.approx(.005, abs=2e-7)
    provider = problem.contact_provider
    diagnostic = provider.current_point_diagnostics()['normal_augmentation']
    assert diagnostic['maximum_penetration'] <= 1e-7
    assert diagnostic['maximum_complementarity'] <= 1e-6
    assert not problem.energy_history_complete
    assert not problem.energy_history
    assert result.metadata['mechanical_energy_ledger']['status'].startswith('unavailable_augmented')
    assert f'{provider.name}_contact_energy' not in result.quantities
    assert not hasattr(provider.evaluate(), 'potential_energy')


def test_augmented_restart_matches_continuous_plastic_path(tmp_path):
    _, full = _problem(increments=3)
    full.solve()
    _, partial = _problem(increments=3)
    partial.solve(until=2/3)
    checkpoint = partial.save_checkpoint(tmp_path / 'augmented_plane')
    _, resumed = _problem(increments=3)
    resumed.load_checkpoint(checkpoint)
    resumed.solve()
    np.testing.assert_allclose(resumed.solution.x.array, full.solution.x.array, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(_state(resumed), _state(full), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        resumed.contact_provider.normal_multiplier_state.accepted_multiplier,
        full.contact_provider.normal_multiplier_state.accepted_multiplier,
        rtol=1e-10, atol=1e-12,
    )
    assert [i.normal_augmentations for i in resumed.accepted_increments] == [
        i.normal_augmentations for i in full.accepted_increments
    ]
    _, incompatible = _problem(penalty=600., increments=3)
    before = incompatible.solution.x.array.copy()
    with pytest.raises(ValueError):
        incompatible.load_checkpoint(checkpoint)
    np.testing.assert_array_equal(incompatible.solution.x.array, before)


def test_accepted_pressure_export_matches_actual_augmented_force(tmp_path):
    _, problem = _problem(increments=3)
    result = problem.solve_result()
    provider = problem.contact_provider
    paths = results.export_accepted_contact_points(problem, tmp_path, result=result, vtk_view=True)
    metadata = json.loads(paths['contact_point_metadata'].read_text())
    assert metadata['pressure_measure'].startswith('augmented_force')
    assert metadata['location'] == 'coalesced_reference_boundary_nodes'
    assert metadata['pairs'][0]['enforcement'].startswith('fixed_trace_normal_augmented')
    with np.load(paths['contact_points'], allow_pickle=False) as data:
        force = np.sum(data['normal_pressures'][:, None] * data['reference_weights'][:, None]
                       * data['surface_normals'], axis=0)
        np.testing.assert_allclose(force, provider.evaluate().contact_force_on_structure,
                                   rtol=1e-12, atol=1e-12)
        assert np.max(data['normal_pressures']) > 1.
        assert np.max(provider.pair.law.penalty * data['penetrations']) < 1e-3
    for method in (provider.quadrature_sensitivity, provider.reference_subdivision_sensitivity,
                   provider.closed_penalty_energy_enclosure):
        with pytest.raises(NotImplementedError, match='AFM-CONTACT-AL'):
            method()


@pytest.mark.parametrize('failure', ['outer_limit', 'newton_limit', 'commit'])
def test_augmentation_failure_restores_entire_increment(failure, monkeypatch):
    _, problem = _problem(increments=3)
    problem.solve(until=1/3)
    provider = problem.contact_provider
    before_solution = problem.solution.x.array.copy()
    before_state = _state(problem)
    before_contact = json.dumps(provider.snapshot_accepted_state(), sort_keys=True)
    original_commit = provider.commit_accepted
    if failure == 'outer_limit':
        provider.normal_multiplier_state.maximum_augmentations = 1
        before_contact = json.dumps(provider.snapshot_accepted_state(), sort_keys=True)
    elif failure == 'newton_limit':
        problem.solver_options = replace(problem.solver_options, maximum_iterations=1)
    else:
        def fail_after_commit():
            original_commit()
            raise RuntimeError('injected after multiplier commit')
        monkeypatch.setattr(provider, 'commit_accepted', fail_after_commit)
    with pytest.raises(RuntimeError):
        problem.solve(until=2/3)
    assert problem.accepted_load_factor == pytest.approx(1/3)
    assert len(problem.accepted_increments) == 1
    np.testing.assert_array_equal(problem.solution.x.array, before_solution)
    np.testing.assert_array_equal(_state(problem), before_state)
    assert json.dumps(provider.snapshot_accepted_state(), sort_keys=True) == before_contact


def test_inner_newton_success_cannot_be_checkpointed(monkeypatch, tmp_path):
    _, problem = _problem(increments=1)
    original = problem._solve_frozen_increment
    blocked = []

    def inner(**kwargs):
        info = original(**kwargs)
        if info.converged:
            with pytest.raises(RuntimeError, match='AL checkpoint'):
                problem.contact_provider.snapshot_accepted_state()
            with pytest.raises(RuntimeError):
                problem.save_checkpoint(tmp_path / 'forbidden_inner')
            blocked.append(True)
        return info

    monkeypatch.setattr(problem, '_solve_frozen_increment', inner)
    problem.solve()
    assert len(blocked) > 1
    assert not list(tmp_path.glob('forbidden_inner*'))
    problem.save_checkpoint(tmp_path / 'accepted')


def test_automatic_cutback_discards_failed_outer_iterates(monkeypatch):
    _, problem = _problem(
        increment_control=steps.automatic(initial=.5, maximum=.5, minimum=.01,
                                          max_increments=50, max_cutbacks=4),
        maximum_augmentations=120,
    )
    original = problem._solve_frozen_increment
    visits = []
    injected = False

    def inner(**coordinates):
        nonlocal injected
        state = problem.contact_provider.normal_multiplier_state
        visits.append((coordinates['attempt'], state.augmentation_count,
                       state.accepted_multiplier.copy(), _state(problem)))
        info = original(**coordinates)
        if not injected and state.augmentation_count == 1:
            injected = True
            return replace(info, converged=False, rejection_reason='test frozen-equilibrium failure')
        return info

    monkeypatch.setattr(problem, '_solve_frozen_increment', inner)
    problem.solve()
    assert injected
    assert not problem.attempted_increments[0].converged
    assert problem.attempted_increments[0].normal_augmentations == 1
    first_retry = next(item for item in visits if item[0] == 2)
    assert first_retry[1] == 0
    np.testing.assert_array_equal(first_retry[2], visits[0][2])
    np.testing.assert_array_equal(first_retry[3], visits[0][3])
    assert problem.accepted_load_factor == 1.
    assert len(problem.accepted_increments) < 20
    assert all(item.converged for item in problem.accepted_increments)
    assert all(item.adaptation_iterations == item.maximum_inner_newton_iterations
               for item in problem.accepted_increments)
    assert any(item.iterations > item.adaptation_iterations for item in problem.accepted_increments)
    assert problem.contact_provider.normal_multiplier_state.augmentation_count == 0


def test_augmented_loading_then_separation_releases_pressure():
    path = amplitudes.tabular((0., .5, 1.), (0., 1., 0.))
    displacement, problem = _problem(increments=6, load_amplitude=path)
    problem.solve(until=.5)
    assert np.max(problem.contact_provider.normal_multiplier_state.accepted_multiplier) > 0
    plastic = _state(problem)
    problem.solve()
    assert results.probe(displacement, at=(1., .5, .5))[0] < .005
    np.testing.assert_allclose(problem.contact_provider.normal_multiplier_state.accepted_multiplier, 0.)
    assert problem.contact_provider.current_point_diagnostics()['maximum_pressure'] == 0.
    assert np.max(_state(problem)[:, -2]) >= np.max(plastic[:, -2])


@pytest.mark.parametrize('cells,penalty', [(1, 250.), (1, 1000.), (2, 250.), (2, 1000.)])
def test_augmented_elastic_limit_matches_independent_nominal_stress(cells, penalty):
    displacement, problem = _problem(
        cells=cells, penalty=penalty, yield_stress=1e9, increments=1,
        maximum_augmentations=150,
    )
    problem.solve()
    stretch = 1.005
    exact_nominal_stress = 1000. * np.log(stretch) / stretch
    force = problem.contact_provider.evaluate().contact_force_on_structure
    assert force[0] == pytest.approx(-(10. - exact_nominal_stress), abs=2e-4)
    np.testing.assert_allclose(force[1:], 0., atol=1e-12)
    assert results.probe(displacement, at=(1., .5, .5))[0] == pytest.approx(.005, abs=1e-7)
    assert np.max(_state(problem)[:, -2]) == 0.


def test_augmented_plastic_limit_matches_independently_prescribed_displacement():
    _, augmented = _problem(increments=1, maximum_augmentations=100)
    augmented.solve()
    _, _, _, prescribed = _patch(
        triangulated=False, initial_penetration_limit=0., increments=1,
        constrain_contact=True, contact_value=.005,
    )
    prescribed.solve()
    np.testing.assert_allclose(augmented.solution.x.array, prescribed.solution.x.array, atol=2e-7)
    np.testing.assert_allclose(_state(augmented), _state(prescribed), rtol=1e-4, atol=3e-7)
    assert np.max(_state(augmented)[:, -2]) > 0.


@pytest.mark.parametrize('cells,yield_stress', [(2, 1e9), (3, 1e9), (2, 2.)])
def test_oblique_nonuniform_pressure_matches_independent_displacement_limit(cells, yield_stress):
    """Independent FE normal reactions, not a continuum pressure-convergence claim."""
    from dolfinx import fem
    from agentfem import constraints, solvers

    gap, tilt = .005, .002
    options = dict(triangulated=False, initial_penetration_limit=0.,
                   cells=cells, yield_stress=yield_stress,
                   transverse_boundary_fixed=True, gap=gap, tilt=tilt,
                   applied_traction=10., make_step=False)
    augmented_model, u, material, _ = _patch(**options)
    augmented = augmented_model.step(
        target=u, material=material, incrementation=steps.fixed(1), progress=False,
        normal_augmentation=steps.normal_augmentation(
            penetration_tolerance=1e-8, complementarity_tolerance=1e-7,
            length_unit='m', pressure_unit='Pa', maximum_augmentations=200,
        ),
        solver_options=solvers.newton(relative_tolerance=1e-9, absolute_tolerance=1e-10),
    )
    reference_model, reference_u, reference_material, _ = _patch(**options)
    reference_model.boundary_models.clear()
    space = reference_u.value.function_space
    prescribed = fem.Function(space)
    prescribed.interpolate(lambda x: np.vstack((gap + tilt*x[1], np.zeros_like(x[0]), np.zeros_like(x[0]))))
    dofs = fem.locate_dofs_geometrical(space, lambda x: np.isclose(x[0], 1.))
    reference_model.constraints.append(constraints.DirichletConstraint(
        fem.dirichletbc(prescribed, dofs), value=prescribed, name='independent_oblique_limit',
    ))
    reference = reference_model.step(
        target=reference_u, material=reference_material,
        incrementation=steps.fixed(1), progress=False,
        solver_options=solvers.newton(relative_tolerance=1e-9, absolute_tolerance=1e-10),
    )
    reference.solve()
    augmented.solve()
    np.testing.assert_allclose(augmented.solution.x.array, reference.solution.x.array, atol=3e-8)
    np.testing.assert_allclose(_state(augmented), _state(reference), rtol=3e-4, atol=1e-7)
    provider = augmented.contact_provider
    sample = provider.accepted_point_samples()
    # The oracle contains no contact law. Normal reaction comes exclusively
    # from its assembled J2-minus-external residual at prescribed contact nodes.
    reaction = reference.reaction_field().x.array.reshape(-1, 3)
    nodes = provider.adapter.trace._direct_nodes
    pressure = reaction[nodes, 0] / (sample['reference_weights'] * sample['surface_normals'][:, 0])
    assert np.min(pressure) > 0.
    assert np.ptp(pressure) > .1
    np.testing.assert_allclose(sample['normal_pressures'], pressure, rtol=2e-4, atol=2e-4)
    assert np.all(sample['active'])


@pytest.mark.parametrize('cells', [2, 3])
def test_partial_contact_matches_independent_primal_dual_active_set(cells):
    """Test-only equality-constraint oracle; no AL/penalty assembly in the oracle.

    A fresh elastic-limit J2 problem is solved for each guessed active set.
    Discrete nodal reaction and gap select the next set. This is independent
    enforcement of the same nodal obstacle, not an external continuum oracle.
    """
    from dolfinx import fem
    from agentfem import constraints, solvers

    gap, tilt = .001, .02
    inputs = dict(triangulated=False, initial_penetration_limit=0., cells=cells,
                  yield_stress=1e9, transverse_boundary_fixed=True,
                  gap=gap, tilt=tilt, applied_traction=10., make_step=False)
    model, u, material, _ = _patch(**inputs)
    newton = solvers.newton(relative_tolerance=1e-9, absolute_tolerance=1e-10)
    augmented = model.step(
        target=u, material=material, incrementation=steps.fixed(1), progress=False,
        solver_options=newton,
        normal_augmentation=steps.normal_augmentation(
            penetration_tolerance=1e-8, complementarity_tolerance=1e-7,
            maximum_augmentations=200, length_unit='m', pressure_unit='Pa',
        ),
    )
    space = u.value.function_space
    coordinates = space.tabulate_dof_coordinates()
    right = fem.locate_dofs_geometrical(space, lambda x: np.isclose(x[0], 1.))
    obstacle = gap + tilt * coordinates[right, 1]
    active = np.zeros(len(right), dtype=bool)
    for _ in range(12):
        oracle_model, oracle_u, oracle_material, _ = _patch(**inputs)
        oracle_model.boundary_models.clear()
        oracle_space = oracle_u.value.function_space
        prescribed = fem.Function(oracle_space)
        prescribed.interpolate(lambda x: np.vstack((gap + tilt*x[1], np.zeros_like(x[0]), np.zeros_like(x[0]))))
        if np.any(active):
            oracle_model.constraints.append(constraints.DirichletConstraint(
                fem.dirichletbc(prescribed, right[active]), value=prescribed,
                name='test_only_active_set',
            ))
        oracle = oracle_model.step(
            target=oracle_u, material=oracle_material, incrementation=steps.fixed(1),
            solver_options=newton, progress=False,
        )
        oracle.solve()
        ux = oracle.solution.x.array.reshape(-1, 3)[right, 0]
        nodal_reaction = oracle.reaction_field().x.array.reshape(-1, 3)[right, 0]
        next_active = -nodal_reaction - 100. * (obstacle - ux) > 1e-9
        if np.array_equal(next_active, active):
            break
        active = next_active
    else:
        pytest.fail('Independent equality-based active set did not stabilize.')
    assert np.any(active) and np.any(~active)
    assert np.min(obstacle - ux) >= -1e-9
    assert np.max(nodal_reaction[active]) <= 1e-9
    augmented.solve()
    np.testing.assert_allclose(augmented.solution.x.array, oracle.solution.x.array, atol=4e-8)
    sample = augmented.contact_provider.accepted_point_samples()
    expected = dict(zip(right.tolist(), active.tolist()))
    nodes = augmented.contact_provider.adapter.trace._direct_nodes
    np.testing.assert_array_equal(sample['active'], [expected[int(node)] for node in nodes])
    np.testing.assert_allclose(
        sample['normal_pressures'] * sample['reference_weights'] * sample['surface_normals'][:, 0],
        oracle.reaction_field().x.array.reshape(-1, 3)[nodes, 0], atol=2e-5,
    )


def test_nonzero_dirichlet_backtracking_stays_feasible_and_matches_full_newton(monkeypatch):
    kwargs = dict(triangulated=False, initial_penetration_limit=0., increments=1,
                  constrain_contact=True, contact_value=.005)
    backtracking = _patch(**kwargs)[3]
    basic = _patch(**kwargs)[3]
    basic.solver_options = replace(basic.solver_options, line_search='basic')
    original = backtracking._update_response
    visited = []

    def check_boundary(**coordinates):
        actual = backtracking.solution.x.array.copy()
        constrained = actual.copy()
        for bc in backtracking.bcs:
            bc.set(constrained)
        np.testing.assert_allclose(actual, constrained, rtol=0., atol=1e-14)
        visited.append(True)
        return original(**coordinates)

    monkeypatch.setattr(backtracking, '_update_response', check_boundary)
    basic.solve()
    backtracking.solve()
    assert visited
    np.testing.assert_allclose(backtracking.solution.x.array, basic.solution.x.array,
                               rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(_state(backtracking), _state(basic), rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize('configuration', [
    {'triangulated': True}, {'hexahedral': True}, {'displacement_degree': 2},
    {'friction_coefficient': .2, 'tangential_penalty': 100.},
    {'contact_quadrature_degree': 5},
    {'constrain_contact': True, 'contact_value': .005},
])
def test_augmentation_scope_rejected_by_capability_and_execution(configuration):
    inputs = dict(triangulated=False, initial_penetration_limit=0., make_step=False)
    inputs.update(configuration)
    model, displacement, material, _ = _patch(**inputs)
    if 'friction_coefficient' in configuration:
        model.boundary_models[0] = replace(
            model.boundary_models[0],
            friction=boundary_models.penalty_coulomb_friction_law(.2, 100.),
        )
    options = steps.normal_augmentation(
        penetration_tolerance=1e-7, complementarity_tolerance=1e-6,
        length_unit='m', pressure_unit='Pa',
    )
    kwargs = dict(material=material, normal_augmentation=options)
    if 'contact_quadrature_degree' in configuration:
        kwargs['contact_quadrature_degree'] = configuration['contact_quadrature_degree']
    capability = step_capability(model, target=displacement, options=kwargs)
    assert not capability['supported']
    with pytest.raises((ValueError, NotImplementedError), match='AFM-CONTACT-AL'):
        model.step(target=displacement, **kwargs)


def test_damaged_multiplier_checkpoint_is_atomically_rejected(tmp_path):
    _, partial = _problem(increments=3)
    partial.solve(until=1/3)
    checkpoint = partial.save_checkpoint(tmp_path / 'state')
    with np.load(checkpoint, allow_pickle=False) as archive:
        fields = {key: archive[key].copy() for key in archive.files}
    state = json.loads(str(fields['contact_state']))
    state['multiplier']['multiplier'][0] = -1.
    fields['contact_state'] = np.asarray(json.dumps(state))
    damaged = tmp_path / 'damaged.npz'
    np.savez(damaged, **fields)
    _, target = _problem(increments=3)
    solution = target.solution.x.array.copy()
    material = _state(target)
    normal = json.dumps(target.contact_provider.snapshot_accepted_state(), sort_keys=True)
    with pytest.raises(ValueError):
        target.load_checkpoint(damaged)
    np.testing.assert_array_equal(target.solution.x.array, solution)
    np.testing.assert_array_equal(_state(target), material)
    assert json.dumps(target.contact_provider.snapshot_accepted_state(), sort_keys=True) == normal


@pytest.mark.parametrize('scope', ['moving', 'multiple', 'raw_vector_bc', 'wrong_options'])
def test_extra_unsupported_scopes_fail_in_public_inspection_and_execution(scope):
    from dolfinx import fem
    from agentfem.constraints import DirichletConstraint

    model, displacement, material, _ = _patch(
        triangulated=False, initial_penetration_limit=0., make_step=False,
    )
    pair = model.boundary_models[0]
    if scope == 'moving':
        model.boundary_models[0] = replace(pair, rigid_body=boundary_models.rigid_body(
            pair.rigid_body.surface,
            motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
                boundary_models.prescribed_rigid_motion(translation=(-.01, 0., 0.)),
                start_time=0., end_time=1.,
            ),
        ))
    elif scope == 'multiple':
        model.add_boundary_model(replace(pair, name='second'))
    elif scope == 'raw_vector_bc':
        space = displacement.value.function_space
        dofs = fem.locate_dofs_geometrical(space, lambda x: np.isclose(x[0], 1.))
        model.constraints.append(DirichletConstraint(fem.dirichletbc(np.zeros(3), dofs, space)))
    options = steps.normal_augmentation(
        penetration_tolerance=1e-7, complementarity_tolerance=1e-6,
        length_unit='m', pressure_unit='Pa',
    )
    if scope == 'wrong_options':
        options = {'penetration_tolerance': 1e-7}
    kwargs = dict(material=material, normal_augmentation=options)
    assert not step_capability(model, target=displacement, options=kwargs)['supported']
    with pytest.raises((ValueError, NotImplementedError), match='AFM-CONTACT-AL'):
        model.step(target=displacement, **kwargs)
