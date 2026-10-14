# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Numerical-policy lifecycle, not physical validation of a scaled solid."""
import numpy as np
import pytest

from agentfem import time
from test_response_material_step import make_step, contact_step
from test_variable_scaled_inertia import assert_snapshot_equal


def scaling(*, update_every=None, budget=100., calls=None, multiplier=1., identity='rows-v1'):
    def bounds(request):
        if calls is not None:
            calls.append(request.increment)
        for name in ('physical_nodal_mass', 'reference_coordinates', 'cell_nodes',
                     'displacement', 'velocity', 'current_factors'):
            array = getattr(request, name)
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.setflags(write=True)
        # Known diagonal-majorant fixture, deliberately not a user-law estimate.
        factor = 2. if request.increment < 4 else 3.
        k = np.repeat(request.physical_nodal_mass[:, None], 3, axis=1)
        k *= 4 * factor * multiplier / (1e-4 / .8)**2
        return time.MassScalingBounds(
            {name: k / len(request.stiffness_names) for name in request.stiffness_names},
            {name: np.zeros_like(k) for name in request.damping_names})
    return time.TargetIncrementMassScaling(
        target_increment=1e-4, bounds=bounds, identity={'fixture': identity},
        maximum_added_mass_ratio=budget, update_every=update_every)


def test_fixed_policy_matches_existing_factor_without_file():
    fresh = make_step(mass_scaling=time.FixedMassScaling(2., maximum_added_mass_ratio=2.))
    old = make_step(mass_scaling_factor=2.)
    fresh.run()
    old.run()
    for field in ('u', 'v', 'a'):
        np.testing.assert_array_equal(getattr(fresh.state, field).value.x.array,
                                      getattr(old.state, field).value.x.array)
    assert fresh.history_records == old.history_records
    assert fresh.summary()['numerical_inertia']['material_density_changed'] is False


def test_initial_target_calculates_factors_once_without_inertia_file():
    calls = []
    step = make_step(mass_scaling=scaling(calls=calls))
    assert calls == [0]
    np.testing.assert_allclose(step.numerical_inertia.factors, 2.)
    step.run()
    assert calls == [0]
    assert step.summary()['mass_scaling_policy']['target_increment'] == 1e-4


def test_variable_cadence_restart_and_energy(tmp_path):
    full_calls, resumed_calls = [], []
    full = make_step(steps=7, mass_scaling=scaling(update_every=3, calls=full_calls))
    full.run()
    assert full_calls == [1, 4, 7]
    partial = make_step(steps=7, mass_scaling=scaling(update_every=3))
    partial.run(until_step=2)
    path = partial.save_checkpoint(tmp_path / 'mass')
    resumed = make_step(steps=7, mass_scaling=scaling(update_every=3, calls=resumed_calls))
    resumed.load_checkpoint(path)
    resumed.run()
    assert resumed_calls == [4, 7]
    for field in ('u', 'v', 'a'):
        np.testing.assert_array_equal(getattr(full.state, field).value.x.array,
                                      getattr(resumed.state, field).value.x.array)
    assert full.history_records == resumed.history_records
    assert_snapshot_equal(full.numerical_inertia.snapshot(), resumed.numerical_inertia.snapshot())
    assert full.history_records[-1]['energy_balance_error'] == pytest.approx(0, abs=1e-12)
    assert full.numerical_inertia.update_count == 2
    changed = make_step(steps=7, mass_scaling=scaling(update_every=2))
    with pytest.raises(ValueError):
        changed.load_checkpoint(path)
    assert changed.completed_steps == 0


def test_fixed_bound_provider_identity_participates_in_restart(tmp_path):
    original = make_step(mass_scaling=scaling())
    original.run(until_step=2)
    path = original.save_checkpoint(tmp_path / 'fixed')
    different = make_step(mass_scaling=scaling(identity='changed-law-same-factors'))
    with pytest.raises(ValueError):
        different.load_checkpoint(path)
    assert different.completed_steps == 0


@pytest.mark.parametrize('variable', (False, True))
def test_budget_failure_does_not_clip_or_change_state(variable):
    policy = scaling(budget=0., update_every=1 if variable else None)
    if not variable:
        with pytest.raises(ValueError, match='added-mass budget'):
            make_step(mass_scaling=policy)
    else:
        step = make_step(mass_scaling=policy)
        before = step.state.snapshot()
        with pytest.raises(ValueError, match='added-mass budget'):
            step.run()
        assert_snapshot_equal(step.state.snapshot(), before)
        assert step.completed_steps == step.numerical_inertia.last_increment == 0


def test_body_commit_failure_rolls_back_variable_mass(monkeypatch):
    step = make_step(steps=7, mass_scaling=scaling(update_every=3))
    step.run(until_step=3)
    before, mass_before = step.state.snapshot(), step.numerical_inertia.snapshot()
    commit = step.material_residual.commit
    def fail():
        commit()
        raise RuntimeError('injected after commit')
    monkeypatch.setattr(step.material_residual, 'commit', fail)
    with pytest.raises(RuntimeError, match='injected'):
        step.run(until_step=4)
    assert_snapshot_equal(step.state.snapshot(), before)
    assert_snapshot_equal(step.numerical_inertia.snapshot(), mass_before)


def test_exact_operator_coverage_rejects_missing_contact():
    def incomplete(request):
        k = np.ones((len(request.physical_nodal_mass), 3))
        return time.MassScalingBounds({'body': k, 'hourglass': k})
    policy = time.TargetIncrementMassScaling(1e-4, incomplete, {'fixture': 'missing'}, 100.)
    with pytest.raises(ValueError, match='operator coverage'):
        contact_step(mass_scaling=policy)


@pytest.mark.parametrize('bad', ('two_components', 'wrong_nodes', 'nan', 'negative', 'boolean'))
def test_invalid_provider_layout_or_values_reject_before_prediction(bad):
    def invalid(request):
        k = np.ones((len(request.physical_nodal_mass), 3))
        if bad == 'two_components':
            k = k[:, :2]
        elif bad == 'wrong_nodes':
            k = k[:-1]
        elif bad == 'nan':
            k[0, 0] = np.nan
        elif bad == 'negative':
            k[0, 0] = -1
        else:
            k = k.astype(bool)
        return time.MassScalingBounds({name: k for name in request.stiffness_names})
    policy = time.TargetIncrementMassScaling(1e-4, invalid, {'fixture': bad}, 100., update_every=1)
    step = make_step(mass_scaling=policy)
    before = step.state.snapshot()
    with pytest.raises(ValueError):
        step.run()
    assert_snapshot_equal(step.state.snapshot(), before)
    assert step.completed_steps == step.numerical_inertia.last_increment == 0


def test_contact_and_viscosity_inventory():
    from agentfem import elements
    seen = []
    def bounds(request):
        seen.append((request.stiffness_names, request.damping_names))
        k = np.ones((len(request.physical_nodal_mass), 3))
        return time.MassScalingBounds({name: k for name in request.stiffness_names},
                                     {name: k for name in request.damping_names})
    policy = time.TargetIncrementMassScaling(1e-4, bounds, {'fixture': 'inventory'}, 100.)
    step = contact_step(bond=True, mass_scaling=policy,
                        bulk_viscosity=elements.BulkViscosityPolicy(1., 1., 1.))
    assert seen == [(('body', 'hourglass', 'cohesive', 'rigid_contact:contact_0',
                      'rigid_contact:contact_1'), ('bulk_viscosity',))]
    step.run()


@pytest.mark.parametrize('kwargs', (
    {'update_every': 0}, {'update_every': True}, {'update_every': 1.5},
    {'maximum_added_mass_ratio': -1}, {'maximum_added_mass_ratio': True},
    {'target_increment': False}, {'target_increment': np.nan},
    {'safety_factor': 1.}, {'safety_factor': 0.}, {'safety_factor': True},
))
def test_invalid_controls(kwargs):
    defaults = dict(target_increment=1e-4, bounds=lambda r: None,
                    identity={'test': 1}, maximum_added_mass_ratio=1.)
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        time.TargetIncrementMassScaling(**defaults)


def test_conflicting_controls_and_target_dt():
    with pytest.raises(ValueError, match='not both'):
        make_step(mass_scaling=scaling(), mass_scaling_factor=2.)
    with pytest.raises(ValueError, match='target increment'):
        make_step(dt=2e-4, mass_scaling=scaling())
    with pytest.raises(ValueError, match='added-mass budget'):
        make_step(mass_scaling=time.FixedMassScaling(100., 1.))


def test_provider_arrays_and_identity_are_snapshots():
    values = np.ones((8, 3))
    record = time.MassScalingBounds({'body': values})
    values[:] = 5
    np.testing.assert_array_equal(record.stiffness['body'], 1.)
    identity = {'law': {'revision': 'one'}}
    policy = time.TargetIncrementMassScaling(1e-4, lambda r: record, identity, 1.)
    identity['law']['revision'] = 'two'
    policy.identity['law']['revision'] = 'three'
    assert policy.summary()['bound_provider'] == {'law': {'revision': 'one'}}
    with pytest.raises(ValueError, match='Masked'):
        time.MassScalingBounds({'body': np.ma.array(values)})


def test_variable_mass_survives_substep_retry_and_terminal_rollback(tmp_path):
    from agentfem import constitutive
    from test_response_material_step import problem
    def build(terminal=False):
        model, u, element = problem()
        original = model.materials[0].item
        def update(request):
            if request.time_increment > 5.1e-5 or (terminal and request.time > 1.4e-4):
                raise constitutive.ExplicitMaterialCutbackRequest('substep fixture')
            return original.update(request)
        model.materials.clear()
        model.material(constitutive.explicit_user_material(state_schema=original.state_schema,
            update=update, identity={'test': 'mass-retry', 'terminal': terminal}, density=1.))
        return model.step(target=u, element_policy=element, dt=1e-4, steps=4,
            time_increment_ceiling=1e-3, stress_free_initial_state=True, progress=False,
            substep_policy=time.ExplicitSubstepPolicy(3, True), history_every=1,
            mass_scaling=scaling(update_every=1))
    whole = build()
    whole.run()
    assert len(whole._substeps.events) == 4
    partial = build()
    partial.run(until_step=2)
    path = partial.save_checkpoint(tmp_path / 'retry')
    resumed = build()
    resumed.load_checkpoint(path)
    resumed.run()
    assert_snapshot_equal(whole.numerical_inertia.snapshot(), resumed.numerical_inertia.snapshot())
    assert whole.history_records == resumed.history_records
    broken = build(terminal=True)
    broken.run(until_step=1)
    before = broken.numerical_inertia.snapshot()
    with pytest.raises(constitutive.ExplicitMaterialCutbackRequest):
        broken.run()
    assert_snapshot_equal(before, broken.numerical_inertia.snapshot())
    assert broken.completed_steps == 1
