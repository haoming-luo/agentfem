# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from test_response_material_step import contact_step, problem
from test_explicit_hex_execution import reference_kernel
from agentfem import constitutive
from agentfem.constitutive.user_material import MaterialStateSchema, MaterialStateVariable


@pytest.mark.parametrize('kernel', (False, True))
def test_owned_matches_fields_with_restart_and_observers(tmp_path, kernel):
    options = dict(execution_kernel=reference_kernel([])) if kernel else {}
    reference = contact_step(bond=True, **options)
    reference.run()
    owner = contact_step(bond=True, material_storage='owned_arrays', **options)
    owner.run(until_step=2)
    snapshot = owner.accepted_material_fields()
    saved = snapshot['state'].copy()
    path = owner.save_checkpoint(tmp_path/'owner')
    continued = contact_step(bond=True, material_storage='owned_arrays', **options)
    continued.load_checkpoint(path)
    continued.run()
    np.testing.assert_array_equal(continued.state.u.value.x.array, reference.state.u.value.x.array)
    for key in ('state','cauchy_stress','deformation_gradient','cell_keys'):
        np.testing.assert_array_equal(continued.accepted_material_fields()[key], reference.accepted_material_fields()[key])
    np.testing.assert_array_equal(snapshot['state'], saved)
    assert not snapshot['state'].flags.writeable


def test_owned_kernel_failure_restores_accepted_owner():
    step = contact_step(material_storage='owned_arrays', execution_kernel=reference_kernel([], fail=True))
    step.run(until_step=1)
    before = step.residual.snapshot()
    with pytest.raises(RuntimeError, match='injected kernel failure'):
        step.run()
    assert step.residual.snapshot() == before
    assert step.material_residual.internal.response._owned.trial is None


def test_general_schema_not_one_fixed_history_block():
    model, u, policy = problem()
    schema = MaterialStateSchema('two_named_variables', (
        MaterialStateVariable('clock'), MaterialStateVariable('tensor_memory', shape=(2,3))))
    def update(request):
        return constitutive.ExplicitMaterialBatchOutput(np.zeros((request.point_count,3,3)),
            request.state_old+request.time_increment, schema, np.zeros(request.point_count))
    model.materials.clear()
    model.material(constitutive.explicit_user_material(state_schema=schema, update=update,
        identity={'test':'multi-variable-owned'}, density=1.))
    step = model.step(target=u, element_policy=policy, dt=1e-4, steps=4,
        time_increment_ceiling=1e-3, stress_free_initial_state=True, progress=False,
        material_storage='owned_arrays')
    step.run()
    np.testing.assert_allclose(step.accepted_material_fields()['state'], 4e-4, rtol=1e-15)


def test_corrupt_restore_does_not_replace_accepted_owner():
    import copy
    step = contact_step(material_storage='owned_arrays')
    step.run(until_step=2)
    body = step.material_residual
    before = body.snapshot()
    invalid = copy.deepcopy(before)
    name = next(k for k in invalid['fields'] if k.startswith('committed_state_'))
    invalid['fields'][name][0] += 1
    with pytest.raises(ValueError, match='unaccepted material history'):
        body.restore(invalid)
    assert body.snapshot() == before
    assert body.internal.response._owned.trial is None


@pytest.mark.parametrize('storage', ['fields', 'owned_arrays'])
def test_body_accepted_records_share_bytes_not_mutable_metadata(tmp_path, storage):
    step = contact_step(bond=True, material_storage=storage)
    step.run(until_step=2)
    body = step.material_residual
    first, second = body.transaction_snapshot(), body.transaction_snapshot()
    owners = {'gradient': body.accepted_gradient, 'accepted_force': body.accepted_force,
              'accepted_displacement': body.accepted_displacement}
    saved = {}
    for key, owner in owners.items():
        a, b = first[key], second[key]
        saved[key] = a.copy()
        assert a is not b and a is not owner
        assert np.shares_memory(a, b) and np.shares_memory(a, owner)
        with pytest.raises(ValueError):
            a.flags.writeable = True
        with pytest.raises(ValueError):
            a.flat[0] = 42
        old_shape = owner.shape
        a.shape = (a.size,)
        assert owner.shape == b.shape == old_shape
    checkpoint = step.save_checkpoint(tmp_path/'accepted')
    step.run()
    for key in saved:
        np.testing.assert_array_equal(second[key], saved[key])
    resumed = contact_step(bond=True, material_storage=storage)
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    assert resumed.residual.snapshot() == step.residual.snapshot()


@pytest.mark.parametrize('storage', ['fields', 'owned_arrays'])
def test_failed_trial_does_not_invalidate_shared_body_records(storage):
    step = contact_step(material_storage=storage, execution_kernel=reference_kernel([], fail=True))
    step.run(until_step=1)
    body = step.material_residual
    record = body.transaction_snapshot()
    saved = body.snapshot()
    with pytest.raises(RuntimeError, match='injected kernel failure'):
        step.run()
    assert body.snapshot() == saved
    after = body.transaction_snapshot()
    for key in ('gradient', 'accepted_force', 'accepted_displacement'):
        np.testing.assert_array_equal(record[key], after[key])
        assert np.shares_memory(record[key], after[key])
