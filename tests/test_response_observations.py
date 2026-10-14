# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import json

import numpy as np
import pytest

from test_response_material_step import make_step, contact_step
from test_response_deformable_step import make_step as deformable_step


def test_named_material_arrays_are_detached_and_read_only():
    step = make_step(material_storage='owned_arrays')
    step.run(until_step=1)
    record = step.accepted_material_fields()
    clock = record['state_variables']['clock']
    assert clock.shape == (len(record['cell_keys']),)
    np.testing.assert_array_equal(clock, record['state'][:, 0])
    with pytest.raises(ValueError):
        clock.setflags(write=True)
    step.run()
    np.testing.assert_allclose(clock, 1e-4)


def test_no_contacts_and_unknown_names_are_distinct():
    step = make_step()
    step.run()
    assert step.accepted_contact_evidence() == ()
    with pytest.raises(KeyError):
        step.accepted_contact_evidence(name='unknown')
    with pytest.raises(ValueError):
        step.accepted_contact_evidence(name='')


@pytest.mark.parametrize('cylinder', (False, True))
def test_rigid_contact_observation_is_detached_and_restartable(cylinder, tmp_path):
    step = contact_step(cylinder=cylinder)
    step.run(until_step=2)
    records = step.accepted_contact_evidence()
    assert len(records) == 2
    json.dumps(records, allow_nan=False)
    for record in records:
        assert record['time'] == pytest.approx(step.completed_steps*step.dt)
        np.testing.assert_allclose(np.add(record['contact_force_on_structure'],
                                         record['contact_force_on_surface']), 0., atol=1e-14)
        assert record['moment_convention'] == 'surface_motion_generalized_force_not_tool_reaction'
    before = json.dumps(records, sort_keys=True)
    checkpoint = step.save_checkpoint(tmp_path / 'contacts')
    resumed = contact_step(cylinder=cylinder)
    resumed.load_checkpoint(checkpoint)
    assert json.dumps(resumed.accepted_contact_evidence(), sort_keys=True) == before
    records[0]['contact_force_on_structure'][0] = 999.
    assert json.dumps(step.accepted_contact_evidence(), sort_keys=True) == before
    assert step.accepted_contact_evidence(name=records[0]['name'])[0]['name'] == records[0]['name']


def test_deformable_contact_resultant_moment_and_coverage():
    step = deformable_step()
    step.run()
    record, = step.accepted_contact_evidence()
    assert record['active_point_count'] > 0
    assert record['maximum_penetration'] > 0
    assert record['active_reference_area'] > 0
    assert record['projected_point_count'] + record['certified_separated_point_count'] + record[
        'zero_gap_resolution_count'] == record['candidate_point_count']
    np.testing.assert_allclose(record['force_balance_residual'], 0., atol=1e-13)
    np.testing.assert_allclose(record['moment_balance_residual'], 0., atol=1e-13)
    # The secondary lower block is being pressed upward into the primary.
    assert record['secondary_force'][2] < 0
    json.dumps(record, allow_nan=False)


def test_observers_refuse_unaccepted_displacement():
    step = contact_step()
    step.run(until_step=1)
    step.state.u.value.x.array[0] += .01
    for observe in (step.accepted_contact_evidence, step.accepted_material_fields):
        with pytest.raises(RuntimeError, match='accepted displacement'):
            observe()
