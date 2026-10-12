# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from threading import Event, get_ident
import numpy as np
import pytest
from agentfem.mechanics._deformable_contact_explicit import DeformableQ1ContactResidual


@pytest.mark.parametrize('body_failure,geometry_failure', [(False,False),(True,False),(False,True),(True,True)])
def test_owner_thread_join_and_failure_precedence(body_failure, geometry_failure):
    contact = object.__new__(DeformableQ1ContactResidual)
    owner = get_ident()
    started, body_started, finished = Event(), Event(), Event()
    calls = []
    vector = SimpleNamespace(destroy=lambda: calls.append('destroy'))
    contact._values = lambda: (np.zeros((2,3)), np.zeros((3,3)))
    def geometry(values):
        assert get_ident() != owner
        assert all(not v.flags.writeable for v in values)
        started.set()
        assert body_started.wait(2)
        finished.set()
        if geometry_failure: raise ValueError('geometry')
        return 'response'
    def body():
        assert get_ident() == owner
        assert started.wait(2)
        body_started.set()
        if body_failure: raise ValueError('body')
        return vector
    def rollback():
        assert get_ident() == owner and finished.is_set()
        contact.trial = None
        calls.append('rollback')
    contact.base = SimpleNamespace(assemble_vector=body)
    contact._evaluate_contact = geometry
    contact.rollback = rollback
    contact._add = lambda v,r: calls.append(('add',get_ident(),r))
    with contact.overlap_execution():
        with pytest.raises(RuntimeError, match='already active'):
            with contact.overlap_execution(): pass
        if body_failure or geometry_failure:
            with pytest.raises(ValueError, match='geometry' if geometry_failure else 'body'):
                contact.assemble_vector()
            assert contact.trial is None
            assert ('destroy' in calls) == (not body_failure)
        else:
            assert contact.assemble_vector() is vector
            assert calls == [('add',owner,'response')]
        assert not contact._geometry_busy
    assert contact._geometry_executor is None


def test_real_contact_same_path_and_restart(tmp_path):
    from test_deformable_contact_explicit import make_step
    serial, overlap = make_step(), make_step()
    serial.run()
    with overlap.residual.overlap_execution(): overlap.run(until_step=7)
    checkpoint = overlap.save_checkpoint(tmp_path/'partial')
    resumed = make_step()
    resumed.load_checkpoint(checkpoint)
    with resumed.residual.overlap_execution(): resumed.run()
    np.testing.assert_array_equal(serial.state.u.value.x.array, resumed.state.u.value.x.array)
    assert serial.history_records == resumed.history_records


def test_step_rejects_overlap_without_deformable_pair():
    from test_response_material_step import make_step
    with pytest.raises(ValueError, match='requires a deformable'):
        make_step(contact_execution='overlap')
    with pytest.raises(ValueError, match='serial or overlap'):
        make_step(contact_execution='anything')


def test_public_step_scopes_worker_and_result(tmp_path):
    from test_response_deformable_step import make_step
    serial = make_step()
    serial.run()
    partial = make_step(contact_execution='overlap')
    partial.run(until_step=2)
    assert partial.deformable_residual._geometry_executor is None
    saved = partial.save_checkpoint(tmp_path/'public')
    resumed = make_step(contact_execution='overlap')
    resumed.load_checkpoint(saved)
    assert resumed.solve_result() is not None
    assert resumed.deformable_residual._geometry_executor is None
    assert resumed.summary()['contact_execution'] == 'overlap'
    np.testing.assert_array_equal(serial.state.u.value.x.array, resumed.state.u.value.x.array)
    assert serial.history_records == resumed.history_records
