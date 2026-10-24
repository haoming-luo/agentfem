# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
from agentfem.diagnostics import PerformanceLedger


def test_failed_phase_is_counted_without_masking_original_error():
    ledger = PerformanceLedger()
    with pytest.raises(RuntimeError, match='original'):
        with ledger.measure('attempted'):
            raise RuntimeError('original')
    assert ledger.counts == {'attempted': 1}
    assert ledger.seconds['attempted'] >= 0


def test_invalid_phase_name_rejects_before_doing_work():
    ledger = PerformanceLedger()
    with pytest.raises(ValueError, match='name cannot be empty'):
        with ledger.measure(' '):
            pytest.fail('Invalid timing request must reject before entering work')
    assert ledger.counts == {}


def test_response_body_phases_share_existing_ledger():
    from test_response_material_step import contact_step
    step = contact_step()
    step.run()
    body = step.material_residual
    assert body.performance is step.performance
    assert body.internal.performance is step.performance
    stages = step.performance.summary()['stages']
    for name in ('body_cell_gather', 'body_kinematics_admission',
                 'body_material_and_force', 'body_force_scatter',
                 'body_residual_inclusive'):
        assert stages[name]['calls'] == step.completed_steps
        assert 0 <= stages[name]['seconds'] <= stages['run_wall']['seconds']
