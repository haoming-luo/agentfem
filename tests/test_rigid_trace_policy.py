# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from agentfem import boundary_models
from test_response_material_step import contact_step


def test_coalesced_uniform_tools_preserve_motion_response():
    plain = contact_step(contact_trace_policy=boundary_models.RigidContactTracePolicy("nodal_lumped"))
    compact = contact_step(contact_trace_policy=boundary_models.RigidContactTracePolicy("nodal_lumped", True))
    plain.run()
    compact.run()
    np.testing.assert_allclose(compact.state.u.value.x.array, plain.state.u.value.x.array, rtol=1e-12, atol=1e-15)


def test_guard_observes_all_displacements_and_rejects_invalid_motion():
    policy = boundary_models.RigidContactTracePolicy("nodal_lumped", True, (0, 0, 1), .01)
    step = contact_step(cylinder=True, contact_trace_policy=policy)
    step.state.u.value.x.array[2] = .02
    with pytest.raises(ValueError, match="budget exceeded"):
        step.residual.adapter.current_nodal_positions(step.state.u.value)
    with pytest.raises(ValueError, match="Tool motion invalidates"):
        contact_step(cylinder=True, contact_trace_policy=boundary_models.RigidContactTracePolicy(
            "nodal_lumped", True, (1, 0, 0), .01))


def test_reduction_does_not_accept_gauss_quadrature():
    with pytest.raises(ValueError, match="nodal_lumped"):
        boundary_models.RigidContactTracePolicy(coalesce_nodes=True)
