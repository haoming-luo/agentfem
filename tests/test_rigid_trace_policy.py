# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from agentfem import boundary_models
from test_response_material_step import contact_step


def test_shared_boundary_reference_is_prepared_once_per_step(monkeypatch):
    from test_response_material_step import problem
    from agentfem import mesh
    original = boundary_models.dolfinx_boundary_region_contact_trace
    calls = []
    def traced(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(boundary_models, 'dolfinx_boundary_region_contact_trace', traced)
    for unused in range(2):
        model, u, policy = problem()
        region = mesh.boundary(u.value.function_space.mesh, lambda x: np.isclose(x[0], 0))
        for index in range(2):
            surface = boundary_models.rigid_plane(point=(-1-index,0,0), normal=(1,0,0))
            pair = boundary_models.rigid_contact_pair(region,
                boundary_models.rigid_body(surface), penalty=100, name=f'tool_{index}')
            model.add_boundary_model(pair)
        step = model.step(target=u, element_policy=policy, dt=1e-4, steps=2,
            time_increment_ceiling=1e-3, stress_free_initial_state=True, progress=False)
        step.run()
        assert len(step.residual.contact_energy_evidence()) == 2
    assert len(calls) == 2  # No global cross-Step geometry cache.


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
