# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from agentfem._q1_contact_curvature import interior_distance_curvature, contact_row_enclosure
from test_deformable_q1_contact import setup


@pytest.mark.parametrize("warp", [0., .3])
def test_analytic_reduced_curvature_against_independent_force_difference(warp):
    op, us, up = setup(warp)
    us[:, 2] = -.08
    response = op.evaluate(us, up)
    analytic = np.zeros((24, 24))
    upper = np.zeros_like(analytic)
    for point, nodes in enumerate(response.primary_nodes):
        w = response.primary_weights[point]
        uv = [w[1]+w[2], w[2]+w[3]]
        local = interior_distance_curvature(op.secondary.vertices[point]+us[point],
            op.primary.vertices[nodes]+up[nodes], uv,
            stiffness=100*op.weights[point])
        ids = np.r_[np.arange(point*3, point*3+3), (12+nodes[:,None]*3+np.arange(3)).ravel()]
        analytic[np.ix_(ids, ids)] += local.tangent
        upper[np.ix_(ids, ids)] += local.fixed_coordinate_upper
    z = np.r_[us.ravel(), up.ravel()]
    numeric = np.empty_like(analytic)
    for j in range(len(z)):
        values = []
        for sign in (-1, 1):
            trial = z.copy(); trial[j] += sign*2e-7
            r = op.evaluate(trial[:12].reshape(4,3), trial[12:].reshape(4,3))
            values.append(np.r_[r.secondary_residual.ravel(), r.primary_residual.ravel()])
        numeric[:,j] = (values[1]-values[0])/4e-7
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-8)
    assert np.linalg.eigvalsh(upper-analytic).min() >= -1e-12
    # Geometric curvature must not be silently presented as positive stiffness.
    assert np.linalg.eigvalsh(analytic).min() < -1e-4
    rows = contact_row_enclosure(op, us, up)
    assert rows.active_points == 4
    assert not rows.summary()["whole_path_stability"]
    assert np.linalg.eigvalsh(np.diag(rows.positive_rows.ravel())-analytic).min() >= -1e-12
    assert np.linalg.eigvalsh(analytic+np.diag(rows.negative_rows.ravel())).min() >= -1e-12
    for axis in range(3):
        shift = np.zeros((8,3)); shift[:,axis] = 1
        np.testing.assert_allclose(analytic @ shift.ravel(), 0, atol=1e-13)


def test_rejects_nonstationary_edge_and_degenerate_projections():
    x = np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]])
    for q, nodes, uv in [([.2,.3,-.1],x,[.5,.5]),
                          ([0,.5,-.1],x,[0,.5]),
                          ([0,0,-.1],np.zeros((4,3)),[.5,.5])]:
        with pytest.raises(ValueError):
            interior_distance_curvature(q,nodes,uv,stiffness=10)


def test_separation_zero_and_activation_rejected():
    op, us, up = setup()
    result = contact_row_enclosure(op, us, up)
    assert result.active_points == 0
    assert not result.positive_rows.any() and not result.negative_rows.any()
    us[:,2] = -.02
    with pytest.raises(ValueError, match="activation"):
        contact_row_enclosure(op, us, up)


def test_rotated_and_scaled_local_curvature():
    x = np.array([[0.,0,0],[1,0,0],[1,1,.2],[0,1,0]])
    uv = np.array([.3,.4])
    u,v = uv
    n = np.array([(1-u)*(1-v),u*(1-v),u*v,(1-u)*v])
    ds = np.array([[-(1-v),1-v,v,-v],[-(1-u),-u,u,1-u]]) @ x
    normal = np.cross(*ds); normal /= np.linalg.norm(normal)
    q = n@x-.03*normal
    original = interior_distance_curvature(q,x,uv,stiffness=3.)
    rotation = np.array([[0.,0,1],[1,0,0],[0,1,0]])
    transformation = np.kron(np.eye(5),rotation)
    for scale in (.01,1.,100.):
        rotated = interior_distance_curvature(scale*(rotation@q),scale*(x@rotation.T),uv,stiffness=3.)
        np.testing.assert_allclose(rotated.tangent,transformation@original.tangent@transformation.T,atol=2e-14)
