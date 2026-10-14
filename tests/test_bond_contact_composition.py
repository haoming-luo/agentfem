# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent bond/contact contributions, not a vendor-equivalence claim."""
import numpy as np
import pytest

from agentfem import interfaces
from agentfem._deformable_q1_contact import DeformableQ1Penalty
from agentfem._interface_pairing import FixedReferenceCohesiveAssembler


def operators():
    primary = interfaces.reference_trace(
        np.array([[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
        np.array([[0, 1, 2, 3]]), topology='quadrilateral', geometry='bilinear', tolerance=1e-10)
    secondary = interfaces.reference_trace(
        np.array([[.3, .3, 0], [.7, .3, 0], [.7, .7, 0], [.3, .7, 0]]),
        np.array([[0, 1, 2, 3]]), topology='quadrilateral', geometry='bilinear', tolerance=1e-10)
    pairing = interfaces.initial_nodal_bond(secondary, primary,
        eligible_nodes=np.ones(4, dtype=bool), tolerance=1e-10)
    bond = FixedReferenceCohesiveAssembler(pairing,
        interfaces.elastic_cohesive(normal_stiffness=100., tangential_stiffness=100.),
        tangential='mixed')
    contact = DeformableQ1Penalty(secondary, primary,
        candidate_nodes=np.ones(4, dtype=bool), penalty=300., search_radius=.5)
    return bond, contact


@pytest.mark.parametrize('translation', ((0, 0, .02), (0, 0, -.02), (.01, 0, 0), (.01, 0, -.02)))
def test_independent_stiffness_force_energy_and_current_moment(translation):
    bond, contact = operators()
    us = np.tile(translation, (4, 1))
    up = np.zeros((4, 3))
    b, c = bond.evaluate(us, up), contact.evaluate(us, up)
    rs, rp = b.negative_residual + c.secondary_residual, b.positive_residual + c.primary_residual
    area = .4**2
    expected = 100.*np.array(translation)
    expected[2] += 300.*min(translation[2], 0)
    np.testing.assert_allclose(rs.sum(axis=0), area*expected, atol=1e-13)
    np.testing.assert_allclose(rs.sum(axis=0)+rp.sum(axis=0), 0., atol=1e-13)
    np.testing.assert_allclose(np.cross(contact.secondary.vertices+us, rs).sum(axis=0)
        + np.cross(contact.primary.vertices+up, rp).sum(axis=0), 0., atol=1e-13)
    expected_energy = .5*area*(100.*np.dot(translation, translation)
                              + 300.*min(translation[2], 0)**2)
    assert b.stored_energy+c.potential_energy == pytest.approx(expected_energy, abs=1e-14)
    if translation[2] != 0:
        rng = np.random.default_rng(718)
        ds, dp = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
        def energy(a, p):
            return bond.evaluate(a, p).stored_energy + contact.evaluate(a, p).potential_energy
        h = 1e-7
        fd = (energy(us+h*ds, up+h*dp)-energy(us-h*ds, up-h*dp))/(2*h)
        assert fd == pytest.approx(np.sum(rs*ds)+np.sum(rp*dp), rel=1e-7, abs=1e-10)


def test_open_compress_shear_unload_reclose_is_reversible():
    bond, contact = operators()
    path = np.array([[0, 0, 0], [0, 0, .02], [0, 0, 0], [0, 0, -.02],
                     [.01, 0, -.02], [.01, 0, 0], [0, 0, 0], [0, 0, -.01]])
    up, old_u, old_r = np.zeros((4, 3)), np.zeros((4, 3)), np.zeros((4, 3))
    work = 0.
    for position in path:
        us = np.tile(position, (4, 1))
        b, c = bond.begin(us, up), contact.evaluate(us, up)
        rs = b.negative_residual + c.secondary_residual
        work += .5*np.sum((old_r+rs)*(us-old_u))
        assert work == pytest.approx(b.stored_energy+c.potential_energy, abs=1e-14)
        assert b.dissipated_energy == 0.
        assert np.all(b.damage == 0.)
        bond.commit()
        old_u, old_r = us, rs
