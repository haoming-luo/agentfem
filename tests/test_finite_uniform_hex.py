# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent hyperelastic oracle for a private finite-kinematics contribution."""

import basix
import numpy as np
import pytest

from agentfem.elements._finite_uniform_hex import FiniteUniformHexBatch


def _operator(distorted=False, count=1, chunk_size=2):
    x = np.tile(basix.cell.geometry(basix.CellType.hexahedron), (count, 1, 1))
    if distorted:
        x += np.random.default_rng(72).uniform(-0.07, 0.07, x.shape)
    return FiniteUniformHexBatch(
        x, density=2, hourglass_modulus=30, hourglass_scale=0.1, chunk_size=chunk_size
    )


def _material(f):
    # Independent compressible Neo-Hookean reference, per reference volume.
    mu, lam = 30, 40
    inverse = np.linalg.inv(f).swapaxes(1, 2)
    logj = np.log(np.linalg.det(f))
    p = mu * f + (lam * logj - mu)[:, None, None] * inverse
    w = 0.5 * mu * (np.sum(f * f, axis=(1, 2)) - 3) - mu * logj + 0.5 * lam * logj**2
    a = mu * np.einsum("ik,JL->iJkL", np.eye(3), np.eye(3))[None]
    a = a + lam * np.einsum("ciJ,ckL->ciJkL", inverse, inverse)
    a -= (lam * logj - mu)[:, None, None, None, None] * np.einsum(
        "ciL,ckJ->ciJkL", inverse, inverse
    )
    return p, w, a


def _response(op, u):
    p, w, _ = _material(op.deformation_gradient(u))
    return op.response(u, first_piola=p, stored_energy_density=w)


@pytest.mark.parametrize("distorted", [False, True])
def test_finite_affine_patch_and_large_rigid_rotations(distorted):
    op = _operator(distorted)
    x = op.coordinates
    f = np.array([[1.2, 0.3, 0.1], [0.0, 0.8, 0.2], [0.0, 0.0, 1.1]])
    u = x @ (f - np.eye(3)).T + [0.2, -0.3, 0.5]
    np.testing.assert_allclose(op.deformation_gradient(u), f[None], atol=1e-13)
    assert _response(op, u).hourglass_energy[0] < 1e-26
    for angle in (0.5, 1.8, 3.14, 5.5):
        q = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        rigid = x @ q.T + [0.5, 0.4, -0.7] - x
        response = _response(op, rigid)
        np.testing.assert_allclose(response.internal_force, 0, atol=1e-12)
        np.testing.assert_allclose(response.physical_energy, 0, atol=1e-12)
        np.testing.assert_allclose(response.hourglass_energy, 0, atol=1e-25)


def test_superposed_rotation_preserves_energy_and_rotates_force_with_hourglass():
    op = _operator(True, count=3)
    rng = np.random.default_rng(111)
    u = rng.uniform(-0.04, 0.04, op.coordinates.shape)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    assert np.linalg.det(q) > 0
    response = _response(op, u)
    rotated = _response(
        op, (op.coordinates + u) @ q.T + [0.2, 0.5, -0.3] - op.coordinates
    )
    assert response.hourglass_energy.sum() > 1e-5
    np.testing.assert_allclose(
        rotated.physical_energy, response.physical_energy, atol=1e-12
    )
    np.testing.assert_allclose(
        rotated.hourglass_energy, response.hourglass_energy, atol=1e-13
    )
    np.testing.assert_allclose(
        rotated.internal_force, response.internal_force @ q.T, atol=1e-12
    )
    np.testing.assert_allclose(response.internal_force.sum(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(
        np.cross(op.coordinates + u, response.internal_force).sum(axis=1), 0, atol=1e-12
    )


def test_finite_force_energy_gradient_and_material_tangent_action():
    op = _operator(True, count=3)
    rng = np.random.default_rng(15)
    u = rng.uniform(-0.02, 0.02, op.coordinates.shape)
    direction = rng.normal(size=u.shape)
    p, w, a = _material(op.deformation_gradient(u))
    base = op.response(u, first_piola=p, stored_energy_density=w)
    h = 1e-6
    plus, minus = _response(op, u + h * direction), _response(op, u - h * direction)
    energy_derivative = (
        (plus.physical_energy + plus.hourglass_energy)
        - (minus.physical_energy + minus.hourglass_energy)
    ) / (2 * h)
    np.testing.assert_allclose(
        energy_derivative,
        np.sum(base.internal_force * direction, axis=(1, 2)),
        rtol=2e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        op.tangent_action(direction, first_piola_tangent=a),
        (plus.internal_force - minus.internal_force) / (2 * h),
        rtol=2e-8,
        atol=1e-8,
    )


def test_finite_batch_chunking_and_inverted_current_geometry():
    a, b = (
        _operator(True, count=4, chunk_size=1),
        _operator(True, count=4, chunk_size=4),
    )
    u = np.random.default_rng(18).uniform(-0.01, 0.01, a.coordinates.shape)
    np.testing.assert_allclose(
        _response(a, u).internal_force, _response(b, u).internal_force
    )
    u[:, :, 0] = -2 * a.coordinates[:, :, 0]
    with pytest.raises(ValueError, match="Jacobian"):
        a.deformation_gradient(u)
    with pytest.raises(ValueError, match="first Piola"):
        a.response(
            np.zeros_like(u),
            first_piola=np.zeros((4, 6)),
            stored_energy_density=np.zeros(4),
        )


def test_finite_operator_does_not_admit_public_finite_step():
    from agentfem import elements

    with pytest.raises(TypeError):
        elements.uniform_strain_hex8(
            hourglass_modulus=30, hourglass_scale=0.1, finite_strain=True
        )


def test_existing_j2_batch_protocol_supplies_work_conjugate_element_tangent():
    from agentfem import constitutive

    op = _operator(True, count=2)
    law = constitutive.finite_strain_j2_logarithmic(
        young=210000, poisson=0.3, yield_stress=250, hardening_modulus=1000
    )
    old = law.state_schema.initial_state()
    old_copy = old.copy()

    def material_response(u):
        f = op.deformation_gradient(u)
        request = constitutive.MaterialPointBatchInput(
            tuple(
                constitutive.MaterialPointInput(
                    deformation_gradient_old=np.eye(3),
                    deformation_gradient_new=gradient,
                    time=0,
                    time_increment=0.1,
                    properties=[],
                    state_old=old,
                    state_schema=law.state_schema,
                )
                for gradient in f
            )
        )
        responses = law.update_batch(request).responses
        sigma = np.stack([response.cauchy_stress for response in responses])
        p = np.linalg.det(f)[:, None, None] * (sigma @ np.linalg.inv(f).swapaxes(1, 2))
        energy = np.array([response.strain_energy_density for response in responses])
        tangent = np.stack(
            [response.consistent_tangent for response in responses]
        ).reshape(-1, 3, 3, 3, 3)
        return op.response(u, first_piola=p, stored_energy_density=energy), tangent

    gradient = np.diag([1.06, 1 / np.sqrt(1.06), 1 / np.sqrt(1.06)])
    u = op.coordinates @ (gradient - np.eye(3)).T
    base, tangent = material_response(u)
    direction = np.random.default_rng(71).normal(size=u.shape)
    h = 1e-7
    plus, _ = material_response(u + h * direction)
    minus, _ = material_response(u - h * direction)
    np.testing.assert_allclose(
        op.tangent_action(direction, first_piola_tangent=tangent),
        (plus.internal_force - minus.internal_force) / (2 * h),
        rtol=2e-6,
        atol=2e-4,
    )
    np.testing.assert_array_equal(old, old_copy)
    assert np.linalg.norm(base.internal_force) > 0
