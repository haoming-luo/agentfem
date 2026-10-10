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


def test_reference_spectral_cache_is_compact_immutable_and_material_independent():
    operator = _operator(True, count=5)
    assert operator._mass_gradient_gram.shape == (5, 3, 3)
    assert operator._hourglass_spectral_bound.shape == (5,)
    assert not operator._mass_gradient_gram.flags.writeable
    assert not operator._hourglass_spectral_bound.flags.writeable
    assert (
        operator._mass_gradient_gram.nbytes + operator._hourglass_spectral_bound.nbytes
        == 5 * 10 * 8
    )


def test_underflowed_lumped_mass_is_rejected_before_spectral_division():
    with pytest.raises(ValueError, match="coefficients"):
        FiniteUniformHexBatch(
            basix.cell.geometry(basix.CellType.hexahedron)[None],
            density=np.nextafter(0.0, 1.0),
            hourglass_modulus=30,
            hourglass_scale=0.1,
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


def test_state_dependent_spectral_screen_bounds_full_nodal_tangent():
    op = _operator(True)
    bounds = []
    for stretch in (1.0, 1.1, 1.3):
        f = stretch * np.eye(3)[None]
        _, _, tangent = _material(f)
        bound = op.tangent_spectral_bound(first_piola_tangent=tangent)
        columns = []
        for basis in np.eye(24):
            columns.append(
                op.tangent_action(
                    basis.reshape(1, 8, 3), first_piola_tangent=tangent
                ).ravel()
            )
        matrix = np.column_stack(columns)
        mass = np.repeat(op.lumped_mass[0], 3)
        exact = np.linalg.eigvalsh(
            matrix / np.sqrt(mass[:, None] * mass[None, :])
        ).max()
        assert bound >= exact * (1 - 1e-12)
        bounds.append(bound)
    assert np.ptp(bounds) > 1


@pytest.mark.parametrize("invalid", ["negative", "asymmetric", "nan"])
def test_spectral_screen_does_not_hide_unsupported_material_tangent(invalid):
    op = _operator()
    tangent = np.eye(9).reshape(1, 3, 3, 3, 3)
    matrix = tangent.reshape(9, 9)
    if invalid == "negative":
        matrix[0, 0] = -1
    elif invalid == "asymmetric":
        matrix[0, 1] = 1
    else:
        matrix[0, 0] = np.nan
    with pytest.raises(ValueError):
        op.tangent_spectral_bound(first_piola_tangent=tangent)


def test_signed_tangent_enclosure_retains_negative_curvature_separately():
    op = _operator(True, count=2)
    rng = np.random.default_rng(78)
    q, _ = np.linalg.qr(rng.normal(size=(9, 9)))
    a = q @ np.diag([-10, -4, -1, 2, 3, 4, 5, 6, 7]) @ q.T
    tangent = np.tile(a.reshape(1, 3, 3, 3, 3), (2, 1, 1, 1, 1))
    report = op.tangent_spectral_report(first_piola_tangent=tangent)
    assert report.negative_material_curvature_cells == 2
    assert report.negative_eigenvalue_magnitude_bound > 0
    for cell in range(2):
        columns = []
        for basis in np.eye(24):
            direction = np.zeros((2, 8, 3))
            direction[cell] = basis.reshape(8, 3)
            columns.append(
                op.tangent_action(direction, first_piola_tangent=tangent)[cell].ravel()
            )
        mass = np.repeat(op.lumped_mass[cell], 3)
        eigenvalues = np.linalg.eigvalsh(
            np.column_stack(columns) / np.sqrt(mass[:, None] * mass[None, :])
        )
        assert eigenvalues[0] < 0
        assert eigenvalues[-1] <= report.positive_eigenvalue_upper_bound * (1 + 1e-12)
        assert eigenvalues[0] >= -report.negative_eigenvalue_magnitude_bound * (
            1 + 1e-12
        )


def test_spherical_gram_shortcut_matches_general_signed_bound():
    operator = _operator(count=7, chunk_size=3)
    assert np.all(operator._isotropic_mass_gradient_bound > 0)
    rng = np.random.default_rng(507)
    matrices = rng.normal(size=(7, 9, 9))
    matrices += matrices.swapaxes(1, 2)
    tangent = matrices.reshape(7, 3, 3, 3, 3)
    fast = operator.tangent_spectral_report(first_piola_tangent=tangent)
    operator._isotropic_mass_gradient_bound = np.zeros(7)
    full = operator.tangent_spectral_report(first_piola_tangent=tangent)
    assert (
        fast.negative_material_curvature_cells == full.negative_material_curvature_cells
    )
    assert fast.positive_eigenvalue_upper_bound == pytest.approx(
        full.positive_eigenvalue_upper_bound, rel=1e-13
    )
    assert fast.negative_eigenvalue_magnitude_bound == pytest.approx(
        full.negative_eigenvalue_magnitude_bound, rel=1e-13
    )


def test_spherical_shortcut_and_general_geometry_mix_in_one_batch():
    regular, distorted = _operator(count=2), _operator(True, count=2)
    combined = FiniteUniformHexBatch(
        np.concatenate((regular.coordinates, distorted.coordinates)),
        density=2,
        hourglass_modulus=30,
        hourglass_scale=0.1,
        chunk_size=3,
    )
    np.testing.assert_array_equal(
        combined._isotropic_mass_gradient_bound > 0, [True, True, False, False]
    )
    tangent = np.tile(np.diag([-2, -1, 0, 1, 2, 3, 4, 5, 6]), (4, 1, 1)).reshape(
        4, 3, 3, 3, 3
    )
    fast = combined.tangent_spectral_report(first_piola_tangent=tangent)
    combined._isotropic_mass_gradient_bound = np.zeros(4)
    full = combined.tangent_spectral_report(first_piola_tangent=tangent)
    assert fast.negative_material_curvature_cells == 4
    assert fast.positive_eigenvalue_upper_bound == pytest.approx(
        full.positive_eigenvalue_upper_bound, rel=1e-13
    )
    assert fast.negative_eigenvalue_magnitude_bound == pytest.approx(
        full.negative_eigenvalue_magnitude_bound, rel=1e-13
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
