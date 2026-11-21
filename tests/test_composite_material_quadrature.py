# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Work-conjugate nonlinear centre response, independent derivative evidence."""

import numpy as np
import pytest
from agentfem.elements.composite_tet import composite_tet11
from agentfem.elements._composite_material import CompositeMaterialBatch
from test_composite_tet_material import local_element, material


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("mean", [False, True])
def test_quadrature_recovers_elastic_operator_and_affine_patch(independent, mean):
    e = local_element()
    # Non-axis-aligned affine geometry exercises physical gradient transforms.
    from agentfem.elements.composite_tet import composite_tet10

    x = e.coordinates @ np.array([[1.2, 0.2, -0.1], [0.1, 0.8, 0.2], [0.2, -0.1, 1.1]])
    e = composite_tet11(x) if independent else composite_tet10(x)
    B, w = e.material_quadrature(mean_dilatation=mean)
    rng = np.random.default_rng(8)
    A = rng.normal(size=(6, 6))
    C = A.T @ A + np.eye(6)  # anisotropy as well as the J2 isotropic case
    K = np.einsum("q,qai,ab,qbj->ij", w, B, C, B)
    np.testing.assert_allclose(K, e.stiffness(C, mean_dilatation=mean), atol=1e-12)
    assert w.sum() == pytest.approx(e.volume)
    assert np.all(w > 0)
    eig = np.linalg.eigvalsh(K)
    assert np.count_nonzero(eig < eig[-1] * 1e-10) == 6
    F = np.array([[0.03, 0.02, -0.01], [-0.01, -0.02, 0.005], [0.02, 0.01, 0.04]])
    eng = np.array(
        [
            F[0, 0],
            F[1, 1],
            F[2, 2],
            F[0, 1] + F[1, 0],
            F[1, 2] + F[2, 1],
            F[0, 2] + F[2, 0],
        ]
    )
    np.testing.assert_allclose(
        B @ (e.coordinates @ F.T).ravel(), np.tile(eng, (len(w), 1)), atol=1e-14
    )


@pytest.mark.parametrize("mean", [False, True])
def test_centre_plastic_history_and_consistent_force_derivative(mean):
    e = composite_tet11(local_element().coordinates)
    m = material()
    b = CompositeMaterialBatch([e], m, mean_dilatation=mean)
    assert b.points_per_cell == 48
    # Pure internal mode: boundary nodes do not move. This must still yield.
    u = np.zeros((1, 33))
    u[0, 30:] = [0.02, -0.013, 0.007]
    f = b.evaluate(u, 0.1)
    assert np.linalg.norm(f[0, 30:]) > 0
    assert b.trial[1][:, 9].max() > 0
    assert np.max(np.abs(b.trial[1][:, 9] - b.trial[1][0, 9])) > 1e-5
    b.commit()
    snap = b.snapshot()
    v = -0.7 * u
    b.evaluate(v, 0.2)
    ref = []
    from agentfem import constitutive

    for i, eps in enumerate(b.trial[0]):
        old = constitutive.J2PlasticState(b.states[i, :9].reshape(3, 3), b.states[i, 9])
        ref.append(m.provider.material.update(eps, old))
    pairs = b.pairs
    C = np.array(
        [
            [[r.algorithmic_tangent[i, j, k, l] for k, l in pairs] for i, j in pairs]
            for r in ref
        ]
    )
    B = b.B[0]
    K = np.einsum("q,qai,qab,qbj->ij", b.weights[0], B, C, B)
    d = np.random.default_rng(12).normal(size=(1, 33))
    h = 1e-7
    fd = (b.evaluate(v + h * d, 0.2) - b.evaluate(v - h * d, 0.2)) / (2 * h)
    np.testing.assert_allclose(fd.ravel(), K @ d.ravel(), rtol=2e-7, atol=2e-8)
    b.evaluate(v, 0.2)
    np.testing.assert_allclose(
        b.trial[2], np.array([r.stress for r in ref]), atol=1e-13
    )
    assert np.all(b.trial[4] >= b.dissipation)
    b.rollback()
    assert b.snapshot() == snap


@pytest.mark.parametrize("mean", [False, True])
def test_external_nonlinear_provider_force_is_energy_gradient(mean):
    from agentfem.constitutive.small_strain_explicit import SmallStrainExplicitMaterial
    from agentfem.constitutive.small_strain_user_material import (
        MaterialParameterSchema,
        SmallStrainMaterialPointOutput,
        small_strain_tangent_convention,
    )
    from agentfem.constitutive.user_material import MaterialStateSchema

    class SmoothMaterial:
        name = "external_smooth_nonlinear_elastic"
        rate_independent = True
        state_schema = MaterialStateSchema("test.smooth.empty", ())
        parameter_schema = MaterialParameterSchema("test.smooth.parameters")
        tangent_convention = small_strain_tangent_convention(
            shear_convention="engineering"
        )

        def summary(self):
            return {"name": self.name, "strain_scale": 0.01}

        def update(self, point):
            e = point.strain_new
            v = np.array(
                [e[0, 0], e[1, 1], e[2, 2], 2 * e[0, 1], 2 * e[1, 2], 2 * e[0, 2]]
            )
            z = v / 0.01
            c = np.array([100, 110, 120, 40, 50, 60])
            sv = c * 0.01 * np.tanh(z)
            stress = sv[np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2]])]
            energy = np.sum(c * 0.01**2 * (np.logaddexp(z, -z) - np.log(2)))
            return SmallStrainMaterialPointOutput(
                stress,
                np.diag(c / np.cosh(z) ** 2),
                np.empty(0),
                self.state_schema,
                self.tangent_convention,
                stored_energy_density=energy,
                dissipation_density_increment=0,
            )

    m = SmallStrainExplicitMaterial(
        SmoothMaterial(), np.diag([100, 110, 120, 40, 50, 60]), 1
    )
    e = composite_tet11(local_element().coordinates)
    b = CompositeMaterialBatch([e], m, mean_dilatation=mean)
    rng = np.random.default_rng(44)
    u = rng.normal(size=(1, 33)) * 0.002
    d = rng.normal(size=(1, 33))
    force = b.evaluate(u, 0.1)

    def energy(v):
        b.evaluate(v, 0.1)
        return b.weights.ravel() @ b.trial[3]

    h = 1e-7
    derivative = (energy(u + h * d) - energy(u - h * d)) / (2 * h)
    assert derivative == pytest.approx(float(np.sum(force * d)), rel=2e-8, abs=1e-10)
    b.evaluate(u, 0.1)
    b.commit()
    assert b.states.shape == (48, 0)
    assert np.sum(b.dissipation) == 0
