# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""External reference orthotropic hyperelastic provider, NOT the tester's VUMAT.

Reference energy W=mu E:E + lambda/2 tr(E)^2 + sum(k_i (a_i.E.a_i)^2)/2.
E=(F.T F-I)/2; material axes are fixed in the reference configuration.
This is a limited St-Venant/Kirchhoff family, not large-compression engineering
validation or an automatically selected replacement for a user's material.
"""

import numpy as np
import pytest

from agentfem.constitutive.material_array_batch import MaterialPointArrayBatchOutput
from test_finite_hex_external_material import ReferenceNeoHookean
from test_finite_hex_bond_contact import prepare


class ReferenceOrthotropic(ReferenceNeoHookean):
    name = "external_reference_orthotropic_green_strain"

    def __init__(self, angle=0.3):
        self.angle = angle
        c, s = np.cos(angle), np.sin(angle)
        axes = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        eye = np.eye(3)
        self.elasticity = (
            40 * np.einsum("ij,kl->ijkl", eye, eye)
            + 30 * (np.einsum("ik,jl->ijkl", eye, eye) + np.einsum("il,jk->ijkl", eye, eye))
            + np.einsum("a,ai,aj,ak,al->ijkl", [10, 40, 90], axes, axes, axes, axes)
        )

    def summary(self):
        return {**super().summary(), "angle": self.angle,
                "reference_elasticity": self.elasticity.tolist(),
                "strain_measure": "green_lagrange", "scope": "test_reference_only"}

    def _response(self, gradient, state_old):
        strain = (gradient.swapaxes(1, 2) @ gradient - np.eye(3)) / 2
        second = np.einsum("ijkl,nkl->nij", self.elasticity, strain)
        piola = gradient @ second
        energy = np.einsum("nij,nij->n", strain, second) / 2
        tangent = (np.einsum("ik,blj->bijkl", np.eye(3), second)
                   + np.einsum("bim,mjlr,bkr->bijkl", gradient, self.elasticity, gradient))
        return MaterialPointArrayBatchOutput(
            cauchy_stress=(piola @ gradient.swapaxes(1, 2)) / np.linalg.det(gradient)[:, None, None],
            consistent_tangent=tangent.reshape(-1, 9, 9),
            state_new=np.maximum(state_old, energy[:, None]), strain_energy_density=energy,
            dissipation_density_increment=np.zeros(len(gradient)),
            tangent_convention=self.tangent_convention, state_schema=self.state_schema,
        )


def test_external_orthotropic_energy_tangent_and_objectivity():
    law = ReferenceOrthotropic()
    f = np.array([[[1.01, 0.02, 0], [0.01, 0.99, -0.01], [0, 0.01, 1.02]]])
    state = np.zeros((1, 1))

    def response(g):
        value = law._response(g, state)
        p = np.linalg.det(g)[:, None, None] * value.cauchy_stress @ np.linalg.inv(g).swapaxes(1, 2)
        return p, value

    p, out = response(f)
    for index in range(9):
        perturb = np.zeros_like(f)
        perturb.reshape(-1)[index] = 1e-6
        pp, plus = response(f + perturb)
        pm, minus = response(f - perturb)
        np.testing.assert_allclose((pp - pm).reshape(-1) / 2e-6,
                                   out.consistent_tangent[0, :, index], atol=3e-8)
        assert float((plus.strain_energy_density - minus.strain_energy_density)[0] / 2e-6) == pytest.approx(p.reshape(-1)[index], abs=3e-8)
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    rotated_p, rotated = response(rotation @ f)
    np.testing.assert_allclose(rotated_p, rotation @ p, atol=1e-12)
    np.testing.assert_allclose(rotated.strain_energy_density, out.strain_energy_density, atol=1e-13)


def test_external_orthotropic_provider_composes_without_core_material_branch(tmp_path):
    full = prepare(material=ReferenceOrthotropic())
    full.run()
    row = full.history_records[-1]
    assert row["interface_stored_energy"] > 0
    assert row["relative_energy_balance_error"] < 1e-4
    partial = prepare(material=ReferenceOrthotropic())
    partial.run(until_step=411)
    saved = partial.save_checkpoint(tmp_path / "orthotropic-composition")
    restored = prepare(material=ReferenceOrthotropic())
    restored.load_checkpoint(saved)
    restored.run()
    np.testing.assert_array_equal(restored.state.u.value.x.array, full.state.u.value.x.array)
    assert restored.residual.snapshot() == full.residual.snapshot()
    with pytest.raises(ValueError, match="identity|differ"):
        prepare(material=ReferenceOrthotropic(angle=0.4)).load_checkpoint(saved)
