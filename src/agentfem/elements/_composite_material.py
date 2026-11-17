# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Transactional material integration over the composite projected quadrature."""

import numpy as np
from ..constitutive.small_strain_user_material import (
    SmallStrainMaterialPointBatchInput,
    validated_small_strain_batch_update,
)


class CompositeMaterialBatch:
    def __init__(
        self, elements, material, *, mean_dilatation=False, dofs_per_cell=None
    ):
        self.material = material
        self.dofs_per_cell = (
            (3 * elements[0].gradient_coefficients.shape[1] if elements else 30)
            if dofs_per_cell is None
            else dofs_per_cell
        )
        self.points_per_cell = 48 if mean_dilatation or self.dofs_per_cell == 33 else 4
        quadrature = [
            e.material_quadrature(mean_dilatation=mean_dilatation) for e in elements
        ]
        self.B = np.asarray([q[0] for q in quadrature]).reshape(
            -1, self.points_per_cell, 6, self.dofs_per_cell
        )
        self.weights = np.asarray([q[1] for q in quadrature]).reshape(
            -1, self.points_per_cell
        )
        self.volume = np.array([e.volume for e in elements])
        n = self.points_per_cell * len(elements)
        self.strain = np.zeros((n, 3, 3))
        self.states = np.tile(material.provider.state_schema.initial_state(), (n, 1))
        self.stress = np.zeros((n, 3, 3))
        self.stored = np.zeros(n)
        self.dissipation = np.zeros(n)
        self.time = 0.0
        self.trial = None
        self.pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))

    def evaluate(self, displacement, time):
        self.trial = None
        eng = np.einsum("cqij,cj->cqi", self.B, displacement).reshape(-1, 6)
        eps = np.zeros_like(self.strain)
        for k, (i, j) in enumerate(self.pairs):
            eps[:, i, j] = eps[:, j, i] = eng[:, k] * (1 if i == j else 0.5)
        if time < self.time or not np.isfinite(time):
            raise ValueError("Material trial time must not precede accepted time.")
        if not len(eps):
            self.trial = (
                eps,
                self.states.copy(),
                self.stress.copy(),
                self.stored.copy(),
                self.dissipation.copy(),
                float(time),
            )
            return np.zeros((0, self.dofs_per_cell))
        m = self.material
        request = SmallStrainMaterialPointBatchInput(
            self.strain,
            eps,
            float(time),
            max(time - self.time, 1e-30),
            m.parameters,
            self.states,
            m.provider.state_schema,
            m.provider.parameter_schema,
        )
        r = validated_small_strain_batch_update(m.provider, request)
        if r.stored_energy_density is None or r.dissipation_density_increment is None:
            raise ValueError(
                "Explicit material must report stored energy and dissipation."
            )
        if r.minimum_suggested_time_scale < 1:
            raise ValueError(
                "Material requests a smaller increment; fixed explicit step rejected."
            )
        C = r.consistent_tangent.copy()
        if r.tangent_convention.shear_convention == "tensor":
            C[:, :, 3:] *= 0.5
        # Congruence keeps strain units balanced and tests the envelope relatively.
        inv = np.linalg.inv(np.linalg.cholesky(m.elastic_envelope))
        scaled = np.einsum("ij,qjk,lk->qil", inv, C, inv)
        if not np.allclose(scaled, scaled.transpose(0, 2, 1), rtol=0, atol=1e-10):
            raise ValueError("Material engineering tangent must be symmetric.")
        eig = np.linalg.eigvalsh(scaled)
        if eig.min() < -1e-9 or eig.max() > 1 + 1e-9:
            raise ValueError(
                "Material tangent violates positive elastic stability envelope."
            )
        if np.any(r.dissipation_density_increment < -1e-12):
            raise ValueError("Material dissipation must be nonnegative.")
        stress = np.stack(
            [r.cauchy_stress[:, i, j] for i, j in self.pairs], axis=-1
        ).reshape(-1, self.points_per_cell, 6)
        force = np.einsum("cq,cqij,cqi->cj", self.weights, self.B, stress)
        self.trial = (
            eps,
            r.state_new.copy(),
            r.cauchy_stress.copy(),
            r.stored_energy_density.copy(),
            self.dissipation + np.maximum(r.dissipation_density_increment, 0),
            float(time),
        )
        return force

    def commit(self):
        if self.trial is None:
            raise RuntimeError("No complete material trial to commit.")
        (
            self.strain,
            self.states,
            self.stress,
            self.stored,
            self.dissipation,
            self.time,
        ) = self.trial
        self.trial = None

    def rollback(self):
        self.trial = None

    def snapshot(self):
        return dict(
            strain=self.strain.tolist(),
            states=self.states.tolist(),
            stress=self.stress.tolist(),
            stored=self.stored.tolist(),
            dissipation=self.dissipation.tolist(),
            time=self.time,
        )

    def restore(self, record):
        values = {}
        if set(record) != set(self.snapshot()):
            raise ValueError("Material checkpoint fields mismatch.")
        for name in ("strain", "states", "stress", "stored", "dissipation"):
            a = np.asarray(record[name], dtype=float).reshape(getattr(self, name).shape)
            if not np.isfinite(a).all():
                raise ValueError("Nonfinite material checkpoint.")
            values[name] = a.copy()
        t = float(record["time"])
        if not np.isfinite(t) or t < 0 or np.any(values["dissipation"] < 0):
            raise ValueError("Invalid material checkpoint time/energy.")
        for name, a in values.items():
            setattr(self, name, a)
        self.time = t
        self.trial = None
