"""Two educational objective history laws; not calibrated engineering models.

Both use Green strain and reference-volume stresses. They deliberately retain
positive-volume, moderate-strain limitations of St Venant--Kirchhoff elasticity.
No model registration, Newton iteration, state commit or file output lives here.
"""

import numpy as np
from agentfem import constitutive as c


class HistoryMaterial:
    tangent_convention = c.MaterialTangentConvention.first_piola_deformation_gradient()

    def __init__(
        self,
        *,
        mechanism="viscoelastic",
        lame=100.0,
        shear=50.0,
        branch=30.0,
        relaxation_time=0.4,
        damage_rate=2.0,
    ):
        if mechanism not in {"viscoelastic", "damage"}:
            raise ValueError("Choose viscoelastic or damage.")
        self.name = f"educational_{mechanism}"
        self.mechanism = mechanism
        self.lame, self.shear, self.branch = float(lame), float(shear), float(branch)
        self.relaxation_time, self.damage_rate = (
            float(relaxation_time),
            float(damage_rate),
        )
        if not all(
            np.isfinite(v) and v > 0
            for v in (lame, shear, branch, relaxation_time, damage_rate)
        ):
            raise ValueError("Example parameters must be positive and finite.")
        history = (
            c.MaterialStateVariable("Q", shape=(3, 3))
            if mechanism == "viscoelastic"
            else c.MaterialStateVariable("KAPPA")
        )
        self.state_schema = c.MaterialStateSchema(
            self.name, (history, c.MaterialStateVariable("AGE", unit="s"))
        )

    def summary(self):
        return {
            "kind": self.name,
            "implementation_version": 1,
            "lame": self.lame,
            "shear": self.shear,
            "branch": self.branch,
            "relaxation_time": self.relaxation_time,
            "damage_rate": self.damage_rate,
            "state_schema": self.state_schema.summary(),
            "scope": "educational moderate strain; no experimental calibration",
        }

    def update_batch(self, request):
        return c.MaterialPointBatchOutput(
            tuple(self.update(point) for point in request.points)
        )

    def update(self, point):
        F = point.deformation_gradient_new
        E = 0.5 * (F.T @ F - np.eye(3))
        identity = np.eye(3)
        sym = np.einsum("ik,jl->ijkl", identity, identity) + np.einsum(
            "il,jk->ijkl", identity, identity
        )
        elastic = (
            self.lame * np.einsum("ij,kl->ijkl", identity, identity) + self.shear * sym
        )
        S = self.lame * np.trace(E) * identity + 2 * self.shear * E
        age = point.state_old[-1] + point.time_increment
        if self.mechanism == "viscoelastic":
            ratio = point.time_increment / self.relaxation_time
            Q = (point.state_old[:9].reshape(3, 3) + ratio * E) / (1 + ratio)
            S += 2 * self.branch * (E - Q)
            C = elastic + self.branch / (1 + ratio) * sym
            state = np.r_[Q.ravel(), age]
            energy = (
                0.5 * self.lame * np.trace(E) ** 2
                + self.shear * np.sum(E * E)
                + self.branch * np.sum((E - Q) ** 2)
            )
        else:
            magnitude = float(np.linalg.norm(E))
            kappa = max(float(point.state_old[0]), magnitude)
            retention = np.exp(-self.damage_rate * kappa)
            C = retention * elastic
            if magnitude > point.state_old[0] and magnitude > 0:
                C -= (
                    self.damage_rate
                    * retention
                    * np.einsum("ij,kl->ijkl", S, E / magnitude)
                )
            S *= retention
            state = np.array([kappa, age])
            energy = None  # This example makes no damage energy/dissipation claim.
        P = F @ S
        tangent = np.einsum("ik,LJ->iJkL", identity, S) + np.einsum(
            "iA,AJBL,kB->iJkL", F, C, F
        )
        return c.MaterialPointOutput(
            cauchy_stress=P @ F.T / np.linalg.det(F),
            consistent_tangent=tangent.reshape(9, 9),
            state_new=state,
            strain_energy_density=energy,
            tangent_convention=self.tangent_convention,
            state_schema=self.state_schema,
        )
