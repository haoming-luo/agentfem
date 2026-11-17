# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Rate-independent small-strain providers with an explicit elastic envelope."""

from dataclasses import dataclass
import numpy as np
from .small_strain_user_material import (
    SmallStrainUserMaterial,
    MaterialParameterSchema,
    SmallStrainMaterialPointOutput,
    small_strain_tangent_convention,
)
from .user_material import MaterialStateSchema, MaterialStateVariable
from .plasticity import J2LinearIsotropicHardening, J2PlasticState


@dataclass(frozen=True)
class SmallStrainExplicitMaterial:
    """Provider plus density and an upper elastic tangent for stable time steps.

    Envelope uses engineering strain, xx yy zz xy yz xz. Only rate-independent
    providers with stored-energy and dissipation reporting are admitted. Each
    evaluated tangent is checked against this envelope; it is not a guarantee
    for arbitrary unseen branches of a user material.
    """

    provider: object
    elastic_envelope: np.ndarray
    density: float
    parameters: object = None

    def __post_init__(self):
        if not isinstance(self.provider, SmallStrainUserMaterial):
            raise TypeError("A SmallStrainUserMaterial provider is required.")
        if not callable(getattr(self.provider, "summary", None)):
            raise TypeError(
                "Explicit provider must expose a stable summary for checkpoints."
            )
        if getattr(self.provider, "rate_independent", False) is not True:
            raise NotImplementedError(
                "Explicit small-strain bridge currently requires rate independence."
            )
        C = np.array(self.elastic_envelope, dtype=float, copy=True)
        if (
            C.shape != (6, 6)
            or not np.isfinite(C).all()
            or not np.allclose(C, C.T, rtol=1e-12, atol=0)
        ):
            raise ValueError("Elastic envelope must be finite symmetric 6x6.")
        np.linalg.cholesky(C)
        if not np.isfinite(self.density) or self.density <= 0:
            raise ValueError("Positive finite density required.")
        C.setflags(write=False)
        object.__setattr__(self, "elastic_envelope", C)
        object.__setattr__(
            self,
            "parameters",
            self.provider.parameter_schema.validate(
                getattr(self.provider, "parameters", {})
                if self.parameters is None
                else self.parameters
            ),
        )

    @property
    def name(self):
        return self.provider.name

    def summary(self):
        return dict(
            kind="small_strain_explicit_material",
            name=self.name,
            parameters=dict(self.parameters),
            density=float(self.density),
            elastic_envelope=self.elastic_envelope.tolist(),
            provider=self.provider.summary(),
            state_schema=self.provider.state_schema.summary(),
        )


class _J2Provider:
    rate_independent = True
    parameter_schema = MaterialParameterSchema("agentfem.native_j2.parameters")
    tangent_convention = small_strain_tangent_convention(shear_convention="engineering")
    state_schema = MaterialStateSchema(
        "agentfem.native_j2.state",
        (
            MaterialStateVariable("plastic_strain", shape=(3, 3), output_name="PE"),
            MaterialStateVariable("equivalent_plastic_strain", output_name="PEEQ"),
        ),
    )

    def __init__(self, material):
        self.material = material
        self.name = material.name

    def summary(self):
        return self.material.as_dict()

    def update(self, point):
        old = J2PlasticState(point.state_old[:9].reshape(3, 3), point.state_old[9])
        r = self.material.update(point.strain_new, old)
        pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
        C = np.array(
            [[r.algorithmic_tangent[i, j, k, l] for k, l in pairs] for i, j in pairs]
        )
        elastic = point.strain_new - r.state.plastic_strain
        stored = (
            0.5 * np.sum(elastic * r.stress)
            + 0.5
            * self.material.hardening_modulus
            * r.state.equivalent_plastic_strain**2
        )
        return SmallStrainMaterialPointOutput(
            r.stress,
            C,
            np.r_[r.state.plastic_strain.ravel(), r.state.equivalent_plastic_strain],
            self.state_schema,
            self.tangent_convention,
            stored_energy_density=float(stored),
            dissipation_density_increment=r.energy_increment.discrete_dissipation,
        )


def explicit_small_strain_j2(material, *, density):
    """Adapt the existing radial-return J2 model without duplicating its law."""
    if not isinstance(material, J2LinearIsotropicHardening):
        raise TypeError("Expected J2LinearIsotropicHardening.")
    pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
    T = material.elastic_tangent()
    C = np.array([[T[i, j, k, l] for k, l in pairs] for i, j in pairs])
    return SmallStrainExplicitMaterial(_J2Provider(material), C, density)
