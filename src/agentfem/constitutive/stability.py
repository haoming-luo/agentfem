# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Material-owned, configuration-explicit bounds; never a solver time step."""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FirstPiolaTangentEnvelope:
    """Declared symmetric dP/dF envelope over a principal-stretch domain.

    The provider must establish -c_minus I <= dP/dF <= c_plus I for ALL its
    admissible internal states and external inputs in this deformation domain.
    The contract does not fit a modulus from sampled stresses or prove a
    provider's assertion. Endpoint checks reject observed violations; they do
    not prove nonlinear trajectory accuracy or stability between endpoints.

    Moduli use the model's consistent stress unit, F is dimensionless, and P
    is work conjugate to F per reference volume. State-restricted envelopes,
    other stress measures and nonsymmetric tangents are not represented here.
    """

    positive_modulus: float
    negative_modulus: float
    minimum_stretch: float
    maximum_stretch: float
    source: str

    def __post_init__(self):
        for name in ("positive_modulus", "negative_modulus", "minimum_stretch", "maximum_stretch"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise ValueError(f"{name} must be a finite real scalar.")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.positive_modulus < 0 or self.negative_modulus < 0:
            raise ValueError("Tangent envelope moduli must be nonnegative.")
        if not 0 < self.minimum_stretch <= 1 <= self.maximum_stretch:
            raise ValueError("Stretch domain must be positive and contain the virgin identity.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("A material-envelope derivation/source is required.")
        object.__setattr__(self, "source", self.source.strip())

    def summary(self):
        return {
            "schema_version": 1,
            "kind": "first_piola_tangent_envelope",
            "configuration": "reference",
            "modulus_unit": "model_consistent_stress",
            "tangent": "dP_iJ/dF_kL",
            "state_scope": "all_provider_admissible_states_and_inputs",
            "evidence": "provider_declared_domain_envelope_not_automatic_verification",
            **asdict(self),
        }

    def validate_deformation(self, deformation_gradient):
        if np.iscomplexobj(deformation_gradient):
            raise ValueError("Stability envelope deformation gradients must be real.")
        f = np.asarray(deformation_gradient, dtype=float)
        if f.ndim != 3 or f.shape[1:] != (3, 3) or not np.isfinite(f).all():
            raise ValueError("Envelope requires finite (points,3,3) deformation gradients.")
        if not len(f):
            return
        singular = np.linalg.svd(f, compute_uv=False)
        if (np.any(np.linalg.det(f) <= 0)
                or np.any(singular[:, -1] < self.minimum_stretch * (1 - 1e-12))
                or np.any(singular[:, 0] > self.maximum_stretch * (1 + 1e-12))):
            raise ValueError("Deformation leaves the material stability envelope domain.")

    def validate_response(self, deformation_gradient, tangent):
        self.validate_deformation(deformation_gradient)
        count = len(deformation_gradient)
        if np.iscomplexobj(tangent):
            raise ValueError("Stability envelope tangents must be real.")
        a = np.asarray(tangent, dtype=float)
        if a.shape not in ((count, 9, 9), (count, 3, 3, 3, 3)) or not np.isfinite(a).all():
            raise ValueError("Envelope requires finite point-matched dP/dF tangents.")
        if not count:
            return
        # Bound temporary symmetric matrices for large integration-point arrays.
        flat = a.reshape(-1, 9, 9)
        for start in range(0, count, 1024):
            matrix = flat[start:start + 1024]
            scale = np.maximum(np.max(np.abs(matrix), axis=(1, 2)), np.finfo(float).tiny)
            if np.any(np.max(np.abs(matrix - matrix.swapaxes(1, 2)), axis=(1, 2)) > 1e-10 * scale):
                raise ValueError("Material stability envelope requires symmetric dP/dF.")
            eigenvalues = np.linalg.eigvalsh((matrix + matrix.swapaxes(1, 2)) / 2)
            tolerance = 1e-10 * np.maximum(scale, max(self.positive_modulus, self.negative_modulus))
            if (np.any(eigenvalues[:, -1] > self.positive_modulus + tolerance)
                    or np.any(eigenvalues[:, 0] < -self.negative_modulus - tolerance)):
                raise ValueError("Observed tangent violates the material stability envelope.")
