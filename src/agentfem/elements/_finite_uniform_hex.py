# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private total-Lagrangian contribution; no public finite-strain Step claim.

Materials own P(F, state) and dP/dF. This operator owns only fixed-reference
kinematics, work-conjugate force mapping and an objective artificial potential.
No constitutive history, state commits or time-step policy is hidden here.
"""

from dataclasses import dataclass

import numpy as np

from ._hex_validity import require_positive_hex_jacobian
from ._uniform_hex import _frozen, _geometry


@dataclass(frozen=True)
class FiniteUniformHexResponse:
    internal_force: np.ndarray
    physical_energy: np.ndarray
    hourglass_energy: np.ndarray


class FiniteUniformHexBatch:
    """Reference-volume-average deformation gradient, in Basix node order.

    This is not a selective volumetric F-bar correction.

    Stress must be first Piola (not Cauchy), energy density per reference volume,
    tangent A[i,J,k,L] = dP[i,J]/dF[k,L]. Reference stiffness stabilization is
    explicit and fixed; neither damage-dependent nor viscous control is implied.
    """

    def __init__(
        self,
        coordinates,
        *,
        density,
        hourglass_modulus,
        hourglass_scale,
        chunk_size=1024,
    ):
        x = np.asarray(coordinates, dtype=float)
        if (
            x.ndim != 3
            or x.shape[1:] != (8, 3)
            or not len(x)
            or not np.isfinite(x).all()
        ):
            raise ValueError("Coordinates must be finite nonempty (cells, 8, 3).")
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, (int, np.integer))
            or chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer.")
        self.chunk_size = int(chunk_size)
        self.coordinates = _frozen(x)
        count = len(x)

        def coefficient(value, name):
            value = np.asarray(value, dtype=float)
            if (
                value.shape not in ((), (count,))
                or not np.isfinite(value).all()
                or np.any(value <= 0)
            ):
                raise ValueError(f"{name} must be positive finite scalar or (cells,).")
            return np.broadcast_to(value, (count,))

        rho = coefficient(density, "density")
        modulus = coefficient(hourglass_modulus, "hourglass_modulus")
        scale = coefficient(hourglass_scale, "hourglass_scale")
        volumes, gradients, modes, nodal = [], [], [], []
        for region in self._regions():
            volume, gradient, gamma, nodal_volume = _geometry(x[region])
            volumes.append(volume)
            gradients.append(gradient)
            modes.append(gamma)
            nodal.append(nodal_volume)
        self.volume = _frozen(np.concatenate(volumes))
        self.average_gradient = _frozen(np.concatenate(gradients))
        self.hourglass_modes = _frozen(np.concatenate(modes))
        self.lumped_mass = _frozen(rho[:, None] * np.concatenate(nodal))
        self.hourglass_coefficient = _frozen(scale * modulus * self.volume ** (1 / 3))
        if (
            not np.isfinite(self.lumped_mass).all()
            or not np.isfinite(self.hourglass_coefficient).all()
        ):
            raise ValueError("Finite Hex8 coefficients overflowed.")

    def _regions(self):
        for start in range(0, len(self.coordinates), self.chunk_size):
            yield slice(start, min(start + self.chunk_size, len(self.coordinates)))

    def _displacement(self, displacement):
        u = np.asarray(displacement, dtype=float)
        if u.shape != self.coordinates.shape or not np.isfinite(u).all():
            raise ValueError("Displacement must be finite (cells, 8, 3).")
        return u

    def deformation_gradient(self, displacement):
        u = self._displacement(displacement)
        gradients = []
        for region in self._regions():
            require_positive_hex_jacobian(self.coordinates[region] + u[region])
            f = np.eye(3) + np.einsum(
                "cai,caJ->ciJ", u[region], self.average_gradient[region]
            )
            if not np.isfinite(f).all() or np.any(np.linalg.det(f) <= 0):
                raise ValueError(
                    "Mean deformation gradient must have positive determinant."
                )
            gradients.append(f)
        return np.concatenate(gradients)

    def response(self, displacement, *, first_piola, stored_energy_density):
        u = self._displacement(displacement)
        # Full cell validity is distinct from det(F_bar)>0: the latter can hide folds.
        self.deformation_gradient(u)
        p = np.asarray(first_piola, dtype=float)
        energy = np.asarray(stored_energy_density, dtype=float)
        count = len(self.coordinates)
        if (
            p.shape != (count, 3, 3)
            or energy.shape != (count,)
            or not np.isfinite(p).all()
            or not np.isfinite(energy).all()
        ):
            raise ValueError(
                "Expected finite first Piola (cells,3,3) and reference energy density (cells,)."
            )
        forces, artificial = [], []
        for region in self._regions():
            gamma = self.hourglass_modes[region]
            # gamma^T X = 0, so gamma^T u = gamma^T x and rotates objectively.
            modes = np.einsum("cam,cai->cmi", gamma, u[region])
            coefficient = self.hourglass_coefficient[region]
            force = self.volume[region, None, None] * np.einsum(
                "ciJ,caJ->cai", p[region], self.average_gradient[region]
            )
            force += coefficient[:, None, None] * np.einsum(
                "cam,cmi->cai", gamma, modes
            )
            forces.append(force)
            artificial.append(0.5 * coefficient * np.sum(modes**2, axis=(1, 2)))
        result = FiniteUniformHexResponse(
            np.concatenate(forces), self.volume * energy, np.concatenate(artificial)
        )
        if any(
            not np.isfinite(value).all()
            for value in (
                result.internal_force,
                result.physical_energy,
                result.hourglass_energy,
            )
        ):
            raise ValueError("Finite Hex8 response overflowed.")
        return result

    def tangent_action(self, direction, *, first_piola_tangent):
        du = self._displacement(direction)
        tangent = np.asarray(first_piola_tangent, dtype=float)
        if (
            tangent.shape != (len(self.coordinates), 3, 3, 3, 3)
            or not np.isfinite(tangent).all()
        ):
            raise ValueError(
                "Expected finite dP[i,J]/dF[k,L] with shape (cells,3,3,3,3)."
            )
        actions = []
        for region in self._regions():
            gradient, gamma = (
                self.average_gradient[region],
                self.hourglass_modes[region],
            )
            df = np.einsum("cak,caL->ckL", du[region], gradient)
            dp = np.einsum("ciJkL,ckL->ciJ", tangent[region], df)
            action = self.volume[region, None, None] * np.einsum(
                "ciJ,caJ->cai", dp, gradient
            )
            action += self.hourglass_coefficient[region, None, None] * np.einsum(
                "cam,cbm,cbi->cai", gamma, gamma, du[region]
            )
            actions.append(action)
        result = np.concatenate(actions)
        if not np.isfinite(result).all():
            raise ValueError("Finite Hex8 tangent action overflowed.")
        return result
