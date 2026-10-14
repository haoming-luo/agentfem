# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private total-Lagrangian contribution; no public finite-strain Step claim.

Materials own P(F, state) and dP/dF. This operator owns only fixed-reference
kinematics, work-conjugate force mapping and an objective artificial potential.
No constitutive history, state commits or time-step policy is hidden here.
"""

from dataclasses import dataclass

import numpy as np

from .._orientation import positive_finite_determinants

from ._hex_validity import (
    require_positive_hex_jacobian,
    reference_perturbation_radius,
    admits_reference_perturbation,
    admits_affine_perturbation,
)
from ._uniform_hex import _frozen, _geometry


@dataclass(frozen=True)
class FiniteUniformHexResponse:
    internal_force: np.ndarray
    physical_energy: np.ndarray | None
    hourglass_energy: np.ndarray


@dataclass(frozen=True)
class FiniteHexTangentSpectrum:
    """Signed enclosure, not a claim that the constrained system is unstable."""

    positive_eigenvalue_upper_bound: float
    negative_eigenvalue_magnitude_bound: float
    negative_material_curvature_cells: int


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
        if x.ndim != 3 or x.shape[1:] != (8, 3) or not np.isfinite(x).all():
            raise ValueError("Coordinates must be finite (cells, 8, 3).")
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
        if count == 0:
            # Empty owned-cell partitions still enter shared material/assembly
            # collectives; no artificial element or mass is introduced.
            self.volume = _frozen(np.empty(0))
            self.average_gradient = _frozen(np.empty((0, 8, 3)))
            self.hourglass_modes = _frozen(np.empty((0, 8, 4)))
            self.lumped_mass = _frozen(np.empty((0, 8)))
            self.hourglass_coefficient = _frozen(np.empty(0))
            self._mass_gradient_gram = _frozen(np.empty((0, 3, 3)))
            self._hourglass_spectral_bound = _frozen(np.empty(0))
            self._isotropic_mass_gradient_bound = _frozen(np.empty(0))
            self._reference_perturbation_radius = _frozen(np.empty(0))
            return
        volumes, gradients, modes, nodal, radii = [], [], [], [], []
        for region in self._regions():
            volume, gradient, gamma, nodal_volume = _geometry(x[region])
            volumes.append(volume)
            gradients.append(gradient)
            modes.append(gamma)
            nodal.append(nodal_volume)
            radii.append(reference_perturbation_radius(x[region]))
        self.volume = _frozen(np.concatenate(volumes))
        self._reference_perturbation_radius = _frozen(np.concatenate(radii))
        self.average_gradient = _frozen(np.concatenate(gradients))
        self.hourglass_modes = _frozen(np.concatenate(modes))
        self.lumped_mass = _frozen(rho[:, None] * np.concatenate(nodal))
        self.hourglass_coefficient = _frozen(scale * modulus * self.volume ** (1 / 3))
        if (
            not np.isfinite(self.lumped_mass).all()
            or np.any(self.lumped_mass <= 0)
            or not np.isfinite(self.hourglass_coefficient).all()
        ):
            raise ValueError("Finite Hex8 coefficients overflowed.")
        # Fixed-reference geometry/mass terms are independent of material state.
        # Retain only 3x3 Gram and scalar hourglass bounds, never cell matrices.
        grams, hourglass_bounds = [], []
        for region in self._regions():
            gradient = self.average_gradient[region]
            gamma = self.hourglass_modes[region]
            inverse_mass = 1 / self.lumped_mass[region]
            grams.append(np.einsum("caJ,caK,ca->cJK", gradient, gradient, inverse_mass))
            hourglass_bounds.append(
                self.hourglass_coefficient[region]
                * np.linalg.eigvalsh(
                    np.einsum("cam,can,ca->cmn", gamma, gamma, inverse_mass)
                )[:, -1]
            )
        self._mass_gradient_gram = _frozen(np.concatenate(grams))
        self._hourglass_spectral_bound = _frozen(np.concatenate(hourglass_bounds))
        gram = self._mass_gradient_gram
        spherical = np.trace(gram, axis1=1, axis2=2) / 3
        difference = np.max(
            np.abs(gram - spherical[:, None, None] * np.eye(3)), axis=(1, 2)
        )
        nearly_isotropic = difference <= 32 * np.finfo(float).eps * np.max(
            np.abs(gram), axis=(1, 2)
        )
        # For G=gI the signed bounds reduce to g*lambda(A+/-). Small
        # roundoff departures use ||G||_infinity >= lambda_max(G), retaining
        # an upper bound rather than treating a nearly spherical Gram as exact.
        self._isotropic_mass_gradient_bound = _frozen(
            np.where(
                nearly_isotropic,
                np.max(np.sum(np.abs(gram), axis=2), axis=1),
                0.0,
            )
        )
        if (
            not np.isfinite(self._mass_gradient_gram).all()
            or not np.isfinite(self._hourglass_spectral_bound).all()
        ):
            raise ValueError("Finite Hex8 reference spectral geometry overflowed.")

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
            f = np.eye(3) + u[region].swapaxes(1, 2) @ self.average_gradient[region]
            admitted = admits_reference_perturbation(
                self.coordinates[region],
                u[region],
                self._reference_perturbation_radius[region],
            )
            if np.any(~admitted):
                admitted[~admitted] = admits_affine_perturbation(
                    self.coordinates[region][~admitted], u[region][~admitted],
                    f[~admitted], self._reference_perturbation_radius[region][~admitted],
                )
            if np.any(~admitted):
                require_positive_hex_jacobian(
                    self.coordinates[region][~admitted] + u[region][~admitted]
                )
            if not positive_finite_determinants(f).all():
                raise ValueError(
                    "Mean deformation gradient must have positive determinant."
                )
            gradients.append(f)
        return np.concatenate(gradients) if gradients else np.empty((0, 3, 3))

    def response(self, displacement, *, first_piola, stored_energy_density):
        """Map stress to force; None energy remains unavailable, never zero."""
        u = self._displacement(displacement)
        # Full cell validity is distinct from det(F_bar)>0: the latter can hide folds.
        self.deformation_gradient(u)
        return self._response_from_checked_displacement(
            u, first_piola=first_piola, stored_energy_density=stored_energy_density
        )

    def _response_from_checked_displacement(
        self, displacement, *, first_piola, stored_energy_density
    ):
        """Internal continuation of one admitted kinematics evaluation.

        Never cache this admission across updates. The material bridge owns a
        frozen displacement copy and validates it before invoking provider code.
        Standalone callers use response(), which always checks full cell validity.
        """
        u = self._displacement(displacement)
        p = np.asarray(first_piola, dtype=float)
        energy = (
            None
            if stored_energy_density is None
            else np.asarray(stored_energy_density, dtype=float)
        )
        count = len(self.coordinates)
        if (
            p.shape != (count, 3, 3)
            or (energy is not None and energy.shape != (count,))
            or not np.isfinite(p).all()
            or (energy is not None and not np.isfinite(energy).all())
        ):
            raise ValueError(
                "Expected finite first Piola (cells,3,3) and reference energy density (cells,)."
            )
        forces, artificial = [], []
        for region in self._regions():
            gamma = self.hourglass_modes[region]
            # gamma^T X = 0, so gamma^T u = gamma^T x and rotates objectively.
            modes = gamma.swapaxes(1, 2) @ u[region]
            coefficient = self.hourglass_coefficient[region]
            force = self.volume[region, None, None] * (
                self.average_gradient[region] @ p[region].swapaxes(1, 2)
            )
            force += coefficient[:, None, None] * (gamma @ modes)
            forces.append(force)
            artificial.append(0.5 * coefficient * np.sum(modes**2, axis=(1, 2)))
        result = FiniteUniformHexResponse(
            np.concatenate(forces) if forces else np.empty((0, 8, 3)),
            None if energy is None else self.volume * energy,
            np.concatenate(artificial) if artificial else np.empty(0),
        )
        if any(
            not np.isfinite(value).all()
            for value in (
                result.internal_force,
                result.physical_energy,
                result.hourglass_energy,
            )
            if value is not None
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
        result = np.concatenate(actions) if actions else np.empty((0, 8, 3))
        if not np.isfinite(result).all():
            raise ValueError("Finite Hex8 tangent action overflowed.")
        return result

    def tangent_spectral_bound(self, *, first_piola_tangent):
        """Instantaneous symmetric nonnegative tangent screen, not a trajectory guarantee.

        The 9-by-9 material Gram and 4-by-4 hourglass Gram replace dense cell
        matrices. Negative/nonsymmetric tangents require a separate physical and
        numerical policy; taking their absolute values would hide instability.
        The Procedure must refresh this screen when the accepted state changes.
        """
        report = self.tangent_spectral_report(first_piola_tangent=first_piola_tangent)
        if report.negative_material_curvature_cells:
            raise ValueError(
                "Indefinite material tangent requires an explicit curvature policy; "
                "a positive time-step bound is not sufficient."
            )
        return report.positive_eigenvalue_upper_bound

    def tangent_spectral_report(self, *, first_piola_tangent):
        """Enclose positive and negative curvature separately using A=A+ - A-.

        For symmetric material tangents, congruence preserves Loewner order:
        -G.T A- G <= G.T A G + H <= G.T A+ G + H, with H nonnegative.
        Element lumped masses then bound assembled Rayleigh quotients. Negative
        material curvature is diagnostic: constraints and assembly can remove it.
        It is never relabeled a positive-frequency stability guarantee.
        """
        tangent = np.asarray(first_piola_tangent, dtype=float)
        count = len(self.coordinates)
        if tangent.shape != (count, 3, 3, 3, 3) or not np.isfinite(tangent).all():
            raise ValueError("Expected finite dP/dF with shape (cells,3,3,3,3).")
        if count == 0:
            return FiniteHexTangentSpectrum(0.0, 0.0, 0)
        bound = 0.0
        negative_bound = 0.0
        negative_cells = 0
        for region in self._regions():
            matrix = tangent[region].reshape(-1, 9, 9)
            scale = np.max(np.abs(matrix), axis=(1, 2))
            tolerance = 1e-10 * np.maximum(scale, np.finfo(float).tiny)
            if np.any(
                np.max(np.abs(matrix - matrix.swapaxes(1, 2)), axis=(1, 2)) > tolerance
            ):
                raise ValueError(
                    "Non-symmetric material tangent has no admitted conservative stability screen."
                )
            matrix = (matrix + matrix.swapaxes(1, 2)) / 2
            shortcut = self._isotropic_mass_gradient_bound[region]
            fast = shortcut > 0
            physical, negative = np.empty(len(matrix)), np.empty(len(matrix))
            if np.any(fast):
                eigenvalues = np.linalg.eigvalsh(matrix[fast])
                negative_cells += int(
                    np.count_nonzero(eigenvalues[:, 0] < -tolerance[fast])
                )
                factor = self.volume[region][fast] * shortcut[fast]
                physical[fast] = factor * np.maximum(eigenvalues[:, -1], 0)
                negative[fast] = factor * np.maximum(-eigenvalues[:, 0], 0)
            if np.any(~fast):
                eigenvalues, vectors = np.linalg.eigh(matrix[~fast])
                negative_cells += int(
                    np.count_nonzero(eigenvalues[:, 0] < -tolerance[~fast])
                )
                # Positive and negative parts are retained separately, never |A|.
                root = vectors * np.sqrt(np.maximum(eigenvalues, 0))[:, None, :]
                small = self._mass_gradient_gram[region][~fast]
                gram = np.einsum("ik,cJL->ciJkL", np.eye(3), small).reshape(-1, 9, 9)
                physical[~fast] = (
                    self.volume[region][~fast]
                    * np.linalg.eigvalsh(root.swapaxes(1, 2) @ gram @ root)[:, -1]
                )
                negative_root = (
                    vectors * np.sqrt(np.maximum(-eigenvalues, 0))[:, None, :]
                )
                negative[~fast] = (
                    self.volume[region][~fast]
                    * np.linalg.eigvalsh(
                        negative_root.swapaxes(1, 2) @ gram @ negative_root
                    )[:, -1]
                )
            negative_bound = max(negative_bound, float(np.max(negative)))
            hourglass = self._hourglass_spectral_bound[region]
            bound = max(bound, float(np.max(physical + hourglass)))
        if not np.isfinite(bound) or bound <= 0:
            raise ValueError(
                "Finite Hex8 tangent spectral bound must be positive finite."
            )
        if not np.isfinite(negative_bound):
            raise ValueError("Finite Hex8 negative-curvature bound overflowed.")
        return FiniteHexTangentSpectrum(bound, negative_bound, negative_cells)

    def tangent_envelope_bounds(
        self, *, positive_modulus, negative_modulus=0.0
    ) -> tuple[float, float]:
        """Convert a declared symmetric dP/dF envelope into spectral bounds.

        The material must establish ``-negative_modulus*I <= A <=
        positive_modulus*I`` in flattened (i,J) coordinates. Moduli have stress
        units, not frequency units. This operator supplies reference geometry,
        element lumped mass and positive hourglass stiffness only. It cannot
        establish the material envelope's state domain or path validity.

        For B mapping nodal displacement to F, the nonzero eigenvalues of
        B M_e^-1 B.T are those of the 3x3 mass-gradient Gram, repeated three
        times. The maximum element Rayleigh bound encloses the assembled
        system when element masses assemble into the same positive lumped mass.
        Return (positive upper bound, negative magnitude bound); never take |A|.
        """
        upper, lower = self.tangent_envelope_cell_bounds(
            positive_modulus=positive_modulus, negative_modulus=negative_modulus
        )
        return float(upper.max(initial=0)), float(lower.max(initial=0))

    def directional_cell_row_majorant(self, *, positive_tensor):
        """Map a supplied symmetric PSD dP/dF upper tensor to nodal rows.

        The provider must establish A <= positive_tensor in the same reference
        frame over the intended domain. This routine proves no material bound.
        It retains directional information rather than replacing the tensor by
        its largest eigenvalue. Chunk-local 24x24 matrices are discarded after
        absolute row summation; no global stiffness or material State is built.
        The returned (cells,8,3) rows include fixed hourglass stiffness.
        """
        raw = np.asarray(positive_tensor)
        count = len(self.coordinates)
        if (
            raw.dtype.kind not in "fiu"
            or raw.shape not in ((3, 3, 3, 3), (count, 3, 3, 3, 3))
            or not np.isfinite(raw).all()
        ):
            raise ValueError(
                "Expected finite real positive tensor (3,3,3,3) or cell batch."
            )
        tensors = np.broadcast_to(raw, (count, 3, 3, 3, 3))
        result = np.empty((count, 8, 3))
        for region in self._regions():
            matrix = np.asarray(tensors[region], dtype=float).reshape(-1, 9, 9)
            scale = np.max(np.abs(matrix), axis=(1, 2))
            tolerance = (
                64 * np.finfo(float).eps * np.maximum(scale, np.finfo(float).tiny)
            )
            if np.any(
                np.max(np.abs(matrix - matrix.swapaxes(1, 2)), axis=(1, 2)) > tolerance
            ):
                raise ValueError("Positive tensor must be symmetric in paired indices.")
            matrix = (matrix + matrix.swapaxes(1, 2)) / 2
            minimum = np.linalg.eigvalsh(matrix)[:, 0]
            if np.any(minimum < -tolerance):
                raise ValueError("Positive tensor must be positive semidefinite.")
            # Lift only admitted eigensolver-scale negative roundoff, never
            # convert genuine softening into positive stiffness.
            matrix = matrix + np.maximum(-minimum, 0)[:, None, None] * np.eye(9)
            g = self.average_gradient[region]
            upper = self.volume[region, None, None, None, None] * np.einsum(
                "caJ,ciJkL,cbL->caibk",
                g,
                matrix.reshape(-1, 3, 3, 3, 3),
                g,
                optimize=True,
            )
            modes = self.hourglass_modes[region]
            stabilization = self.hourglass_coefficient[region, None, None] * np.einsum(
                "cam,cbm->cab", modes, modes
            )
            upper += (
                stabilization[:, :, None, :, None] * np.eye(3)[None, None, :, None, :]
            )
            result[region] = np.abs(upper).sum(axis=(3, 4))
        if not np.isfinite(result).all():
            raise ValueError("Directional Hex8 majorant overflowed.")
        return _frozen(result)

    def directional_cell_mass_majorant(self, *, positive_tensor):
        """Return a mass-shaped diagonal upper bound for a supplied PSD tensor.

        K_e <= lambda_e M_e implies K <= assembled diag(lambda_e M_e).
        The material spectrum uses a 9x9 Gram; the positive hourglass bound
        is added separately. This is an alternative to absolute element rows,
        not a claim that either bound is always smaller. The caller still owns
        the tensor's domain of validity; no evolving material bound is inferred.
        """
        raw = np.asarray(positive_tensor)
        count = len(self.coordinates)
        if (raw.dtype.kind not in "fiu"
                or raw.shape not in ((3, 3, 3, 3), (count, 3, 3, 3, 3))
                or not np.isfinite(raw).all()):
            raise ValueError("Expected finite real positive tensor (3,3,3,3) or cell batch.")
        tensors = np.broadcast_to(raw, (count, 3, 3, 3, 3))
        result = np.empty((count, 8, 3))
        for region in self._regions():
            matrix = np.asarray(tensors[region], dtype=float).reshape(-1, 9, 9)
            scale = np.max(np.abs(matrix), axis=(1, 2))
            tolerance = 64 * np.finfo(float).eps * np.maximum(scale, np.finfo(float).tiny)
            if np.any(np.max(abs(matrix-matrix.swapaxes(1, 2)), axis=(1, 2)) > tolerance):
                raise ValueError("Positive tensor must be symmetric in paired indices.")
            eigenvalues, vectors = np.linalg.eigh((matrix+matrix.swapaxes(1, 2))/2)
            if np.any(eigenvalues[:, 0] < -tolerance):
                raise ValueError("Positive tensor must be positive semidefinite.")
            root = vectors * np.sqrt(np.maximum(eigenvalues, 0))[:, None, :]
            gram = np.einsum("ik,cJL->ciJkL", np.eye(3),
                             self._mass_gradient_gram[region]).reshape(-1, 9, 9)
            reduced = root.swapaxes(1, 2) @ gram @ root
            physical = self.volume[region] * np.maximum(
                np.linalg.eigvalsh((reduced+reduced.swapaxes(1, 2))/2)[:, -1], 0)
            upper = (physical+self._hourglass_spectral_bound[region]) * (
                1+256*np.finfo(float).eps)
            result[region] = upper[:, None, None] * self.lumped_mass[region, :, None]
        if not np.isfinite(result).all():
            raise ValueError("Mass-shaped Hex8 majorant overflowed.")
        return _frozen(result)

    def tangent_envelope_cell_row_majorant(self, *, positive_modulus):
        """Unscaled isotropic nodal diagonal majorant, including hourglass.

        Given symmetric A <= c I, K_e <= (c V G G.T + k_hg Gamma Gamma.T)
        tensor I. Absolute row sums of this 8x8 upper operator give a diagonal
        majorant without storing 24x24 element matrices. Assemble these (cell,8)
        rows before combining other operators in an inertia plan. This is not
        evidence that the actual material is PSD or its declared c is valid.
        """
        raw = np.asarray(positive_modulus)
        count = len(self.coordinates)
        if (
            raw.dtype.kind not in "fiu"
            or raw.shape not in ((), (count,))
            or not np.isfinite(raw).all()
            or np.any(raw < 0)
        ):
            raise ValueError(
                "positive_modulus must be nonnegative real scalar or (cells,)."
            )
        coefficient = np.broadcast_to(raw, (count,))
        result = np.empty((count, 8))
        for region in self._regions():
            g, gamma = self.average_gradient[region], self.hourglass_modes[region]
            upper = (coefficient[region] * self.volume[region])[
                :, None, None
            ] * np.einsum("caJ,cbJ->cab", g, g)
            upper += self.hourglass_coefficient[region, None, None] * np.einsum(
                "cam,cbm->cab", gamma, gamma
            )
            result[region] = np.abs(upper).sum(axis=2)
        if not np.isfinite(result).all():
            raise ValueError("Finite Hex8 diagonal majorant overflowed.")
        return _frozen(result)

    def tangent_envelope_cell_bounds(self, *, positive_modulus, negative_modulus=0.0):
        """Per-element physical-mass envelope for an external inertia policy.

        Same supplied material-envelope assumptions as tangent_envelope_bounds;
        positive and negative curvature remain separate. Numerical mass scaling
        cannot cure negative physical curvature.
        """
        count = len(self.coordinates)

        def coefficients(value, name):
            array = np.asarray(value, dtype=float)
            if (
                array.shape not in ((), (count,))
                or not np.isfinite(array).all()
                or np.any(array < 0)
            ):
                raise ValueError(
                    f"{name} must be a nonnegative finite scalar or (cells,)."
                )
            return np.broadcast_to(array, (count,))

        positive = coefficients(positive_modulus, "positive_modulus")
        negative = coefficients(negative_modulus, "negative_modulus")
        if count == 0:
            return np.empty(0), np.empty(0)
        # No cell 24x24 stiffness and no material 9x9 array is constructed.
        factor = self.volume * np.linalg.eigvalsh(self._mass_gradient_gram)[:, -1]
        upper = positive * factor + self._hourglass_spectral_bound
        lower_magnitude = negative * factor
        if not np.isfinite(upper).all() or not np.isfinite(lower_magnitude).all():
            raise ValueError(
                "Finite Hex8 material-envelope spectral conversion overflowed."
            )
        return upper, lower_magnitude
