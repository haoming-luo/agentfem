# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Local small-strain uniform-gradient Hex8 candidate; no global Step claim.

Basix owns reference ordering, basis and geometric quadrature. The one-point
constitutive approximation uses the volume-averaged physical gradient, not
the centre gradient. Stabilization is explicit energy-based stiffness control.
"""

from dataclasses import dataclass
from functools import lru_cache

import basix
import numpy as np

from ._hex_validity import require_positive_hex_jacobian


@lru_cache(maxsize=1)
def _reference():
    cell = basix.CellType.hexahedron
    element = basix.create_element(
        basix.ElementFamily.P, cell, 1, basix.LagrangeVariant.equispaced
    )
    points, weights = basix.make_quadrature(cell, 4)
    table = element.tabulate(1, points)
    corners = basix.cell.geometry(cell)
    derivatives = table[1:4, :, :, 0].transpose(1, 2, 0)
    corner_derivatives = element.tabulate(1, corners)[1:4, :, :, 0].transpose(1, 2, 0)
    signs = 2 * corners - 1
    x, y, z = signs.T
    modes = np.column_stack((y * z, x * z, x * y, x * y * z)) / np.sqrt(8)
    return weights, table[0, :, :, 0], derivatives, corner_derivatives, modes


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _frozen(value):
    value = np.array(value, copy=True)
    value.setflags(write=False)
    return value


def _geometry(coordinates):
    """Vectorized geometry preparation; caller bounds temporary batch size."""
    require_positive_hex_jacobian(coordinates)
    weights, shape, gradient, corners, modes = _reference()
    centered = coordinates - coordinates.mean(axis=1, keepdims=True)
    jacobian = np.einsum("cai,qaj->cqij", centered, gradient)
    corner_jacobian = np.einsum("cai,qaj->cqij", centered, corners)
    checks = np.concatenate((jacobian, corner_jacobian), axis=1)
    det = np.linalg.det(checks)
    scale = np.prod(np.linalg.norm(checks, axis=2), axis=2)
    invalid = (
        (~np.isfinite(det))
        | (~np.isfinite(scale))
        | (det <= 64 * np.finfo(float).eps * scale)
    )
    if np.any(invalid):
        cells = np.flatnonzero(np.any(invalid, axis=1)).tolist()
        raise ValueError(
            f"Hex8 has an inverted or singular sampled Jacobian: cells {cells}."
        )
    measure = weights * det[:, : len(weights)]
    volume = measure.sum(axis=1)
    physical = np.einsum("qai,cqij->cqaj", gradient, np.linalg.inv(jacobian))
    average = np.einsum("cq,cqaj->caj", measure, physical) / volume[:, None, None]
    gamma = modes - np.einsum(
        "cai,cim->cam", average, np.einsum("cai,am->cim", centered, modes)
    )
    return volume, average, gamma, measure @ shape


@dataclass(frozen=True)
class UniformHexResponse:
    strain: np.ndarray
    stress: np.ndarray
    internal_force: np.ndarray
    physical_energy: float
    hourglass_energy: float


class UniformHex8:
    """Fixed-reference, positive-Jacobian, small-strain elastic local operator.

    Coordinates follow Basix Hex8 ordering. C maps engineering strains
    (xx, yy, zz, 2yz, 2xz, 2xy) to physical stress components. Hourglass scale
    and modulus are mandatory, recorded inputs, not inferred material choices.
    """

    def __init__(
        self, coordinates, stiffness, *, density, hourglass_modulus, hourglass_scale
    ):
        coordinates = np.asarray(coordinates, dtype=float)
        stiffness = np.asarray(stiffness, dtype=float)
        if coordinates.shape != (8, 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("Hex8 coordinates must be finite (8, 3) in Basix order.")
        if (
            stiffness.shape != (6, 6)
            or not np.all(np.isfinite(stiffness))
            or not np.allclose(stiffness, stiffness.T, rtol=1e-12, atol=0)
        ):
            raise ValueError(
                "Elastic stiffness must be a finite symmetric 6 by 6 matrix."
            )
        if np.linalg.eigvalsh(stiffness).min() <= 0:
            raise ValueError("Elastic stiffness must be positive definite.")
        self.density = _positive(density, "density")
        self.hourglass_modulus = _positive(hourglass_modulus, "hourglass_modulus")
        self.hourglass_scale = _positive(hourglass_scale, "hourglass_scale")
        volume, gradients, gammas, nodal_volume = _geometry(coordinates[None])
        self.volume = float(volume[0])
        average = gradients[0]
        b = np.zeros((6, 8, 3))
        for i in range(3):
            b[i, :, i] = average[:, i]
        for row, (i, j) in enumerate(((1, 2), (0, 2), (0, 1)), start=3):
            b[row, :, i] = average[:, j]
            b[row, :, j] = average[:, i]
        b = b.reshape(6, 24)
        # Remove the affine displacement contribution from the hourglass modes.
        gamma = gammas[0]
        coefficient = (
            self.hourglass_scale * self.hourglass_modulus * self.volume ** (1 / 3)
        )
        self.hourglass_coefficient = coefficient
        physical = self.volume * b.T @ stiffness @ b
        hourglass = coefficient * np.kron(gamma @ gamma.T, np.eye(3))
        mass = self.density * nodal_volume[0]
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
            raise ValueError("Hex8 lumped mass must be positive.")
        if not np.all(np.isfinite(physical)) or not np.all(np.isfinite(hourglass)):
            raise ValueError("Hex8 stiffness overflowed.")
        self.coordinates = _frozen(coordinates)
        self.stiffness = _frozen(stiffness)
        self.average_gradient = _frozen(average)
        self.strain_operator = _frozen(b)
        self.hourglass_modes = _frozen(gamma)
        self.physical_matrix = _frozen(physical)
        self.hourglass_matrix = _frozen(hourglass)
        self.lumped_mass = _frozen(mass)

    def response(self, displacement):
        displacement = np.asarray(displacement, dtype=float)
        if displacement.shape != (8, 3) or not np.all(np.isfinite(displacement)):
            raise ValueError("Hex8 displacement must be finite (8, 3).")
        vector = displacement.ravel()
        strain = self.strain_operator @ vector
        stress = self.stiffness @ strain
        physical_force = self.volume * self.strain_operator.T @ stress
        modes = self.hourglass_modes.T @ displacement
        hg_force = (self.hourglass_coefficient * self.hourglass_modes @ modes).ravel()
        response = UniformHexResponse(
            strain,
            stress,
            (physical_force + hg_force).reshape(8, 3),
            float(0.5 * self.volume * strain @ stress),
            float(0.5 * self.hourglass_coefficient * np.sum(modes**2)),
        )
        if not np.all(np.isfinite(response.internal_force)) or not np.all(
            np.isfinite([response.physical_energy, response.hourglass_energy])
        ):
            raise ValueError("Hex8 response overflowed.")
        return response

    def stability_bound(self):
        """Gershgorin upper bound for this cell's combined mass-scaled stiffness."""
        mass = np.repeat(self.lumped_mass, 3)
        scaled = (self.physical_matrix + self.hourglass_matrix) / np.sqrt(
            mass[:, None] * mass[None, :]
        )
        return float(np.max(np.sum(np.abs(scaled), axis=1)))

    def summary(self):
        return {
            "kind": "uniform_gradient_hex8_local",
            "kinematics": "small_strain",
            "basis_owner": "Basix",
            "constitutive_points": 1,
            "geometric_quadrature_points": len(_reference()[0]),
            "hourglass_method": "affine_corrected_energy_stiffness",
            "hourglass_scale": self.hourglass_scale,
            "hourglass_modulus": self.hourglass_modulus,
            "volume": self.volume,
            "global_step_integrated": False,
            "jacobian_check": "bounded_bernstein_with_floating_point_margin",
        }


@dataclass(frozen=True)
class UniformHexBatchResponse:
    strain: np.ndarray
    stress: np.ndarray
    internal_force: np.ndarray
    physical_energy: np.ndarray
    hourglass_energy: np.ndarray


class UniformHexBatch:
    """Compact fixed-reference cell operator, with bounded geometry temporaries.

    No dense cell stiffness matrices are retained. This is not a mesh assembler
    or time integrator: ownership, DOF mapping and communication stay outside.
    All arrays use the same conventions as UniformHex8. Iterate responses to
    avoid materializing forces and fields for the whole model simultaneously.
    """

    def __init__(
        self,
        coordinates,
        stiffness,
        *,
        density,
        hourglass_modulus,
        hourglass_scale,
        chunk_size=1024,
    ):
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, (int, np.integer))
            or chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer.")
        self.chunk_size = int(chunk_size)
        coordinates = np.asarray(coordinates, dtype=float)
        if (
            coordinates.ndim != 3
            or coordinates.shape[1:] != (8, 3)
            or not len(coordinates)
            or not np.all(np.isfinite(coordinates))
        ):
            raise ValueError("Hex8 coordinates must be finite nonempty (cells, 8, 3).")
        self.size = len(coordinates)
        stiffness = np.asarray(stiffness, dtype=float)
        if stiffness.shape not in ((6, 6), (self.size, 6, 6)) or not np.all(
            np.isfinite(stiffness)
        ):
            raise ValueError("Stiffness must be finite (6, 6) or (cells, 6, 6).")
        if not np.allclose(
            stiffness, stiffness.swapaxes(-1, -2), rtol=1e-12, atol=0
        ) or np.any(np.linalg.eigvalsh(stiffness) <= 0):
            raise ValueError("Elastic stiffness must be symmetric positive definite.")
        self.stiffness = _frozen(stiffness)

        def parameter(value, name):
            array = np.asarray(value, dtype=float)
            if (
                array.shape not in ((), (self.size,))
                or not np.all(np.isfinite(array))
                or np.any(array <= 0)
            ):
                raise ValueError(f"{name} must be positive finite scalar or (cells,).")
            return np.broadcast_to(array, (self.size,))

        rho = parameter(density, "density")
        modulus = parameter(hourglass_modulus, "hourglass_modulus")
        scale = parameter(hourglass_scale, "hourglass_scale")
        self.volume = np.empty(self.size)
        self.average_gradient = np.empty((self.size, 8, 3))
        self.hourglass_modes = np.empty((self.size, 8, 4))
        self.lumped_mass = np.empty((self.size, 8))
        for start in range(0, self.size, self.chunk_size):
            region = slice(start, min(start + self.chunk_size, self.size))
            try:
                volume, gradient, modes, mass = _geometry(coordinates[region])
            except ValueError as exc:
                raise ValueError(f"Hex8 batch starting at cell {start}: {exc}") from exc
            self.volume[region] = volume
            self.average_gradient[region] = gradient
            self.hourglass_modes[region] = modes
            self.lumped_mass[region] = mass * rho[region, None]
        self.hourglass_coefficient = scale * modulus * np.cbrt(self.volume)
        for array in (
            self.volume,
            self.average_gradient,
            self.hourglass_modes,
            self.lumped_mass,
            self.hourglass_coefficient,
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError("Hex8 batch geometry or stabilization overflowed.")
            array.setflags(write=False)
        if np.any(self.lumped_mass <= 0):
            raise ValueError("Hex8 lumped mass must be positive.")

    @property
    def storage_bytes(self):
        return sum(
            array.nbytes
            for array in (
                self.stiffness,
                self.volume,
                self.average_gradient,
                self.hourglass_modes,
                self.lumped_mass,
                self.hourglass_coefficient,
            )
        )

    def stability_bound(self):
        """Conservative element bound from small physical/hourglass Gram spectra.

        The assembled mass must be the sum of these same positive cell masses.
        Extra stiffness (contact, interfaces) must be bounded separately.
        Physical nonzero eigenvalues use a 6x6 Gram matrix; hourglass modes
        use 4x4. Their largest eigenvalues add to bound the combined stiffness.
        No global eigensolve or persistent 24x24 cell matrix is needed.
        """
        bound = 0.0
        shared = (
            np.linalg.cholesky(self.stiffness) if self.stiffness.ndim == 2 else None
        )
        for start in range(0, self.size, self.chunk_size):
            region = slice(start, min(start + self.chunk_size, self.size))
            gradient = self.average_gradient[region]
            b = np.zeros((len(gradient), 6, 8, 3))
            for i in range(3):
                b[:, i, :, i] = gradient[:, :, i]
            for row, (i, j) in enumerate(((1, 2), (0, 2), (0, 1)), start=3):
                b[:, row, :, i] = gradient[:, :, j]
                b[:, row, :, j] = gradient[:, :, i]
            b /= np.sqrt(self.lumped_mass[region])[:, None, :, None]
            b = b.reshape(-1, 6, 24)
            factor = (
                shared
                if shared is not None
                else np.linalg.cholesky(self.stiffness[region])
            )
            gram = b @ b.swapaxes(-1, -2)
            physical_gram = factor.swapaxes(-1, -2) @ gram @ factor
            physical = self.volume[region] * np.linalg.eigvalsh(physical_gram)[:, -1]
            gamma = (
                self.hourglass_modes[region]
                / np.sqrt(self.lumped_mass[region])[:, :, None]
            )
            hourglass = (
                self.hourglass_coefficient[region]
                * np.linalg.eigvalsh(gamma.swapaxes(-1, -2) @ gamma)[:, -1]
            )
            bound = max(bound, float(np.max(physical + hourglass)))
        bound *= 1 + 64 * np.finfo(float).eps
        if not np.isfinite(bound) or bound <= 0:
            raise ValueError("Hex8 spectral bound is not positive finite.")
        return bound

    def _iter_kinematics(self, displacement, *, node_map=None):
        """Evaluate bounded chunks, optionally gathering from global nodal values.

        Supplying the mesh owner's connectivity avoids a whole-mesh (cells,8,3)
        displacement copy on every residual evaluation.
        """
        displacement = np.asarray(displacement, dtype=float)
        if node_map is None:
            if displacement.shape != (self.size, 8, 3):
                raise ValueError(
                    "Hex8 batch displacement must be finite (cells, 8, 3)."
                )
        else:
            node_map = np.asarray(node_map)
            if displacement.ndim != 2 or displacement.shape[1] != 3:
                raise ValueError("Hex8 nodal displacement must have shape (nodes, 3).")
            if (
                node_map.shape != (self.size, 8)
                or node_map.dtype.kind not in "iu"
                or np.any(node_map < 0)
                or np.any(node_map >= len(displacement))
            ):
                raise ValueError(
                    "Hex8 node map must contain valid integer (cells, 8) indices."
                )
        if not np.all(np.isfinite(displacement)):
            raise ValueError("Hex8 displacement must be finite.")
        for start in range(0, self.size, self.chunk_size):
            region = slice(start, min(start + self.chunk_size, self.size))
            u = (
                displacement[region]
                if node_map is None
                else displacement[node_map[region]]
            )
            gradient = self.average_gradient[region]
            du = np.einsum("cai,caj->cij", u, gradient)
            strain = np.column_stack(
                (
                    du[:, 0, 0],
                    du[:, 1, 1],
                    du[:, 2, 2],
                    du[:, 1, 2] + du[:, 2, 1],
                    du[:, 0, 2] + du[:, 2, 0],
                    du[:, 0, 1] + du[:, 1, 0],
                )
            )
            c = self.stiffness if self.stiffness.ndim == 2 else self.stiffness[region]
            stress = np.einsum("ij,cj->ci" if c.ndim == 2 else "cij,cj->ci", c, strain)
            gamma = self.hourglass_modes[region]
            modes = np.einsum("cam,cai->cmi", gamma, u)
            coefficient = self.hourglass_coefficient[region]
            physical = 0.5 * self.volume[region] * np.einsum("ci,ci->c", strain, stress)
            hourglass = 0.5 * coefficient * np.einsum("cmi,cmi->c", modes, modes)
            if not all(
                np.all(np.isfinite(a)) for a in (strain, stress, physical, hourglass)
            ):
                raise ValueError(
                    f"Hex8 response overflowed in batch starting at cell {start}."
                )
            yield region, strain, stress, modes, physical, hourglass

    def iter_responses(self, displacement, *, node_map=None):
        """Stream complete cell responses without whole-mesh gathers."""
        for region, strain, stress, modes, physical, hourglass in self._iter_kinematics(
            displacement, node_map=node_map
        ):
            tensor = stress[:, np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])]
            force = self.volume[region, None, None] * np.einsum(
                "cij,caj->cai", tensor, self.average_gradient[region]
            )
            force += self.hourglass_coefficient[region, None, None] * np.einsum(
                "cam,cmi->cai", self.hourglass_modes[region], modes
            )
            if not np.all(np.isfinite(force)):
                raise ValueError(
                    f"Hex8 force overflowed in batch starting at cell {region.start}."
                )
            yield (
                region,
                UniformHexBatchResponse(strain, stress, force, physical, hourglass),
            )

    def iter_energies(self, displacement, *, node_map=None):
        """Evaluate physical/artificial energies without unused nodal forces."""
        for region, _, _, _, physical, hourglass in self._iter_kinematics(
            displacement, node_map=node_map
        ):
            yield region, physical, hourglass
