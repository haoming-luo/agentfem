# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Artificial volumetric damping for the finite uniform-strain Hex8 operator.

This is not constitutive dissipation. Current density, wave speed and length
must be explicit; no mass scaling or stable increment is inferred here.
"""

from dataclasses import dataclass
from math import fsum

import numpy as np

from ._uniform_hex import _frozen


@dataclass(frozen=True)
class BulkViscosityResponse:
    internal_force: np.ndarray
    artificial_power: np.ndarray
    volumetric_rate: np.ndarray
    isotropic_stress: np.ndarray
    mass_scaled_damping_bound: np.ndarray
    damping_row_majorant: np.ndarray | None = None


def _cell_increment_work(force, cell_nodes, nodal_increment, *, chunk_size=4096):
    """Contract admitted cell forces with nodal motion using bounded scratch.

    Returns signed work and absolute-product sum for the caller's roundoff
    check. Does not substitute power for force-displacement conjugacy.
    """
    force = np.asarray(force)
    nodes = np.asarray(cell_nodes)
    increment = np.asarray(nodal_increment)
    if (
        force.shape != nodes.shape + (3,)
        or nodes.ndim != 2
        or nodes.shape[1] != 8
        or nodes.dtype.kind not in "iu"
        or increment.ndim != 2
        or increment.shape[1] != 3
        or np.iscomplexobj(force)
        or np.iscomplexobj(increment)
        or isinstance(chunk_size, bool)
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size < 1
    ):
        raise ValueError("Cell work requires real Hex8 forces, node map and nodal increments.")
    increment = np.asarray(increment, dtype=float)
    signed_parts, absolute_parts = [], []
    for start in range(0, len(nodes), chunk_size):
        selected = nodes[start:start + chunk_size]
        if np.any(selected < 0) or np.any(selected >= len(increment)):
            raise ValueError("Cell work node index is out of range.")
        products = increment[selected]
        products *= force[start:start + chunk_size]
        if not np.isfinite(products).all():
            raise ValueError("Nonfinite cell increment work.")
        signed_parts.append(float(products.sum()))
        np.abs(products, out=products)
        absolute_parts.append(float(products.sum()))
    work, absolute = fsum(signed_parts), fsum(absolute_parts)
    if not np.isfinite([work, absolute]).all():
        raise ValueError("Cell increment work overflowed.")
    return work, absolute


def bulk_viscosity_response(
    cells,
    displacement,
    velocity,
    *,
    density,
    wave_speed,
    characteristic_length,
    linear=0.06,
    quadratic=1.2,
    include_row_majorant=False,
):
    """Tension-positive stress q I, with q*div(v) >= 0.

    q = rho*b1*c*L*theta - rho*(b2*L*min(theta,0))**2.
    The returned bound is per-element M^-1/2 D M^-1/2, not a global time-step
    certificate. D differentiates damping force w.r.t. velocity at fixed F.
    """
    if np.iscomplexobj(displacement):
        raise ValueError("Kinematics must be real.")
    gradient = cells.deformation_gradient(displacement)
    return _bulk_viscosity_from_checked_gradient(
        cells,
        gradient,
        velocity,
        density=density,
        wave_speed=wave_speed,
        characteristic_length=characteristic_length,
        linear=linear,
        quadratic=quadratic,
        include_row_majorant=include_row_majorant,
    )


def _bulk_viscosity_from_checked_gradient(
    cells,
    gradient,
    velocity,
    *,
    density,
    wave_speed,
    characteristic_length,
    linear=0.06,
    quadratic=1.2,
    include_row_majorant=False,
):
    """Continue an admitted, unchanged cell evaluation inside one trial scope.

    This does not admit cell geometry: the caller must already have performed
    the full Jacobian check on its frozen configuration. Standalone callers use
    bulk_viscosity_response(), which never bypasses that check.
    """
    n = len(cells.volume)

    def values(value, name, positive=False):
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real.")
        array = np.asarray(value, dtype=float)
        if array.shape not in ((), (n,)) or not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite scalar or cellwise.")
        if np.any(array <= 0 if positive else array < 0):
            raise ValueError(f"Invalid {name} sign.")
        return np.broadcast_to(array, (n,))

    rho = values(density, "current density", True)
    speed = values(wave_speed, "wave speed", True)
    length = values(characteristic_length, "current characteristic length", True)
    b1, b2 = (
        values(linear, "linear viscosity"),
        values(quadratic, "quadratic viscosity"),
    )
    if np.iscomplexobj(velocity) or np.iscomplexobj(gradient):
        raise ValueError("Kinematics must be real.")
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != (n, 8, 3) or not np.isfinite(velocity).all():
        raise ValueError("Velocity must be finite (cells, 8, 3).")
    f = np.asarray(gradient, dtype=float)
    if (
        f.shape != (n, 3, 3)
        or not np.isfinite(f).all()
    ):
        raise ValueError(
            "Admitted gradients must be finite and orientation-preserving."
        )
    jacobian = np.linalg.det(f)
    if np.any(jacobian <= 0) or not np.isfinite(jacobian).all():
        raise ValueError("Admitted gradients require positive finite determinants.")
    gradients = cells.average_gradient @ np.linalg.inv(f)
    theta = np.einsum("nai,nai->n", gradients, velocity)
    compression = np.minimum(theta, 0)
    linear_factor = rho * b1 * speed * length
    quadratic_factor = rho * (b2 * length) ** 2
    q = linear_factor * theta - quadratic_factor * compression**2
    eta = linear_factor - 2 * quadratic_factor * compression
    volume = cells.volume * jacobian
    force = volume[:, None, None] * q[:, None, None] * gradients
    power = volume * q * theta
    bound = (
        volume * eta * np.sum(gradients**2 / cells.lumped_mass[..., None], axis=(1, 2))
    )
    if not isinstance(include_row_majorant, (bool, np.bool_)):
        raise ValueError("include_row_majorant must be boolean.")
    # At fixed configuration D_e = V*eta*vec(grad) outer vec(grad).
    # Return unscaled (cell,node,component) rows only when an inertia policy
    # needs them; the common force path retains its existing memory footprint.
    rows = None
    if include_row_majorant:
        absolute = np.abs(gradients)
        rows = (volume * eta * absolute.sum(axis=(1, 2)))[:, None, None] * absolute
        if not np.isfinite(rows).all():
            raise ValueError("Nonfinite damping row majorant.")
        rows = _frozen(rows)
    arrays = (force, power, theta, q, bound)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Nonfinite bulk viscosity response.")
    return BulkViscosityResponse(*(_frozen(array) for array in arrays), rows)
