# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Private radial-tool Hessian envelope on an explicitly bounded domain.

Fixed reference trace, frictionless penalty, prescribed rigid tool only.
Negative geometric curvature is retained, never made positive by mass scaling.
This is not a nonlinear trajectory stability certificate.
"""

from dataclasses import dataclass

import numpy as np

from .._penalty_stability import penalty_stencil_row_sums
from .contact_trace import ContactTrace
from .rigid import RigidCylinderSurface, RigidSphereSurface, _readonly_array


@dataclass(frozen=True)
class RadialContactSpectralEnvelope:
    positive_rows: np.ndarray
    negative_rows: np.ndarray
    minimum_radial_distance: float
    geometry_fingerprint: str

    def __post_init__(self):
        for name in ("positive_rows", "negative_rows"):
            object.__setattr__(self, name, _readonly_array(getattr(self, name)))

    def summary(self):
        return {
            "positive_spectral_upper_bound": float(self.positive_rows.max(initial=0)),
            "negative_spectral_magnitude_bound": float(
                self.negative_rows.max(initial=0)
            ),
            "minimum_radial_distance": self.minimum_radial_distance,
            "geometry_fingerprint": self.geometry_fingerprint,
            "scope": "fixed_reference_trace_radial_tool_domain",
            "whole_system_stability_certificate": False,
            "negative_curvature_removed_by_mass_scaling": False,
        }


def radial_contact_spectral_envelope(
    trace, nodal_mass, *, surface, normal_penalty, minimum_radial_distance
):
    """Return local rows bounding -L <= M^-1/2 K M^-1/2 <= U.

    For active radial penalty E=k/2*(r-R)^2, the radial Hessian
    eigenvalue is k and the curved tangential eigenvalue k*(1-R/r).
    A cylinder additionally has a zero axial eigenvalue. Exterior contact
    has negative tangential curvature; cavity contact (r>R) does not.
    At activation, the normal generalized derivative lies in [0,k].

    The caller must enforce r>=minimum_radial_distance throughout its admitted
    path, reject singular/rim projections and reverse-add ghost rows before a
    global maximum. No area-change, friction or deformable-master term exists.
    """
    if (
        not isinstance(trace, ContactTrace)
        or trace.measure_configuration != "reference"
    ):
        raise ValueError("Radial envelope requires a fixed reference ContactTrace.")
    if not isinstance(surface, (RigidCylinderSurface, RigidSphereSurface)):
        raise TypeError("Radial envelope requires a sphere/circle or lateral cylinder.")
    radius = surface.radius
    floor = np.asarray(minimum_radial_distance)
    if (
        floor.ndim != 0
        or floor.dtype.kind not in "fiu"
        or not np.isfinite(floor)
        or floor <= 0
        or floor > radius
    ):
        raise ValueError(
            "Radial domain floor must be positive and no greater than radius."
        )
    penalty = np.asarray(normal_penalty)
    if (
        penalty.dtype.kind not in "fiu"
        or penalty.shape not in ((), (trace.point_count,))
        or not np.isfinite(penalty).all()
        or np.any(penalty <= 0)
    ):
        raise ValueError("Radial penalty must be positive finite and point-compatible.")
    mass = np.asarray(nodal_mass)
    if mass.ndim != 2 or mass.shape[1] != surface.dimension:
        raise ValueError("Radial mass dimension differs from the surface.")
    negative_factor = (
        max(0.0, radius / float(floor) - 1)
        if surface.admissible_side == "exterior"
        else 0.0
    )
    upper = penalty_stencil_row_sums(
        trace.node_ids, trace.shape_values, trace.weights, penalty, mass
    )
    lower = penalty_stencil_row_sums(
        trace.node_ids,
        trace.shape_values,
        trace.weights,
        penalty * negative_factor,
        mass,
    )
    return RadialContactSpectralEnvelope(
        upper, lower, float(floor), surface.geometry_fingerprint
    )


def validate_radial_envelope_projection(envelope, surface, projection):
    """Reject out-of-domain endpoints; endpoints alone do not certify a path."""
    if (
        projection.geometry_fingerprint != envelope.geometry_fingerprint
        or surface.geometry_fingerprint != envelope.geometry_fingerprint
    ):
        raise ValueError("Radial envelope/projection geometry identities differ.")
    if not np.all(projection.valid):
        raise ValueError(
            "Radial envelope requires valid nonsingular lateral projections."
        )
    side = 1 if surface.admissible_side == "exterior" else -1
    radial_distance = surface.radius + side * projection.signed_gaps
    if not np.isfinite(radial_distance).all() or np.any(
        radial_distance < envelope.minimum_radial_distance
    ):
        raise ValueError("Projection leaves the declared radial envelope domain.")
