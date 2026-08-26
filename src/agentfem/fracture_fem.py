"""DOLFINx lowering for the solver-neutral two-dimensional LEFM integrals."""

from __future__ import annotations

from math import isfinite
from typing import Mapping

import basix
from dolfinx import fem
from mpi4py import MPI
import numpy as np
import ufl

from .fracture_geometry import (
    CrackSet2D,
    CrackTip2D,
    LinearElasticFractureMaterial2D,
    StressIntensityReport,
)
from .fracture_integrals import (
    InteractionIntegralSamples2D,
    WilliamsField2D,
    interaction_integral,
    interaction_integral_report,
)


_EMPTY_ANNULUS_MESSAGE = "No owned quadrature point lies in the integration annulus."


def dolfinx_interaction_integral_samples(
    domain,
    actual_stress,
    actual_displacement_gradient,
    *,
    tip: CrackTip2D,
    auxiliary: WilliamsField2D,
    inner_radius: float,
    outer_radius: float,
    quadrature_degree: int = 6,
    metadata: Mapping[str, object] | None = None,
) -> InteractionIntegralSamples2D:
    """Lower rank-local DOLFINx fields to the common interaction samples.

    ``actual_stress`` and ``actual_displacement_gradient`` are UFL expressions
    in global coordinates. Only owned cells are sampled, so a caller must use
    :func:`dolfinx_interaction_integral` for the MPI-global value.
    """

    if int(domain.topology.dim) != 2 or int(domain.geometry.dim) != 2:
        raise ValueError("The first DOLFINx interaction adapter requires a 2D mesh.")
    if tuple(getattr(actual_stress, "ufl_shape", ())) != (2, 2):
        raise ValueError("actual_stress must be a two-dimensional 2x2 UFL tensor.")
    if tuple(getattr(actual_displacement_gradient, "ufl_shape", ())) != (2, 2):
        raise ValueError(
            "actual_displacement_gradient must be a 2x2 UFL tensor."
        )
    if not isinstance(tip, CrackTip2D):
        raise TypeError("tip must be a CrackTip2D record.")
    if not isinstance(auxiliary, WilliamsField2D):
        raise TypeError("auxiliary must be a WilliamsField2D record.")
    if auxiliary.tip.tip_id != tip.tip_id:
        raise ValueError("Auxiliary field and integration domain must share one tip.")
    inner = float(inner_radius)
    outer = float(outer_radius)
    if not isfinite(inner) or not isfinite(outer) or not 0.0 < inner < outer:
        raise ValueError("Radii must satisfy 0 < inner_radius < outer_radius.")
    degree = int(quadrature_degree)
    if degree < 1:
        raise ValueError("quadrature_degree must be positive.")

    points, reference_weights = basix.make_quadrature(domain.basix_cell(), degree)
    cell_map = domain.topology.index_map(domain.topology.dim)
    cells = np.arange(int(cell_map.size_local), dtype=np.int32)
    count = len(cells)
    coordinates = np.asarray(
        fem.Expression(ufl.SpatialCoordinate(domain), points).eval(domain, cells),
        dtype=float,
    ).reshape((-1, 2))
    determinants = np.abs(
        np.asarray(
            fem.Expression(ufl.JacobianDeterminant(domain), points).eval(
                domain, cells
            ),
            dtype=float,
        ).reshape(-1)
    )
    stress = np.asarray(
        fem.Expression(actual_stress, points).eval(domain, cells), dtype=float
    ).reshape((-1, 2, 2))
    gradient = np.asarray(
        fem.Expression(actual_displacement_gradient, points).eval(domain, cells),
        dtype=float,
    ).reshape((-1, 2, 2))
    weights = determinants * np.tile(np.asarray(reference_weights, dtype=float), count)

    rotation = np.asarray((tip.extension_direction, tip.normal), dtype=float).T
    local_coordinates = (
        coordinates - np.asarray(tip.point, dtype=float)
    ) @ rotation
    radius = np.linalg.norm(local_coordinates, axis=1)
    selected = (radius > inner) & (radius < outer)
    if not np.any(selected):
        raise ValueError(_EMPTY_ANNULUS_MESSAGE)
    coordinates = coordinates[selected]
    local_coordinates = local_coordinates[selected]
    radius = radius[selected]
    q_gradient = -local_coordinates / radius[:, None] / (outer - inner)

    def to_local(values):
        return np.einsum(
            "ia,nab,bj->nij", rotation.T, values, rotation
        )

    return InteractionIntegralSamples2D(
        actual_stress=to_local(stress[selected]),
        actual_displacement_gradient=to_local(gradient[selected]),
        auxiliary_stress=to_local(auxiliary.stress(coordinates)),
        auxiliary_displacement_gradient=to_local(
            auxiliary.displacement_gradient(coordinates)
        ),
        q_gradient=q_gradient,
        weights=weights[selected],
        metadata={
            "provider": "dolfinx",
            "tip_id": tip.tip_id,
            "inner_radius": inner,
            "outer_radius": outer,
            "quadrature_degree": degree,
            "owned_cells": count,
            **dict(metadata or {}),
        },
    )


def dolfinx_interaction_integral(
    domain,
    actual_stress,
    actual_displacement_gradient,
    **options,
) -> float:
    """Evaluate one MPI-global DOLFINx interaction integral."""

    try:
        samples = dolfinx_interaction_integral_samples(
            domain,
            actual_stress,
            actual_displacement_gradient,
            **options,
        )
    except ValueError as exc:
        if str(exc) != _EMPTY_ANNULUS_MESSAGE:
            raise
        local = 0.0
        locally_active = 0
    else:
        local = interaction_integral(samples)
        locally_active = 1
    globally_active = int(domain.comm.allreduce(locally_active, op=MPI.SUM))
    if globally_active == 0:
        raise ValueError("No global quadrature point lies in the integration annulus.")
    return float(domain.comm.allreduce(local, op=MPI.SUM))


def dolfinx_interaction_integral_report(
    domain,
    actual_stress,
    actual_displacement_gradient,
    *,
    crack: CrackSet2D,
    tip_id: str,
    material: LinearElasticFractureMaterial2D,
    integration_radii,
    inner_radius_fraction: float = 0.25,
    quadrature_degree: int = 6,
    relative_path_tolerance: float = 0.03,
    metadata: Mapping[str, object] | None = None,
) -> StressIntensityReport:
    """Extract mixed-mode SIFs from DOLFINx fields over multiple tip domains."""

    fraction = float(inner_radius_fraction)
    if not isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("inner_radius_fraction must satisfy 0 < value < 1.")
    tip = crack.tip(tip_id)
    mode_i_auxiliary = WilliamsField2D(tip, material, k_i=1.0)
    mode_ii_auxiliary = WilliamsField2D(tip, material, k_ii=1.0)
    radii = tuple(float(item) for item in integration_radii)
    mode_i = []
    mode_ii = []
    for outer in radii:
        common = dict(
            tip=tip,
            inner_radius=fraction * outer,
            outer_radius=outer,
            quadrature_degree=quadrature_degree,
        )
        mode_i.append(
            dolfinx_interaction_integral(
                domain,
                actual_stress,
                actual_displacement_gradient,
                auxiliary=mode_i_auxiliary,
                **common,
            )
        )
        mode_ii.append(
            dolfinx_interaction_integral(
                domain,
                actual_stress,
                actual_displacement_gradient,
                auxiliary=mode_ii_auxiliary,
                **common,
            )
        )
    return interaction_integral_report(
        crack=crack,
        tip_id=tip_id,
        integration_radii=radii,
        mode_i_integrals=mode_i,
        mode_ii_integrals=mode_ii,
        material=material,
        relative_path_tolerance=relative_path_tolerance,
        metadata={
            "provider": "dolfinx",
            "inner_radius_fraction": fraction,
            "quadrature_degree": int(quadrature_degree),
            **dict(metadata or {}),
        },
    )


__all__ = [
    "dolfinx_interaction_integral",
    "dolfinx_interaction_integral_report",
    "dolfinx_interaction_integral_samples",
]
