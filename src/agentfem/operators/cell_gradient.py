# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free energies built from auditable cell-gradient operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentfem.mesh import CellGradientOperator


@dataclass(frozen=True)
class CellGradientEnergyOperator:
    r"""Quadratic energy and exact first/second actions for a cell field.

    For a scalar or vector cell field :math:`v`, this object evaluates

    .. math::

       \Pi(v) = \frac{1}{2}\sum_c w_c k_c\,\lVert (Gv)_c\rVert^2,

    where ``G`` is one cached :class:`CellGradientOperator`.  Residual and
    tangent actions use the exact transpose of that same ``G``; no second
    reconstruction or dense global matrix is introduced.

    The returned residual contains local and ghost-cell entries.  A backend
    assembling a distributed global residual must reverse-scatter the ghost
    entries to their owning ranks.
    """

    gradient: CellGradientOperator
    cell_weights: np.ndarray
    stiffness: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.cell_weights, dtype=float)
        stiffness = np.asarray(self.stiffness, dtype=float)
        if weights.shape != (self.gradient.owned_cells,) or not np.all(
            np.isfinite(weights)
        ):
            raise ValueError("cell_weights must be finite with one value per owned cell.")
        if np.any(weights <= 0.0):
            raise ValueError("cell_weights must be strictly positive.")
        if stiffness.ndim == 0:
            stiffness = np.full(self.gradient.owned_cells, float(stiffness))
        if stiffness.shape != weights.shape or not np.all(np.isfinite(stiffness)):
            raise ValueError("stiffness must be finite and scalar or one value per owned cell.")
        if np.any(stiffness < 0.0):
            raise ValueError("stiffness must be nonnegative for a quadratic energy.")
        object.__setattr__(self, "cell_weights", weights.copy())
        object.__setattr__(self, "stiffness", stiffness.copy())

    @property
    def coefficients(self) -> np.ndarray:
        """Return the owned-cell integration weights times stiffness."""

        return self.cell_weights * self.stiffness

    def energy(self, cell_values) -> float:
        """Return this rank's owned-cell contribution to the energy."""

        gradients = self.gradient.apply(cell_values).gradients
        axes = tuple(range(1, gradients.ndim))
        squared_norm = np.sum(gradients * gradients, axis=axes)
        return 0.5 * float(np.dot(self.coefficients, squared_norm))

    def residual(self, cell_values) -> np.ndarray:
        """Return exact local-and-ghost derivatives of the local energy."""

        gradients = self.gradient.apply(cell_values).gradients
        coefficient_shape = (self.gradient.owned_cells,) + (1,) * (
            gradients.ndim - 1
        )
        duals = gradients * self.coefficients.reshape(coefficient_shape)
        return self.gradient.apply_adjoint(duals)

    def tangent_action(self, increment) -> np.ndarray:
        """Apply the constant exact tangent to one field increment."""

        return self.residual(increment)

    def as_dict(self) -> dict[str, object]:
        """Return inspectable numerical semantics without dense coefficients."""

        return {
            "kind": "cell_gradient_energy_operator",
            "energy": "0.5 * sum(cell_weight * stiffness * norm(Gv)^2)",
            "residual": "G_transpose * cell_weight * stiffness * Gv",
            "tangent": "same_linear_action_as_residual",
            "owned_cells": self.gradient.owned_cells,
            "total_local_and_ghost_cells": self.gradient.total_cells,
            "minimum_stiffness": float(np.min(self.stiffness)),
            "maximum_stiffness": float(np.max(self.stiffness)),
            "assembly": "local_energy_and_local_plus_ghost_residual",
            "mpi_requirement": "reverse_scatter_ghost_residual_to_owners",
            "gradient_operator": self.gradient.as_dict(),
        }


def cell_gradient_energy(
    gradient: CellGradientOperator,
    *,
    cell_weights,
    stiffness=1.0,
) -> CellGradientEnergyOperator:
    """Create one quadratic matrix-free energy from a cached gradient."""

    return CellGradientEnergyOperator(
        gradient=gradient,
        cell_weights=cell_weights,
        stiffness=stiffness,
    )


__all__ = ["CellGradientEnergyOperator", "cell_gradient_energy"]
