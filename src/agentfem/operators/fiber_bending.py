# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Neighbour-reconstructed bending energy for independent fibre directions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentfem.mesh import CellGradientOperator


def _owned_parameter(value, owned: int, *, name: str, nonnegative: bool) -> np.ndarray:
    selected = np.asarray(value, dtype=float)
    if selected.ndim == 0:
        selected = np.full(owned, float(selected))
    if selected.shape != (owned,) or not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must be finite and scalar or one value per owned cell.")
    if nonnegative and np.any(selected < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return selected.copy()


@dataclass(frozen=True)
class FiberDirectionBendingResponse:
    """Curvature energy and exact direction/surface-tangent duals."""

    energy: float
    in_plane_curvature: np.ndarray
    normal_curvature: np.ndarray
    residual: np.ndarray
    surface_tangent_residual: np.ndarray

    def __post_init__(self) -> None:
        in_plane = np.asarray(self.in_plane_curvature, dtype=float)
        normal = np.asarray(self.normal_curvature, dtype=float)
        residual = np.asarray(self.residual, dtype=float)
        tangent_residual = np.asarray(self.surface_tangent_residual, dtype=float)
        owned = in_plane.size
        if not np.isfinite(self.energy):
            raise ValueError("energy must be finite.")
        if in_plane.shape != (owned,) or not np.all(np.isfinite(in_plane)):
            raise ValueError("in_plane_curvature must be a finite owned-cell array.")
        if normal.shape != (owned,) or not np.all(np.isfinite(normal)):
            raise ValueError("normal_curvature must match in_plane_curvature.")
        if residual.ndim != 2 or residual.shape[1:] != (3,) or not np.all(
            np.isfinite(residual)
        ):
            raise ValueError("residual must have finite shape (cells, 3).")
        if residual.shape[0] < owned:
            raise ValueError("residual must include every owned cell.")
        if tangent_residual.shape != (owned, 3, 2) or not np.all(
            np.isfinite(tangent_residual)
        ):
            raise ValueError(
                "surface_tangent_residual must have finite shape "
                "(owned_cells, 3, 2)."
            )
        object.__setattr__(self, "in_plane_curvature", in_plane.copy())
        object.__setattr__(self, "normal_curvature", normal.copy())
        object.__setattr__(self, "residual", residual.copy())
        object.__setattr__(
            self, "surface_tangent_residual", tangent_residual.copy()
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "fiber_direction_bending_response",
            "energy": self.energy,
            "owned_cells": int(self.in_plane_curvature.size),
            "maximum_absolute_in_plane_curvature": float(
                np.max(np.abs(self.in_plane_curvature))
            ),
            "maximum_absolute_normal_curvature": float(
                np.max(np.abs(self.normal_curvature))
            ),
            "residual_shape": list(self.residual.shape),
            "surface_tangent_residual_shape": list(
                self.surface_tangent_residual.shape
            ),
        }


@dataclass(frozen=True)
class FiberDirectionBendingOperator:
    r"""Exact first variation of a two-channel fibre-curvature energy.

    The primary values are nonzero three-component fibre directions on local
    and ghost cells.  They are normalized before reconstruction, making the
    energy invariant to positive pointwise rescaling.  Current surface
    tangents are explicit inputs and the directions must lie in their span on
    owned cells.  The response differentiates the energy with respect to both
    inputs; the matrix-free second action currently differentiates only the
    direction path while holding tangents fixed.

    This is the bending contribution for an *independent direction field*.
    It is not yet a displacement shell operator: compatibility forces,
    boundary moments, MPI reverse scatter and the complete displacement-level
    nonlinear second variation remain separate owners.
    """

    gradient: CellGradientOperator
    current_tangents: np.ndarray
    cell_weights: np.ndarray
    in_plane_stiffness: np.ndarray
    normal_stiffness: np.ndarray
    reference_in_plane_curvature: np.ndarray
    reference_normal_curvature: np.ndarray

    def __post_init__(self) -> None:
        owned = self.gradient.owned_cells
        tangents = np.asarray(self.current_tangents, dtype=float)
        if tangents.shape != (owned, 3, 2) or not np.all(np.isfinite(tangents)):
            raise ValueError("current_tangents must have shape (owned_cells, 3, 2).")
        for cell, item in enumerate(tangents):
            if np.linalg.matrix_rank(item) != 2:
                raise ValueError(f"current_tangents[{cell}] must have rank two.")
        weights = _owned_parameter(
            self.cell_weights, owned, name="cell_weights", nonnegative=False
        )
        if np.any(weights <= 0.0):
            raise ValueError("cell_weights must be strictly positive.")
        object.__setattr__(self, "current_tangents", tangents.copy())
        object.__setattr__(self, "cell_weights", weights)
        for name in ("in_plane_stiffness", "normal_stiffness"):
            object.__setattr__(
                self,
                name,
                _owned_parameter(
                    getattr(self, name), owned, name=name, nonnegative=True
                ),
            )
        for name in (
            "reference_in_plane_curvature",
            "reference_normal_curvature",
        ):
            object.__setattr__(
                self,
                name,
                _owned_parameter(
                    getattr(self, name), owned, name=name, nonnegative=False
                ),
            )

    def evaluate(self, directions) -> FiberDirectionBendingResponse:
        """Evaluate energy, curvature, and exact direction/tangent duals."""

        response, _ = self._evaluate_state(directions)
        return response

    def tangent_action(self, directions, increment) -> np.ndarray:
        """Apply the exact nonlinear direction-field tangent to an increment."""

        _, action = self._evaluate_state(directions, increment=increment)
        return action

    def _evaluate_state(self, directions, *, increment=None):
        """Evaluate the common first- and optional second-variation kernel."""

        raw = np.asarray(directions, dtype=float)
        if raw.shape != (self.gradient.total_cells, 3) or not np.all(
            np.isfinite(raw)
        ):
            raise ValueError(
                "directions must have shape (local_and_ghost_cells, 3)."
            )
        norms = np.linalg.norm(raw, axis=1)
        if np.any(norms <= np.finfo(float).eps):
            raise ValueError("directions must have nonzero length on every cell.")
        unit = raw / norms[:, None]
        linearized = increment is not None
        if linearized:
            increment = np.asarray(increment, dtype=float)
            if increment.shape != raw.shape or not np.all(np.isfinite(increment)):
                raise ValueError("increment must be finite with the direction shape.")
            norm_increments = np.sum(unit * increment, axis=1)
            unit_increments = (
                increment - unit * norm_increments[:, None]
            ) / norms[:, None]
        else:
            norm_increments = None
            unit_increments = None
        reconstructed = self.gradient.apply(unit)
        gradients = reconstructed.gradients
        gradient_increments = (
            self.gradient.apply(unit_increments).gradients if linearized else None
        )
        owned = self.gradient.owned_cells
        in_plane = np.empty(owned, dtype=float)
        normal_curvature = np.empty(owned, dtype=float)
        gradient_duals = np.empty_like(gradients)
        explicit = np.zeros((self.gradient.total_cells, 3), dtype=float)
        surface_tangent_residual = np.empty((owned, 3, 2), dtype=float)
        if linearized:
            gradient_dual_increments = np.empty_like(gradients)
            explicit_increments = np.zeros_like(explicit)
        energy = 0.0

        for output_cell, stencil in enumerate(self.gradient.stencils):
            surface = self.current_tangents[output_cell]
            direction = unit[stencil.cell]
            surface_normal = np.cross(surface[:, 0], surface[:, 1])
            surface_normal /= np.linalg.norm(surface_normal)
            if abs(float(np.dot(direction, surface_normal))) > 1.0e-9:
                raise ValueError(
                    f"directions[{stencil.cell}] must be tangent to the surface."
                )
            metric = surface.T @ surface
            dual_tangents = np.linalg.solve(metric, surface.T)
            coordinates = dual_tangents @ direction
            if not np.allclose(
                surface @ coordinates, direction, atol=1.0e-9, rtol=1.0e-9
            ):
                raise ValueError(
                    f"directions[{stencil.cell}] must lie in the tangent span."
                )
            if linearized:
                direction_increment = unit_increments[stencil.cell]
                if abs(float(np.dot(direction_increment, surface_normal))) > 1.0e-9:
                    raise ValueError(
                        f"increment[{stencil.cell}] must be tangent to the surface."
                    )
                coordinate_increment = dual_tangents @ direction_increment
            directional_derivative = gradients[output_cell] @ coordinates
            in_plane_normal = np.cross(surface_normal, direction)
            k_in_plane = float(np.dot(directional_derivative, in_plane_normal))
            k_normal = float(np.dot(directional_derivative, surface_normal))
            in_plane[output_cell] = k_in_plane
            normal_curvature[output_cell] = k_normal
            delta_in_plane = (
                k_in_plane - self.reference_in_plane_curvature[output_cell]
            )
            delta_normal = (
                k_normal - self.reference_normal_curvature[output_cell]
            )
            weighted_in_plane = (
                self.cell_weights[output_cell]
                * self.in_plane_stiffness[output_cell]
                * delta_in_plane
            )
            weighted_normal = (
                self.cell_weights[output_cell]
                * self.normal_stiffness[output_cell]
                * delta_normal
            )
            energy += 0.5 * (
                weighted_in_plane * delta_in_plane
                + weighted_normal * delta_normal
            )
            curvature_dual = (
                weighted_in_plane * in_plane_normal
                + weighted_normal * surface_normal
            )
            gradient_duals[output_cell] = np.outer(curvature_dual, coordinates)
            coordinate_dual = gradients[output_cell].T @ curvature_dual
            coordinate_tangent_residual = -np.outer(
                dual_tangents.T @ coordinate_dual,
                coordinates,
            )
            surface_normal_dual = (
                weighted_in_plane * np.cross(direction, directional_derivative)
                + weighted_normal * directional_derivative
            )
            surface_normal_dual -= (
                np.dot(surface_normal_dual, surface_normal) * surface_normal
            )
            surface_jacobian = np.linalg.norm(
                np.cross(surface[:, 0], surface[:, 1])
            )
            normal_dual = surface_normal_dual / surface_jacobian
            normal_tangent_residual = np.column_stack(
                (
                    np.cross(surface[:, 1], normal_dual),
                    np.cross(normal_dual, surface[:, 0]),
                )
            )
            surface_tangent_residual[output_cell] = (
                coordinate_tangent_residual + normal_tangent_residual
            )
            local_in_plane_gradient = np.cross(
                directional_derivative, surface_normal
            ) + dual_tangents.T @ gradients[output_cell].T @ in_plane_normal
            local_normal_gradient = (
                dual_tangents.T @ gradients[output_cell].T @ surface_normal
            )
            explicit[stencil.cell] += (
                weighted_in_plane * local_in_plane_gradient
                + weighted_normal * local_normal_gradient
            )
            if linearized:
                gradient_increment = gradient_increments[output_cell]
                directional_derivative_increment = (
                    gradient_increment @ coordinates
                    + gradients[output_cell] @ coordinate_increment
                )
                in_plane_normal_increment = np.cross(
                    surface_normal, direction_increment
                )
                in_plane_curvature_increment = float(
                    np.dot(directional_derivative_increment, in_plane_normal)
                    + np.dot(
                        directional_derivative, in_plane_normal_increment
                    )
                )
                normal_curvature_increment = float(
                    np.dot(directional_derivative_increment, surface_normal)
                )
                weighted_in_plane_increment = (
                    self.cell_weights[output_cell]
                    * self.in_plane_stiffness[output_cell]
                    * in_plane_curvature_increment
                )
                weighted_normal_increment = (
                    self.cell_weights[output_cell]
                    * self.normal_stiffness[output_cell]
                    * normal_curvature_increment
                )
                curvature_dual_increment = (
                    weighted_in_plane_increment * in_plane_normal
                    + weighted_in_plane * in_plane_normal_increment
                    + weighted_normal_increment * surface_normal
                )
                gradient_dual_increments[output_cell] = (
                    np.outer(curvature_dual_increment, coordinates)
                    + np.outer(curvature_dual, coordinate_increment)
                )
                local_in_plane_gradient_increment = np.cross(
                    directional_derivative_increment, surface_normal
                ) + dual_tangents.T @ (
                    gradient_increment.T @ in_plane_normal
                    + gradients[output_cell].T @ in_plane_normal_increment
                )
                local_normal_gradient_increment = (
                    dual_tangents.T @ gradient_increment.T @ surface_normal
                )
                explicit_increments[stencil.cell] += (
                    weighted_in_plane_increment * local_in_plane_gradient
                    + weighted_in_plane * local_in_plane_gradient_increment
                    + weighted_normal_increment * local_normal_gradient
                    + weighted_normal * local_normal_gradient_increment
                )

        unit_residual = self.gradient.apply_adjoint(gradient_duals) + explicit
        residual = np.empty_like(unit_residual)
        if linearized:
            unit_residual_increment = (
                self.gradient.apply_adjoint(gradient_dual_increments)
                + explicit_increments
            )
            tangent_action = np.empty_like(unit_residual)
        for cell, direction in enumerate(unit):
            projector = np.eye(3) - np.outer(direction, direction)
            residual[cell] = projector @ unit_residual[cell] / norms[cell]
            if linearized:
                direction_increment = unit_increments[cell]
                projector_increment = -(
                    np.outer(direction_increment, direction)
                    + np.outer(direction, direction_increment)
                )
                normalization_increment = (
                    projector_increment / norms[cell]
                    - projector * norm_increments[cell] / norms[cell] ** 2
                )
                tangent_action[cell] = (
                    normalization_increment @ unit_residual[cell]
                    + projector @ unit_residual_increment[cell] / norms[cell]
                )
        response = FiberDirectionBendingResponse(
            energy=float(energy),
            in_plane_curvature=in_plane,
            normal_curvature=normal_curvature,
            residual=residual,
            surface_tangent_residual=surface_tangent_residual,
        )
        return response, tangent_action if linearized else None

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "fiber_direction_bending_operator",
            "primary_field": "independent_nonzero_cell_fiber_directions",
            "normalization": "pointwise_before_gradient_reconstruction",
            "energy_channels": ("in_plane_curvature", "normal_curvature"),
            "first_variation": "exact_direction_and_surface_tangent_duals",
            "surface_tangent_dual": "exact_direct_energy_derivative",
            "nonlinear_tangent": "exact_matrix_free_action_for_fixed_surface_tangents",
            "mpi_requirement": "reverse_scatter_ghost_residual_to_owners",
            "scope": "bending_contribution_not_complete_shell_equilibrium",
            "gradient_operator": self.gradient.as_dict(),
        }


def fiber_direction_bending(
    gradient: CellGradientOperator,
    *,
    current_tangents,
    cell_weights,
    in_plane_stiffness=0.0,
    normal_stiffness=0.0,
    reference_in_plane_curvature=0.0,
    reference_normal_curvature=0.0,
) -> FiberDirectionBendingOperator:
    """Create an independent-direction fibre-bending energy operator."""

    return FiberDirectionBendingOperator(
        gradient=gradient,
        current_tangents=current_tangents,
        cell_weights=cell_weights,
        in_plane_stiffness=in_plane_stiffness,
        normal_stiffness=normal_stiffness,
        reference_in_plane_curvature=reference_in_plane_curvature,
        reference_normal_curvature=reference_normal_curvature,
    )


__all__ = [
    "FiberDirectionBendingOperator",
    "FiberDirectionBendingResponse",
    "fiber_direction_bending",
]
