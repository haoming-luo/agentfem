# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Displacement-level composition of convected fibre bending operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agentfem.mesh import CellGradientOperator

from .cell_transfer import ConvectedCellFiberOperator
from .fiber_bending import FiberDirectionBendingResponse, fiber_direction_bending


@dataclass(frozen=True)
class DisplacementFiberBendingOperator:
    """Exact energy, residual, and tangent for one convected fibre family."""

    kinematics: ConvectedCellFiberOperator
    gradient: CellGradientOperator
    cell_weights: object
    in_plane_stiffness: object = 0.0
    normal_stiffness: object = 0.0
    reference_in_plane_curvature: object = 0.0
    reference_normal_curvature: object = 0.0

    def __post_init__(self) -> None:
        if self.kinematics.total_cells != self.gradient.total_cells:
            raise ValueError(
                "kinematics and gradient must address the same local and "
                "ghost cells."
            )
        source_mesh = self.kinematics.transfer.source_space.mesh
        cell_map = source_mesh.topology.index_map(source_mesh.topology.dim)
        if int(cell_map.size_local) != self.gradient.owned_cells:
            raise ValueError(
                "kinematics and gradient must share one owned-cell partition."
            )
        probe = np.broadcast_to(
            self.kinematics.reference_tangents[0],
            (self.gradient.owned_cells, 3, 2),
        )
        self._bending(probe)

    def _bending(self, current_tangents):
        return fiber_direction_bending(
            self.gradient,
            current_tangents=current_tangents,
            cell_weights=self.cell_weights,
            in_plane_stiffness=self.in_plane_stiffness,
            normal_stiffness=self.normal_stiffness,
            reference_in_plane_curvature=self.reference_in_plane_curvature,
            reference_normal_curvature=self.reference_normal_curvature,
        )

    def _evaluate(self, displacement):
        state = self.kinematics.apply(displacement)
        bending = self._bending(
            state.current_tangents[: self.gradient.owned_cells]
        )
        response = bending.evaluate(state.direction)
        return state, bending, response

    def response(self, displacement) -> FiberDirectionBendingResponse:
        """Return curvature, local energy, and the two cell-level duals."""

        return self._evaluate(displacement)[2]

    def energy(self, displacement) -> float:
        """Return this rank's owned-cell bending-energy contribution."""

        return self.response(displacement).energy

    def residual(self, displacement):
        """Return the exact displacement-space bending residual."""

        state, _, response = self._evaluate(displacement)
        tangent_duals = np.zeros_like(state.current_tangents)
        tangent_duals[: self.gradient.owned_cells] = (
            response.surface_tangent_residual
        )
        return self.kinematics.apply_adjoint(
            displacement,
            direction_duals=response.residual,
            tangent_duals=tangent_duals,
        )

    def tangent_action(self, displacement, increment):
        """Apply the exact displacement-level consistent tangent."""

        state, bending, response = self._evaluate(displacement)
        kinematic_increment = self.kinematics.directional_derivative(
            displacement, increment
        )
        response_increment = bending.linearized_response(
            state.direction,
            kinematic_increment.direction_increment,
            surface_tangent_increment=kinematic_increment.tangent_increment[
                : self.gradient.owned_cells
            ],
        )
        tangent_duals = np.zeros_like(state.current_tangents)
        tangent_duals[: self.gradient.owned_cells] = (
            response.surface_tangent_residual
        )
        tangent_dual_increments = np.zeros_like(state.current_tangents)
        tangent_dual_increments[: self.gradient.owned_cells] = (
            response_increment.surface_tangent_residual_increment
        )
        return self.kinematics.apply_adjoint_derivative(
            displacement,
            increment,
            direction_duals=response.residual,
            direction_dual_increments=(
                response_increment.direction_residual_increment
            ),
            tangent_duals=tangent_duals,
            tangent_dual_increments=tangent_dual_increments,
        )

    def as_dict(self) -> dict[str, object]:
        """Return inspectable composition and ownership semantics."""

        return {
            "kind": "displacement_fiber_bending_operator",
            "primary_field": "three_component_surface_displacement",
            "energy": "rank_local_owned_cell_contribution",
            "residual": "exact_displacement_space_pullback",
            "tangent": "exact_matrix_free_consistent_action",
            "scope": "one_fiber_bending_contribution_not_complete_shell",
            "kinematics": self.kinematics.as_dict(),
            "gradient": self.gradient.as_dict(),
        }


def displacement_fiber_bending(
    kinematics: ConvectedCellFiberOperator,
    gradient: CellGradientOperator,
    *,
    cell_weights,
    in_plane_stiffness=0.0,
    normal_stiffness=0.0,
    reference_in_plane_curvature=0.0,
    reference_normal_curvature=0.0,
) -> DisplacementFiberBendingOperator:
    """Compose one displacement-derived neighbour-bending contribution."""

    return DisplacementFiberBendingOperator(
        kinematics=kinematics,
        gradient=gradient,
        cell_weights=cell_weights,
        in_plane_stiffness=in_plane_stiffness,
        normal_stiffness=normal_stiffness,
        reference_in_plane_curvature=reference_in_plane_curvature,
        reference_normal_curvature=reference_normal_curvature,
    )


__all__ = ["DisplacementFiberBendingOperator", "displacement_fiber_bending"]
