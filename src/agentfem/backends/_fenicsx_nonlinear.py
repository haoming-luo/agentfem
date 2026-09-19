# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""FEniCSx bindings for matrix-free nonlinear operator contributions."""

from __future__ import annotations

from dataclasses import dataclass, field

from dolfinx import fem
from petsc4py import PETSc


@dataclass
class FEniCSxTangentAction:
    """Bind a nonlinear contribution and state to a PETSc tangent callback."""

    contribution: object
    state: object
    increment: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.state, fem.Function):
            raise TypeError("state must be a dolfinx.fem.Function.")
        tangent_action = getattr(self.contribution, "tangent_action", None)
        if not callable(tangent_action):
            raise TypeError("contribution must provide tangent_action(state, increment).")
        self.increment = fem.Function(
            self.state.function_space,
            name=f"D_{self.state.name or 'STATE'}",
        )

    def __call__(self, source: PETSc.Vec, target: PETSc.Vec) -> None:
        """Scatter one PETSc increment and write its contribution action."""

        if source.getSizes() != self.increment.x.petsc_vec.getSizes():
            raise ValueError("source vector does not match the primary function space.")
        source.copy(self.increment.x.petsc_vec)
        self.increment.x.scatter_forward()
        result = self.contribution.tangent_action(self.state, self.increment)
        if not isinstance(result, PETSc.Vec):
            raise TypeError("contribution.tangent_action() must return a PETSc Vec.")
        try:
            if result.getSizes() != target.getSizes():
                raise ValueError(
                    "contribution tangent result does not match the target vector."
                )
            result.copy(target)
        finally:
            result.destroy()

    def summary(self) -> dict[str, object]:
        """Return the adapter identity without claiming solver ownership."""

        describe = getattr(self.contribution, "as_dict", None)
        return {
            "kind": "fenicsx_tangent_action",
            "state": self.state.name or "unnamed",
            "contribution": describe() if callable(describe) else None,
            "ghost_update": "scatter_forward_before_tangent_action",
        }


def fenicsx_tangent_action(contribution, state) -> FEniCSxTangentAction:
    """Create a reusable PETSc callback for one FEniCSx primary field."""

    return FEniCSxTangentAction(contribution=contribution, state=state)


__all__ = ["FEniCSxTangentAction", "fenicsx_tangent_action"]
