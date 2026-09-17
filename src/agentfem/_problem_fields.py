# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared diagnostic fields recovered from discrete problems."""

from __future__ import annotations

from dolfinx import fem


def reaction_field(residual_form, solution, *, name: str):
    """Assemble an unconstrained residual into a field-shaped diagnostic."""

    import dolfinx.fem.petsc as fem_petsc
    from petsc4py import PETSc

    residual = fem_petsc.assemble_vector(fem.form(residual_form))
    residual.ghostUpdate(
        addv=PETSc.InsertMode.ADD,
        mode=PETSc.ScatterMode.REVERSE,
    )
    reaction = fem.Function(solution.function_space, name=name)
    values = residual.array_r
    reaction.x.array[: len(values)] = values
    reaction.x.scatter_forward()
    residual.destroy()
    return reaction


__all__ = ()
