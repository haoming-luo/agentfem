"""Reusable operators for fourth-order scalar problems."""

from __future__ import annotations

import ufl

from .core import OperatorForm


def split_laplacian_operator(
    trial,
    test,
    *,
    measure=ufl.dx,
    name: str = "K_split_laplacian",
) -> OperatorForm:
    """Return one second-order block of a mixed biharmonic split."""

    return OperatorForm(
        name=name,
        kind="split_laplacian_operator",
        role="matrix",
        family="fourth_order_split",
        expression=ufl.inner(ufl.grad(trial), ufl.grad(test)) * measure,
        metadata={"equation": "-laplacian(field) = source"},
    )


def auxiliary_laplacian_boundary(boundary_expression):
    """Return ``-Delta(g)`` for the auxiliary field ``w=-Delta(u)``."""

    return -ufl.div(ufl.grad(boundary_expression))


__all__ = ["auxiliary_laplacian_boundary", "split_laplacian_operator"]
