# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Composition of assembled dual vectors with variational vectors."""

from dataclasses import dataclass
import numpy as np
from petsc4py import PETSc


@dataclass(frozen=True)
class DiscreteVectorSum:
    terms: tuple
    name: str = "F"
    role: str = "vector"

    def __post_init__(self):
        if not self.terms or any(not np.isfinite(factor) for factor, _ in self.terms):
            raise ValueError(
                "Discrete vector composition requires finite, nonempty terms."
            )

    def assemble_vector(self):
        from .core import assemble_vector

        result = None
        try:
            for factor, term in self.terms:
                vector = assemble_vector(term)
                try:
                    if result is None:
                        result = vector.copy()
                        result.scale(factor)
                    else:
                        result.axpy(factor, vector)
                finally:
                    vector.destroy()
            result.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )
            return result
        except BaseException:
            if result is not None:
                result.destroy()
            raise

    def summary(self):
        return {
            "kind": "assembled_dual_vector_sum",
            "name": self.name,
            "terms": [
                {"factor": float(a), "operator": b.summary()} for a, b in self.terms
            ],
        }
