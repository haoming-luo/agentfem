# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Private same-mesh fixed-point iteration used by coupled Procedures.

Owns iteration scratch and compiled residual norms, not accepted fields, time,
physics, checkpointing or a reference solution. The caller must enclose run()
in its State transaction and commit only after its acceptance checks pass.
"""

from math import isfinite

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI


class StaggeredFieldIteration:
    def __init__(self, fields):
        self.fields = dict(fields)
        if not self.fields:
            raise ValueError("At least one coupled field is required.")
        domain = next(iter(self.fields.values())).function_space.mesh
        if any(f.function_space.mesh is not domain for f in self.fields.values()):
            raise ValueError("Coupled iteration requires one shared mesh.")
        self.comm = domain.comm
        declaration = tuple(self.fields)
        if any(item != declaration for item in self.comm.allgather(declaration)):
            raise ValueError("AFM-COUPLING-003: field order differs across ranks.")
        dx = ufl.Measure("dx", domain=domain)
        self.volume = self._sum(fem.form(fem.Constant(domain, 1.0) * dx))
        if self.volume <= 0:
            raise ValueError("Coupled iteration requires positive domain volume.")
        self.previous = {
            name: fem.Function(f.function_space) for name, f in self.fields.items()
        }
        # Compile once. The coefficients change, not their forms or measures.
        self.norm_forms = {}
        for name, field in self.fields.items():
            delta = field - self.previous[name]
            self.norm_forms[name] = (
                fem.form(ufl.inner(delta, delta) * dx),
                fem.form(ufl.inner(field, field) * dx),
            )

    def _sum(self, form):
        value = float(self.comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM).real)
        if not isfinite(value):
            raise RuntimeError("AFM-COUPLING-002: non-finite coupled field norm.")
        return value

    def run(
        self,
        solves,
        *,
        absolute_tolerances,
        rtol,
        relaxation,
        max_iterations,
        decoupled=False,
    ):
        """Return unrelaxed residual history; never advance accepted time."""
        solves = tuple(solves)
        tolerances = dict(absolute_tolerances)
        declaration = (
            tuple(tolerances.items()),
            rtol,
            relaxation,
            max_iterations,
            bool(decoupled),
            len(solves),
        )
        if any(item != declaration for item in self.comm.allgather(declaration)):
            raise ValueError("AFM-COUPLING-003: iteration policy differs across ranks.")
        if (
            set(tolerances) != set(self.fields)
            or not solves
            or type(max_iterations) is not int
            or max_iterations < 1
            or not all(isfinite(v) and v > 0 for v in (*tolerances.values(), rtol))
            or not isfinite(relaxation)
            or not 0 < relaxation <= 1
        ):
            raise ValueError(
                "Finite positive field tolerances, integer iteration limit and 0 < relaxation <= 1 required."
            )
        residuals = []
        for iteration in range(1, max_iterations + 1):
            for name, field in self.fields.items():
                self.previous[name].x.array[:] = field.x.array
                self.previous[name].x.scatter_forward()
            for solve in solves:
                solve()
            record = {"iteration": iteration}
            converged = True
            for name, (difference, magnitude) in self.norm_forms.items():
                absolute = float(np.sqrt(max(0.0, self._sum(difference) / self.volume)))
                scale = float(np.sqrt(max(0.0, self._sum(magnitude) / self.volume)))
                record[f"{name}_absolute"] = absolute
                record[f"{name}_relative"] = absolute / scale if scale else None
                converged = converged and absolute <= tolerances[name] + rtol * scale
            residuals.append(record)
            if decoupled or converged:
                return residuals
            if iteration == max_iterations:
                raise RuntimeError(
                    "AFM-COUPLING-001: outer iteration exhausted; window rejected."
                )
            for name, field in self.fields.items():
                previous = self.previous[name].x.array
                field.x.array[:] = previous + relaxation * (field.x.array - previous)
                field.x.scatter_forward()
        raise AssertionError("Validated positive iteration limit must return or fail.")
