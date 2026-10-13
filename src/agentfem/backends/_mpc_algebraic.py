# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded distributed exact-MPC projection of fully assembled operators."""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc


class PreparedAlgebraicMPCLinearProblem:
    """Form T.T A T explicitly for homogeneous one-master periodic constraints.

    Global constraint/root numbering is replicated, numerical matrices remain
    distributed. This reference route prioritizes correctness over scalable
    graph setup. Nonzero essential data, chained or weighted MPCs are rejected.
    """

    def __init__(
        self,
        bilinear_form,
        linear_form,
        solution,
        constraint,
        *,
        bcs=None,
        options=None,
        petsc_options_prefix="agentfem_algebraic_mpc_",
    ):
        from ..solvers import LinearSolverOptions

        self.solution, self.constraint = solution, constraint
        self.options = options or LinearSolverOptions()
        self.bcs = [getattr(b, "bc", b) for b in (bcs or ())]
        probe = fem.Function(solution.function_space)
        fem_petsc.set_bc(probe.x.petsc_vec, self.bcs)
        maximum = solution.function_space.mesh.comm.allreduce(
            float(np.max(np.abs(probe.x.array), initial=0.0)), op=MPI.MAX
        )
        if maximum > 64.0 * np.finfo(float).eps:
            raise ValueError(
                "Algebraic periodic projection supports homogeneous strong constraints only."
            )
        self.a, self.L = fem.form(bilinear_form), fem.form(linear_form)
        self.prefix = petsc_options_prefix
        self.closed = False
        self.last_solve_info = None
        self.solve_count = 0
        self.relative_residual = None
        self.periodic_error = None

    def solve(self):
        from ..solvers import LinearSolveInfo

        if self.closed:
            raise RuntimeError("Algebraic MPC problem is closed.")
        u, backend = self.solution, self.constraint.backend
        comm = u.function_space.mesh.comm
        dm = u.function_space.dofmap
        bs = dm.index_map_bs
        owned = dm.index_map.size_local * bs

        def global_dof(local):
            return (
                int(
                    dm.index_map.local_to_global(
                        np.array([int(local) // bs], dtype=np.int32)
                    )[0]
                )
                * bs
                + int(local) % bs
            )

        bdm = backend.function_space.dofmap

        def master_global(local):
            return (
                int(
                    bdm.index_map.local_to_global(
                        np.array([int(local) // bs], dtype=np.int32)
                    )[0]
                )
                * bs
                + int(local) % bs
            )

        co, offsets = backend.coefficients()
        local_pairs = []
        error = None
        for slave in backend.slaves[: backend.num_local_slaves]:
            masters = backend.masters.links(int(slave))
            coeff = co[offsets[slave] : offsets[slave + 1]]
            if len(masters) != 1 or not np.allclose(coeff, 1.0, atol=1.0e-8, rtol=0):
                error = "Algebraic periodic projection requires exactly one unit master per slave."
            else:
                local_pairs.append((global_dof(slave), master_global(masters[0])))
        errors = comm.allgather(error)
        if any(errors):
            raise ValueError(str(errors))
        pairs = dict(pair for group in comm.allgather(local_pairs) for pair in group)
        if any(master in pairs for master in pairs.values()):
            raise ValueError("Chained periodic relations are not supported.")
        local_fixed = []
        for bc in self.bcs:
            dofs, stop = bc.dof_indices()
            local_fixed.extend(global_dof(i) for i in dofs[:stop])
        fixed = set(i for group in comm.allgather(local_fixed) for i in group)
        if fixed.intersection(pairs):
            raise ValueError("Strong constraints overlap periodic slaves.")
        local_global = [global_dof(i) for i in range(owned)]
        roots = sorted(
            i
            for group in comm.allgather(
                [g for g in local_global if g not in pairs and g not in fixed]
            )
            for i in group
        )
        if not roots:
            raise ValueError("Periodic projection has no free unknowns.")
        root_index = {g: i for i, g in enumerate(roots)}
        resources = []
        try:
            T = PETSc.Mat().createAIJ(
                size=(
                    (owned, dm.index_map.size_global * bs),
                    (PETSc.DECIDE, len(roots)),
                ),
                nnz=1,
                comm=comm,
            )
            resources.append(T)
            for g in local_global:
                master = pairs.get(g, g)
                if master not in fixed:
                    T.setValue(g, root_index[master], 1.0)
            T.assemble()
            A = fem_petsc.assemble_matrix(self.a)
            resources.append(A)
            A.assemble()
            b = fem_petsc.assemble_vector(self.L)
            resources.append(b)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            Ar = A.PtAP(T)
            resources.append(Ar)
            br = T.createVecRight()
            resources.append(br)
            T.multTranspose(b, br)
            x = Ar.createVecRight()
            resources.append(x)
            ksp = PETSc.KSP().create(comm)
            resources.append(ksp)
            ksp.setOptionsPrefix(self.prefix)
            ksp.setOperators(Ar)
            ksp.setType(self.options.ksp_type)
            ksp.getPC().setType(self.options.pc_type)
            if self.options.factor_solver_type:
                ksp.getPC().setFactorSolverType(self.options.factor_solver_type)
            ksp.setTolerances(
                rtol=self.options.rtol or 1.0e-10,
                atol=self.options.atol or 1.0e-14,
                max_it=self.options.max_it or 1000,
            )
            ksp.solve(br, x)
            residual = br.duplicate()
            resources.append(residual)
            Ar.mult(x, residual)
            residual.axpy(-1.0, br)
            self.relative_residual = residual.norm() / max(br.norm(), 1.0e-30)
            self.last_solve_info = LinearSolveInfo(
                converged_reason=int(ksp.getConvergedReason()),
                iterations=int(ksp.getIterationNumber()),
                residual_norm=residual.norm(),
            )
            if not self.last_solve_info.converged:
                raise RuntimeError(
                    "AFM-RVE-009: reduced periodic system failed to converge."
                )
            full = T.createVecLeft()
            resources.append(full)
            T.mult(x, full)
            u.x.array[:owned] = full.array
            u.x.scatter_forward()
            values = dict(
                pair
                for group in comm.allgather(
                    list(zip(local_global, full.array.tolist()))
                )
                for pair in group
            )
            norm = max((abs(value) for value in values.values()), default=0.0)
            self.periodic_error = max(
                (
                    abs(values[slave] - values[master])
                    for slave, master in pairs.items()
                ),
                default=0.0,
            ) / max(norm, 1.0e-30)
            self.solve_count += 1
            return u
        finally:
            for item in reversed(resources):
                item.destroy()

    def summary(self):
        return {
            "kind": "algebraic_periodic_projection",
            "assembly": "T.T A T",
            "matrix_values_reassembled": True,
            "graph_numbering": "replicated",
            "solve_count": self.solve_count,
            "relative_residual": self.relative_residual,
            "periodic_relative_error": self.periodic_error,
        }

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False
