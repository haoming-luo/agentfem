# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Distributed buckling pencil; the geometric matrix need not be definite."""

import numpy as np
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc
from ..dependencies import require
from .._modal_fem import orient_mode_deterministically, require_symmetric_operator
from ._modal import _free_dof_layout, _run_rank_local_phase


def solve_buckling(
    stiffness, geometric, target, bcs, *, modes, tolerance, maximum_iterations
):
    """Solve A phi = lambda B phi with B=-KG using GNHEP, never GHEP.

    Common free-DOF elimination removes spurious constrained eigenvalues.
    Shift-invert about zero selects the nearest positive critical factors.
    """
    SLEPc = require("slepc4py.SLEPc", extra="modal", capability="linear buckling")
    u = getattr(target, "value", target)
    V, comm = u.function_space, u.function_space.mesh.comm
    resources = []
    try:
        A = stiffness.assemble_matrix(bcs=None)
        resources.append(A)
        B = geometric.assemble_matrix(bcs=None)
        resources.append(B)
        B.scale(-1.0)
        local, glob, _ = _run_rank_local_phase(
            comm, stage="buckling free dofs", operation=lambda: _free_dof_layout(V, bcs)
        )
        count = int(comm.allreduce(len(local), op=MPI.SUM))
        if count <= modes + 1:
            raise ValueError(
                "AFM-BUCKLING-003: too few unconstrained DOFs for requested modes."
            )
        indices = PETSc.IS().createGeneral(glob, comm=comm)
        resources.append(indices)
        Ar = A.createSubMatrix(indices, indices)
        resources.append(Ar)
        Br = B.createSubMatrix(indices, indices)
        resources.append(Br)
        for name, matrix in (("elastic/base", Ar), ("geometric", Br)):
            require_symmetric_operator(matrix, name=name, relative_tolerance=1.0e-10)
        eps = SLEPc.EPS().create(comm)
        resources.append(eps)
        eps.setOperators(Ar, Br)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setDimensions(min(count - 1, max(2 * modes + 8, 12)))
        eps.setTarget(0.0)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eps.getST().setType(SLEPc.ST.Type.SINVERT)
        ksp = eps.getST().getKSP()
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        eps.setTolerances(tol=tolerance, max_it=maximum_iterations)
        eps.solve()
        vr = Ar.createVecRight()
        resources.append(vr)
        vi = Ar.createVecRight()
        resources.append(vi)
        av = Ar.createVecLeft()
        resources.append(av)
        bv = Br.createVecLeft()
        resources.append(bv)
        candidates = []
        for i in range(eps.getConverged()):
            lam = eps.getEigenvalue(i)
            if (
                not np.isfinite(lam)
                or np.real(lam) <= 0
                or abs(np.imag(lam)) > tolerance * max(1.0, abs(lam))
            ):
                continue
            lam = float(np.real(lam))
            eps.getEigenvector(i, vr, vi)
            Ar.mult(vr, av)
            Br.mult(vr, bv)
            scale = av.norm() + abs(lam) * bv.norm()
            av.axpy(-lam, bv)
            residual = av.norm() / max(scale, np.finfo(float).tiny)
            if not np.isfinite(residual) or residual > max(1.0e-8, 10 * tolerance):
                continue
            shape = fem.Function(V, name=f"Buckling_mode_{i + 1}")
            shape.x.array[local] = vr.array
            shape.x.scatter_forward()
            orient_mode_deterministically(shape, local, glob)
            norm = comm.allreduce(
                float(
                    np.max(
                        np.abs(
                            shape.x.array[
                                : V.dofmap.index_map.size_local * V.dofmap.index_map_bs
                            ]
                        ),
                        initial=0.0,
                    )
                ),
                op=MPI.MAX,
            )
            shape.x.array[:] /= norm
            shape.x.scatter_forward()
            candidates.append((lam, shape, residual))
        candidates.sort(key=lambda item: item[0])
        if len(candidates) < modes:
            raise RuntimeError(
                f"AFM-BUCKLING-004: accepted {len(candidates)} positive real modes; requested {modes}. Check compression, supports and search size."
            )
        return candidates[:modes], {
            "problem_type": "GNHEP",
            "free_dofs": count,
            "converged_eigenpairs": eps.getConverged(),
            "reason": int(eps.getConvergedReason()),
            "normalization": "maximum_absolute_nodal_component_one",
            "search": "nearest_zero_positive_real",
        }
    finally:
        for resource in reversed(resources):
            resource.destroy()
