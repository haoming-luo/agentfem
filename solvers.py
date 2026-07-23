"""Linear algebra solver helpers for finite-element problems."""

from __future__ import annotations

from dataclasses import dataclass

from petsc4py import PETSc

import dolfinx.fem.petsc as fem_petsc


@dataclass(frozen=True)
class LinearSolverOptions:
    """PETSc KSP options for a linear solve."""

    ksp_type: str = "preonly"
    pc_type: str = "lu"
    rtol: float | None = None
    atol: float | None = None
    max_it: int | None = None


def create_ksp(comm, options: LinearSolverOptions | None = None):
    """Create and configure a PETSc KSP object."""

    options = options or LinearSolverOptions()
    ksp = PETSc.KSP().create(comm)
    ksp.setType(options.ksp_type)
    pc = ksp.getPC()
    pc.setType(options.pc_type)
    if options.rtol is not None or options.atol is not None or options.max_it is not None:
        ksp.setTolerances(
            rtol=options.rtol,
            atol=options.atol,
            max_it=options.max_it,
        )
    return ksp


def solve_matrix_system(A, b, x, options: LinearSolverOptions | None = None) -> None:
    """Solve ``A x = b`` into a PETSc vector."""

    ksp = create_ksp(A.comm, options)
    ksp.setOperators(A)
    ksp.solve(b, x)
    ksp.destroy()


def solve_linear_problem(
    bilinear_form,
    linear_form,
    solution,
    *,
    bcs=None,
    options: LinearSolverOptions | None = None,
):
    """Assemble and solve a standard linear variational problem.

    Parameters
    ----------
    bilinear_form:
        Compiled form for the left-hand side.
    linear_form:
        Compiled form for the right-hand side.
    solution:
        DOLFINx function storing the solution.
    bcs:
        Optional list of strong Dirichlet boundary conditions.
    options:
        PETSc KSP configuration.
    """

    bcs = [] if bcs is None else list(bcs)
    A = fem_petsc.assemble_matrix(bilinear_form, bcs=bcs)
    A.assemble()
    b = fem_petsc.assemble_vector(linear_form)
    fem_petsc.apply_lifting(b, [bilinear_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)

    solve_matrix_system(A, b, solution.x.petsc_vec, options)
    solution.x.scatter_forward()

    b.destroy()
    A.destroy()
    return solution
