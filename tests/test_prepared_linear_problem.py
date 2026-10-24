from __future__ import annotations

import numpy as np
from dolfinx import fem
from mpi4py import MPI
import pytest

from agentfem import constraints, fields, mesh, operators, results, solvers
from agentfem._solver_lifecycle import PreparedSolve


@pytest.mark.parametrize("ksp,pc", [("cg", "jacobi"), ("preonly", "lu")])
def test_prepared_zero_rhs_ignores_stale_norm_after_numpy_write(ksp, pc):
    import ufl
    domain = mesh.rectangle((0., 0.), (1., 1.), (2, 2), comm=MPI.COMM_WORLD)
    unknown = fields.scalar_unknown(domain, degree=1, name="zero")
    source = fem.Constant(domain, 0.)
    with solvers.prepare_linear_problem(
        unknown.trial*unknown.test*ufl.dx,
        source*unknown.test*ufl.dx,
        unknown.value,
        options=solvers.LinearSolverOptions(ksp_type=ksp, pc_type=pc),
    ) as prepared:
        prepared.solve()
        assert unknown.value.x.petsc_vec.norm() == 0.
        unknown.value.x.array[:] = 3.
        prepared.solve()
        np.testing.assert_array_equal(unknown.value.x.array, 0.)
        assert prepared.matrix_assembly_count == 1
        assert prepared.solve_count == 2


def test_prepared_linear_problem_reuses_matrix_for_updated_rhs():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (6, 6),
        comm=MPI.COMM_SELF,
        cell_type="triangle",
    )
    unknown = fields.scalar_unknown(domain, degree=1, name="u")
    boundary = mesh.boundary(
        domain,
        lambda x: (
            np.isclose(x[0], 0.0)
            | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0)
            | np.isclose(x[1], 1.0)
        ),
        name="boundary",
        tag=1,
    )
    fixed = constraints.fixed(unknown, on=boundary, value=0.0)
    source = fem.Constant(domain, 1.0)
    stiffness = operators.diffusion_operator(unknown, 1.0)
    force = operators.source_vector(source, unknown)

    with solvers.prepare_linear_problem(
        stiffness.expression,
        force.expression,
        unknown.value,
        bcs=fixed.bcs,
        options=solvers.direct_solver(),
    ) as prepared:
        assert isinstance(prepared, PreparedSolve)
        assert not prepared.closed
        first = prepared.solve()
        first_center = results.probe(first, at=(0.5, 0.5))
        source.value = 2.0
        second = prepared.solve()
        second_center = results.probe(second, at=(0.5, 0.5))
        summary = prepared.summary()

    assert second_center > first_center
    np.testing.assert_allclose(second_center, 2.0 * first_center, rtol=1.0e-12)
    assert summary["matrix_reused"] is True
    assert summary["solve_count"] == 2
    assert summary["last_solve"]["converged"] is True
    assert prepared.closed
    assert prepared.summary() == summary
    with pytest.raises(RuntimeError, match="PreparedLinearProblem is closed"):
        prepared.solve()


@pytest.mark.parametrize("bc_assembly", ["lifting", "matrix_elimination"])
def test_failed_preparation_releases_already_owned_resources(monkeypatch, bc_assembly):
    import ufl
    domain = mesh.rectangle((0., 0.), (1., 1.), (1, 1), comm=MPI.COMM_SELF)
    unknown = fields.scalar_unknown(domain, degree=1, name="cleanup")
    source = fem.Constant(domain, 1.)
    captured = []
    close = solvers.PreparedLinearProblem.close

    def tracked_close(instance):
        captured.extend(resource for resource in (
            instance.matrix, instance._lifting_matrix,
            instance._prescribed, instance._boundary_action,
        ) if resource is not None)
        close(instance)

    def failed_ksp(*args, **kwargs):
        raise RuntimeError("injected KSP preparation failure")

    monkeypatch.setattr(solvers.PreparedLinearProblem, "close", tracked_close)
    monkeypatch.setattr(solvers, "create_ksp", failed_ksp)
    with pytest.raises(RuntimeError, match="injected KSP preparation failure"):
        solvers.prepare_linear_problem(
            unknown.trial * unknown.test * ufl.dx,
            source * unknown.test * ufl.dx, unknown.value,
            bc_assembly=bc_assembly,
        )
    assert len(captured) == (4 if bc_assembly == "matrix_elimination" else 1)
    assert all(not resource for resource in captured)


@pytest.mark.parametrize("bc_assembly", ["lifting", "matrix_elimination"])
def test_failed_prepared_solve_frees_rhs_and_does_not_reuse_success_evidence(monkeypatch, bc_assembly):
    import ufl
    domain = mesh.rectangle((0., 0.), (1., 1.), (1, 1), comm=MPI.COMM_SELF)
    unknown = fields.scalar_unknown(domain, degree=1, name="cleanup")
    source = fem.Constant(domain, 1.)
    with solvers.prepare_linear_problem(
        unknown.trial * unknown.test * ufl.dx,
        source * unknown.test * ufl.dx, unknown.value, bc_assembly=bc_assembly,
    ) as prepared:
        prepared.solve()
        assert prepared.last_solve_info.converged
        original = solvers.fem_petsc.assemble_vector
        vectors = []

        def tracked_assembly(*args, **kwargs):
            vector = original(*args, **kwargs)
            vectors.append(vector)
            return vector

        class FailingKSP:
            def solve(self, *args):
                raise RuntimeError("injected KSP solve failure")

        actual = prepared.ksp
        prepared.ksp = FailingKSP()
        try:
            with monkeypatch.context() as patch:
                patch.setattr(solvers.fem_petsc, "assemble_vector", tracked_assembly)
                with pytest.raises(RuntimeError, match="injected KSP solve failure"):
                    prepared.solve()
            # DOLFINx's singledispatch allocator calls its vector-filling
            # overload internally; both observations can be the same handle.
            unique_vectors = {id(vector): vector for vector in vectors}
            assert len(unique_vectors) == 1 and all(not vector for vector in unique_vectors.values())
            assert prepared.last_solve_info is None
            assert prepared.solve_count == 1 and prepared.rhs_assembly_count == 2
            assert not prepared.closed
        finally:
            prepared.ksp = actual
        prepared.solve()
        assert prepared.last_solve_info.converged
        assert prepared.solve_count == 2


@pytest.mark.parametrize("bc_assembly", ["lifting", "matrix_elimination"])
def test_failed_refresh_cannot_leave_a_partially_modified_operator(monkeypatch, bc_assembly):
    import ufl
    domain = mesh.rectangle((0., 0.), (1., 1.), (1, 1), comm=MPI.COMM_SELF)
    unknown = fields.scalar_unknown(domain, degree=1, name="cleanup")
    source = fem.Constant(domain, 1.)
    with solvers.prepare_linear_problem(
        unknown.trial * unknown.test * ufl.dx,
        source * unknown.test * ufl.dx, unknown.value, bc_assembly=bc_assembly,
    ) as prepared:
        prepared.solve()
        matrix = prepared.matrix
        def fail(*args, **kwargs):
            raise RuntimeError("injected reassembly failure")
        monkeypatch.setattr(solvers.fem_petsc, "assemble_matrix", fail)
        with pytest.raises(RuntimeError, match="injected reassembly failure"):
            prepared.refresh_matrix()
        assert prepared.closed and not matrix
        assert prepared.last_solve_info is None
        assert prepared.matrix_assembly_count == 1
        with pytest.raises(RuntimeError, match="closed"):
            prepared.solve()
