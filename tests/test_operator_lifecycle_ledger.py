# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from agentfem._operator_lifecycle import OperatorLifecycleLedger, boundary_dof_identity


def _snapshot(rhs, *, matrices=1):
    return {
        "matrix_assembly_count": matrices,
        "rhs_assembly_count": rhs,
        "solve_count": rhs,
    }


def test_prepared_generations_preserve_totals_and_iterations():
    ledger = OperatorLifecycleLedger()
    ledger.begin_prepared()
    ledger.record(_snapshot(1), SimpleNamespace(iterations=2), accumulate=False)
    ledger.record(_snapshot(2), SimpleNamespace(iterations=3), accumulate=False)
    ledger.begin_prepared()
    ledger.record(_snapshot(1), SimpleNamespace(iterations=1), accumulate=False)
    assert ledger.summary() == {
        "matrix_assembly_count": 2,
        "rhs_assembly_count": 3,
        "solve_count": 3,
        "matrix_refresh_count": 1,
        "ksp_iterations_total": 6,
        "ksp_iterations_maximum": 3,
    }


def test_refresh_and_empty_records():
    ledger = OperatorLifecycleLedger()
    ledger.record(None, None, accumulate=True)
    for _ in range(3):
        ledger.record(_snapshot(1), None, accumulate=True)
    assert ledger.summary()["matrix_assembly_count"] == 3
    assert ledger.summary()["solve_count"] == 3
    assert ledger.summary()["ksp_iterations_total"] == 0
    assert OperatorLifecycleLedger().summary()["solve_count"] == 0


def test_invalid_backend_counters_do_not_partially_change_evidence():
    ledger = OperatorLifecycleLedger()
    ledger.record(_snapshot(2), None, accumulate=False)
    previous = ledger.summary()
    with pytest.raises(ValueError, match="regressed"):
        ledger.record(_snapshot(1, matrices=2), None, accumulate=False)
    assert ledger.summary() == previous
    with pytest.raises(ValueError, match="nonnegative"):
        ledger.record(_snapshot(3), SimpleNamespace(iterations=-1), accumulate=False)
    assert ledger.summary() == previous
    ledger.record(_snapshot(3), None, accumulate=False)
    assert ledger.summary()["solve_count"] == 3


@pytest.mark.parametrize("size", [0, 1, 10000])
def test_boundary_identity_is_an_exact_immutable_live_snapshot(size):
    dofs = np.arange(size, dtype=np.int32)
    bc = SimpleNamespace(dof_indices=lambda: (dofs, size))
    original = boundary_dof_identity([bc])
    assert original == boundary_dof_identity([bc])
    if size:
        dofs[-1] += 1
        assert original != boundary_dof_identity([bc])
    assert original != boundary_dof_identity([bc, bc])


def test_boundary_identity_binds_ownership_and_order_not_values():
    dofs = np.array([1, 5, 8], dtype=np.int32)
    a = SimpleNamespace(dof_indices=lambda: (dofs, 2), value=0.0)
    b = SimpleNamespace(dof_indices=lambda: (dofs, 3))
    original = boundary_dof_identity([a])
    a.value = 9.0
    assert boundary_dof_identity([a]) == original
    assert boundary_dof_identity([b]) != original
    assert boundary_dof_identity([a, b]) != boundary_dof_identity([b, a])


@pytest.mark.parametrize("kind", ["heat", "dynamics"])
def test_release_and_continue_preserves_evidence_and_solution(kind):
    from test_first_order_operator_lifecycle import _heat_step
    from test_implicit_dynamics_lifecycle import _implicit_cantilever

    factory = _heat_step if kind == "heat" else _implicit_cantilever
    step = factory("reuse", steps=4)
    reference = factory("reuse", steps=4)
    try:
        step.run(until_step=2)
        step.close()
        step.close()  # Resource release is idempotent, not an evidence reset.
        step.run()
        reference.run()
        report = step.operator_lifecycle_summary()
        assert report["matrix_assembly_count"] == 2
        assert report["rhs_assembly_count"] == report["solve_count"] == 4
        actual = step.current if kind == "heat" else step.state.u.value
        expected = reference.current if kind == "heat" else reference.state.u.value
        np.testing.assert_allclose(
            actual.x.array, expected.x.array, rtol=1e-12, atol=1e-13
        )
    finally:
        step.close()
        reference.close()


@pytest.mark.parametrize("effect", ["rhs", "operator"])
def test_live_transferred_input_refreshes_only_when_required(effect):
    """A live field input must change the computed response, not just metadata."""

    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI
    from petsc4py import PETSc

    from agentfem import operators, problems, time

    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    space = fem.functionspace(domain, ("Lagrange", 1))
    current, previous = fem.Function(space), fem.Function(space)
    previous.x.array[:] = 1.0
    current.x.array[:] = 1.0
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    coefficient = fem.Constant(domain, PETSc.ScalarType(1.0))
    source = fem.Constant(domain, PETSc.ScalarType(2.0))

    def update(t):
        if effect == "operator":
            coefficient.value = 1.0 + t
        else:
            source.value = 2.0 + t

    dt, count = 0.1, 4
    step = problems.first_order_transient_run(
        capacity=operators.from_ufl(trial * test * ufl.dx, name="C", role="matrix"),
        stiffness=operators.from_ufl(
            coefficient * trial * test * ufl.dx, name="K", role="matrix"
        ),
        history=operators.from_ufl(
            previous * test * ufl.dx, name="history", role="vector"
        ),
        source=operators.from_ufl(source * test * ufl.dx, name="source", role="vector"),
        current=current,
        previous=previous,
        dt=dt,
        steps=count,
        update_load=time.input_update(update, effects=effect),
        progress=False,
    )
    try:
        step.run()
        expected = 1.0
        for n in range(1, count + 1):
            t = n * dt
            stiffness = 1.0 + t if effect == "operator" else 1.0
            force = 2.0 + t if effect == "rhs" else 2.0
            expected = (expected + dt * force) / (1.0 + dt * stiffness)
        np.testing.assert_allclose(current.x.array, expected, rtol=1e-12, atol=1e-13)
        report = step.operator_lifecycle_summary()
        assert report["matrix_assembly_count"] == (count if effect == "operator" else 1)
        assert report["rhs_assembly_count"] == report["solve_count"] == count
    finally:
        step.close()
