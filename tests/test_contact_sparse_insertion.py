# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Batch insertion keeps serial contact row identity and additive semantics."""
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

from agentfem.boundary_models._petsc_contact_rows import add_contact_coo_tangent
from agentfem.boundary_models.implicit_contact import _coalesce_trace_tangent


@pytest.mark.parametrize("values", [
    np.array([1. + 2.j, 3.]),
    np.array(["1", "3"]),
    np.array([True, False]),
    np.array([1.e308, 1.e308]),
])
def test_invalid_or_overflowed_contact_tangent_is_rejected_before_matrix_mutation(values):
    class UntouchedMatrix:
        def getSize(self):
            return 2, 2

        getLocalSize = getSize

        def assemble(self, *args):
            pytest.fail("Invalid contact entries must not flush the bulk matrix")

        def setValuesIJV(self, *args, **kwargs):
            pytest.fail("Invalid contact entries must not reach PETSc")

    with pytest.raises(ValueError, match="Contact tangent"):
        add_contact_coo_tangent(UntouchedMatrix(), [0, 0], [1, 1], values)


@pytest.mark.parametrize("values", [np.array([1. + 2.j, 3.]), [1.e308, 1.e308]])
def test_rejected_tangent_preserves_real_petsc_matrix_and_allows_valid_insertion(values):
    matrix = PETSc.Mat().createAIJ((2, 2), nnz=2, comm=MPI.COMM_SELF)
    dofs = np.arange(2, dtype=PETSc.IntType)
    initial = np.array([[3., 1.], [1., 2.]])
    try:
        matrix.setValues(dofs, dofs, initial)
        matrix.assemble()
        state = matrix.stateGet()
        with pytest.raises(ValueError, match="Contact tangent"):
            add_contact_coo_tangent(matrix, [0, 0], [1, 1], values)
        assert matrix.stateGet() == state
        np.testing.assert_array_equal(matrix.getValues(dofs, dofs), initial)
        add_contact_coo_tangent(matrix, [0, 0], [1, 1], [2., -1.])
        matrix.assemble()
        np.testing.assert_array_equal(matrix.getValues(dofs, dofs), [[3., 2.], [1., 2.]])
    finally:
        matrix.destroy()


def test_contact_assembly_rejection_rolls_back_joint_state_and_can_resume(monkeypatch):
    from test_retained_material_trial import make_step

    step = make_step()
    step.solve(until=.25)
    before = step.solution.x.array.copy()
    material = step.response.state.committed_state_vectors().copy()
    multiplier = step.contact_provider.normal_multiplier_state.accepted_multiplier.copy()
    work = step.accepted_history_recorders["normal_contact_work:interior_plane"]
    accepted_work = work.checkpoint_state()

    def invalid_tangent(matrix, contribution, **kwargs):
        add_contact_coo_tangent(matrix, [0, 0], [0, 0], [1.e308, 1.e308])

    with monkeypatch.context() as patch:
        patch.setattr(step.contact_provider, "add_tangent", invalid_tangent)
        with pytest.raises((ValueError, RuntimeError), match="Contact tangent"):
            step.solve(until=.5)
    assert step.accepted_load_factor == .25
    np.testing.assert_array_equal(step.solution.x.array, before)
    np.testing.assert_array_equal(step.response.state.committed_state_vectors(), material)
    np.testing.assert_array_equal(
        step.contact_provider.normal_multiplier_state.accepted_multiplier, multiplier)
    assert work.checkpoint_state() == accepted_work
    assert step.state_transaction._retained_trial_digest is None

    step.solve(until=.5)
    reference = make_step()
    reference.solve(until=.5)
    np.testing.assert_array_equal(step.solution.x.array, reference.solution.x.array)
    np.testing.assert_array_equal(step.response.state.committed_state_vectors(),
                                  reference.response.state.committed_state_vectors())
    assert work.checkpoint_state() == reference.accepted_history_recorders[
        "normal_contact_work:interior_plane"].checkpoint_state()


@pytest.mark.parametrize("count", [0, 1, 31, 8192])
@pytest.mark.parametrize("fixed", [(), (0, 8, 12), tuple(range(17))])
def test_shuffled_duplicate_contact_entries_match_scalar_reference(count, fixed):
    rng = np.random.default_rng(5925)
    rows, columns = rng.integers(0, 17, size=(2, count))
    values = rng.normal(size=count)
    expected = 3. * np.eye(17)
    free = ~np.isin(rows, fixed) & ~np.isin(columns, fixed)
    # Deliberately independent scalar accumulation, including duplicate rows,
    # repeated columns, negative entries and absent matrix rows.
    np.add.at(expected, (rows[free], columns[free]), values[free])
    matrix = PETSc.Mat().createAIJ((17, 17), nnz=17, comm=MPI.COMM_SELF)
    # FINAL assembly discards unused preallocation; allow the deliberately
    # independent contact graph to introduce entries after the bulk diagonal.
    matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    dofs = np.arange(17, dtype=PETSc.IntType)
    try:
        for i in dofs:
            matrix.setValue(i, i, 3.)
        matrix.assemble()
        add_contact_coo_tangent(matrix, rows, columns, values, constrained_dofs=fixed)
        matrix.assemble()
        np.testing.assert_allclose(matrix.getValues(dofs, dofs), expected, rtol=2e-14, atol=2e-14)
    finally:
        matrix.destroy()


def test_one_native_insertion_retains_legacy_duplicate_summation_order():
    class Recorder:
        calls = 0
        def getSize(self):
            return 31, 31
        getLocalSize = getSize
        def assemble(self, *args):
            pass
        def setValuesIJV(self, pointers, columns, values, *, rowmap, addv):
            self.calls += 1
            self.received = pointers, columns, values, rowmap
    rng = np.random.default_rng(422)
    rows = rng.choice([2, 8, 29], 3000)
    columns = rng.integers(0, 31, 3000)
    values = rng.normal(size=3000) * rng.choice([1e-9, 1., 1e9], 3000)
    order = np.lexsort((columns, rows))
    rr, cc, vv = rows[order], columns[order], values[order]
    expected = []
    for row in np.unique(rr):
        selected = rr == row
        unique, inverse = np.unique(cc[selected], return_inverse=True)
        expected.append((row, unique, np.bincount(inverse, weights=vv[selected])))
    matrix = Recorder()
    add_contact_coo_tangent(matrix, rows, columns, values)
    assert matrix.calls == 1
    pointers, actual_columns, actual_values, rowmap = matrix.received
    for i, (row, cols, vals) in enumerate(expected):
        segment = slice(pointers[i], pointers[i + 1])
        assert rowmap[i] == row
        np.testing.assert_array_equal(actual_columns[segment], cols)
        np.testing.assert_array_equal(actual_values[segment], vals)


@pytest.mark.parametrize("width", [3, 4, 6, 9])
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("dimension", [2, 3])
def test_trace_stencil_integration_retains_every_current_point(width, interleave, dimension):
    rng = np.random.default_rng(512)
    # Shared vertices, distinct stencils, noncontiguous repetitions and a
    # reversed local ordering must not be mistaken for the same stencil.
    stencils = np.stack((np.arange(width), np.arange(1, width + 1), np.arange(width)[::-1]))
    nodes = np.repeat(stencils, [13, 7, 5], axis=0)
    matrices = rng.normal(size=(len(nodes), width, dimension, width, dimension))
    if interleave:
        order = rng.permutation(len(nodes))
        nodes, matrices = nodes[order], matrices[order]
    reduced_nodes, reduced = _coalesce_trace_tangent(nodes, matrices)
    size = dimension * (width + 1)
    def dense(nn, mm):
        result = np.zeros((size, size))
        for ids, matrix in zip(nn, mm, strict=True):
            dofs = (dimension * ids[:, None] + np.arange(dimension)).ravel()
            result[np.ix_(dofs, dofs)] += matrix.reshape((width * dimension, width * dimension))
        return result
    np.testing.assert_allclose(dense(reduced_nodes, reduced), dense(nodes, matrices), rtol=2e-14, atol=2e-14)
    if not interleave:
        assert len(reduced_nodes) == 3
