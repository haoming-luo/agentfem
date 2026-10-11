import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

from agentfem.constitutive.quadrature import (
    QuadratureField,
    QuadratureTransaction,
    _state_value_snapshot,
)


@pytest.mark.parametrize("permuted", [False, True])
def test_concrete_snapshot_is_independent_for_both_layouts(permuted):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1,
                                   cell_type=mesh.CellType.hexahedron)
    field = QuadratureField.create(domain, degree=1, value_shape=(3,), name="state")
    indices = field._array_indices()
    if permuted:
        field._array_indices_cache = indices[::-1].copy()
        field._identity_layout_cache = False
    expected = np.arange(field.values.size).reshape(field.values.shape)
    field.assign(expected)
    snapshot = _state_value_snapshot(field)
    assert not np.shares_memory(snapshot, field.function.x.array)
    np.testing.assert_array_equal(snapshot, expected)
    field.assign(np.zeros_like(expected))
    np.testing.assert_array_equal(snapshot, expected)


class ViewState:
    def __init__(self, values):
        self.values = values

    def assign(self, values):
        self.values[:] = values


def test_open_protocol_snapshots_all_sources_before_cross_aliased_assignment():
    a = ViewState(np.array([1., 2.]))
    b = ViewState(np.array([3., 4.]))
    transaction = QuadratureTransaction(
        committed={"a": a, "b": b}, trial={"a": b, "b": a}, schema="test"
    )
    transaction.commit()
    np.testing.assert_array_equal(a.values, [3., 4.])
    np.testing.assert_array_equal(b.values, [1., 2.])
    snapshot = transaction.snapshot()
    transaction.rollback()
    np.testing.assert_array_equal(a.values, [1., 2.])
    np.testing.assert_array_equal(b.values, [3., 4.])
    np.testing.assert_array_equal(snapshot["a"], [3., 4.])


def test_subclass_live_view_keeps_defensive_copy():
    class LiveField(QuadratureField):
        @property
        def values(self):
            return self.points

    field = LiveField(None, np.array([1., 2.]), np.ones(2), ())
    snapshot = _state_value_snapshot(field)
    field.points[:] = 0
    np.testing.assert_array_equal(snapshot, [1., 2.])
