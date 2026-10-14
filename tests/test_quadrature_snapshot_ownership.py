import numpy as np
import pytest
from dolfinx import mesh
from mpi4py import MPI

from agentfem.constitutive.quadrature import (
    QuadratureField,
    QuadratureTransaction,
    _state_value_snapshot,
    _disjoint_serial_state_views,
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


def test_disjoint_transfer_borrows_only_until_assignment():
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1,
                                  cell_type=mesh.CellType.hexahedron)
    def make(name):
        return QuadratureField.create(domain, degree=1, value_shape=(3,), name=name)
    source, target = make("trial"), make("accepted")
    expected = np.arange(source.values.size).reshape(source.values.shape)
    source.assign(expected)
    views = _disjoint_serial_state_views({"s": source}, {"s": target})
    assert views is not None
    assert np.shares_memory(views["s"], source.function.x.array)
    transaction = QuadratureTransaction({"s": target}, {"s": source}, "transfer")
    transaction.commit()
    source.assign(np.zeros_like(expected))
    np.testing.assert_array_equal(target.values, expected)
    snapshot = transaction.snapshot()
    transaction.rollback()
    np.testing.assert_array_equal(source.values, expected)
    target.assign(np.zeros_like(expected))
    np.testing.assert_array_equal(snapshot["s"], expected)


def test_concrete_cross_alias_and_permutation_use_snapshots():
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1,
                                  cell_type=mesh.CellType.hexahedron)
    a, b = [QuadratureField.create(domain, degree=1, name=n) for n in ("a", "b")]
    a.assign(np.ones(a.point_count))
    b.assign(np.full(b.point_count, 2.))
    sources, targets = {"a": a, "b": b}, {"a": b, "b": a}
    assert _disjoint_serial_state_views(sources, targets) is None
    QuadratureTransaction(targets, sources, "swap").commit()
    np.testing.assert_array_equal(a.values, 2.)
    np.testing.assert_array_equal(b.values, 1.)
    a._array_indices_cache = a._array_indices()[::-1].copy()
    a._identity_layout_cache = False
    assert _disjoint_serial_state_views({"a": a}, {"a": b}) is None


def test_custom_assignment_keeps_detached_argument(monkeypatch):
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1,
                                  cell_type=mesh.CellType.hexahedron)
    a, b = [QuadratureField.create(domain, degree=1, name=n) for n in ("a", "b")]
    captured = []
    assign = b.assign
    def retaining_assign(values):
        captured.append(values)
        assign(values)
    monkeypatch.setattr(b, "assign", retaining_assign)
    assert _disjoint_serial_state_views({"a": a}, {"a": b}) is None
    QuadratureTransaction({"a": b}, {"a": a}, "custom").commit()
    assert not np.shares_memory(captured[0], a.function.x.array)
