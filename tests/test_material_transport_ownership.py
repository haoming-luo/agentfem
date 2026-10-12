"""Immutable batch slices are reusable; writable backing is never borrowed."""

import numpy as np
import pytest

from agentfem.constitutive.material_array_batch import _array


@pytest.mark.parametrize("selection", [slice(0, 3), slice(2, 6), slice(6, 6)])
def test_contiguous_immutable_slice_reuses_payload_with_independent_metadata(selection):
    whole = _array(np.arange(42.).reshape(7, 6), (7, 6), "whole")
    selected = whole[selection]
    expected = selected.copy()
    result = _array(selected, selected.shape, "slice")
    if result.size:
        assert np.shares_memory(result, whole)
    with pytest.raises(ValueError):
        result.setflags(write=True)
    selected.shape = (selected.size,)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == expected.shape


@pytest.mark.parametrize("kind", ["writable", "readonly_view", "readonly_memoryview",
                                  "strided", "reversed"])
def test_untrusted_or_noncontiguous_payload_remains_detached(kind):
    source = np.arange(42.).reshape(7, 6)
    if kind == "readonly_view":
        candidate = source.view()
        candidate.setflags(write=False)
    elif kind == "readonly_memoryview":
        candidate = np.asarray(memoryview(source).toreadonly())
    elif kind in ("strided", "reversed"):
        frozen = _array(source, source.shape, "whole")
        candidate = frozen[::2] if kind == "strided" else frozen[::-1]
    else:
        candidate = source
    expected = candidate.copy()
    result = _array(candidate, candidate.shape, "candidate")
    assert not np.shares_memory(result, candidate)
    source[:] = -3
    np.testing.assert_array_equal(result, expected)
    with pytest.raises(ValueError):
        result.setflags(write=True)


def test_byte_offset_and_dtype_conversion():
    # A bytes owner need not begin on a float boundary.
    owner = b"x" + np.arange(12., dtype=float).tobytes() + b"tail"
    selected = np.frombuffer(owner, dtype=float, count=12, offset=1).reshape(4, 3)[1:3]
    result = _array(selected, (2, 3), "offset")
    assert np.shares_memory(selected, result)
    np.testing.assert_array_equal(result, selected)
    integers = np.arange(6, dtype=np.int64).reshape(2, 3)
    result = _array(integers, (2, 3), "converted")
    assert result.dtype == np.float64
    assert not np.shares_memory(result, integers)


def test_immutable_storage_does_not_skip_validation():
    for data, shape in [(np.array([np.nan]), (1,)), (np.ones(2), (1,))]:
        candidate = np.frombuffer(data.tobytes(), dtype=float)
        with pytest.raises(ValueError):
            _array(candidate, shape, "invalid")


def test_single_block_history_freezes_without_intermediate_values_copy(monkeypatch):
    from agentfem.constitutive._explicit_material_driver import _immutable_committed_history
    from agentfem.constitutive.quadrature import QuadratureField
    from test_explicit_material_quadrature import fixture

    _, state, _ = fixture()
    expected = state.committed_state_vectors()

    def forbidden(_):
        raise AssertionError("History transport must not first materialize values")

    monkeypatch.setattr(QuadratureField, "values", property(forbidden))
    result = _immutable_committed_history(state)
    next(iter(state.committed.values())).function.x.array[:] = 7
    np.testing.assert_array_equal(result, expected)
    with pytest.raises(ValueError):
        result.setflags(write=True)


def test_open_history_provider_keeps_defensive_copy():
    from agentfem.constitutive._explicit_material_driver import _immutable_committed_history
    from agentfem.constitutive.user_material import MaterialStateSchema, MaterialStateVariable

    class History:
        state_schema = MaterialStateSchema("test", (MaterialStateVariable("x"),))
        data = np.ones((2, 1))

        def committed_state_vectors(self):
            return self.data

    owner = History()
    result = _immutable_committed_history(owner)
    owner.data[:] = 9
    np.testing.assert_array_equal(result, 1)
    with pytest.raises(ValueError):
        result.setflags(write=True)


def test_multiple_named_history_blocks_keep_schema_order():
    from dolfinx import mesh
    from mpi4py import MPI
    from agentfem.constitutive._explicit_material_driver import _immutable_committed_history
    from agentfem.constitutive.quadrature import MaterialQuadratureState
    from agentfem.constitutive.user_material import MaterialStateSchema, MaterialStateVariable

    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1,
                                   cell_type=mesh.CellType.hexahedron)
    schema = MaterialStateSchema("two_blocks", (
        MaterialStateVariable("scalar"), MaterialStateVariable("tensor", shape=(3, 3)),
    ))
    state = MaterialQuadratureState.create(domain, schema, degree=1)
    state.assign_trial_state_vectors(np.arange(10.).reshape(1, 10))
    state.commit()
    expected = state.committed_state_vectors()
    actual = _immutable_committed_history(state)
    for field in state.committed.values():
        field.function.x.array[:] = 42
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError):
        actual.setflags(write=True)


def test_material_chunks_share_one_frozen_history_allocation(monkeypatch):
    from agentfem.constitutive import _explicit_material_driver as driver
    from test_explicit_material_quadrature import fixture

    material, state, options = fixture()
    captured = []
    original = driver.validated_explicit_material_update

    def capture(material, request):
        captured.append(request.state_old)
        return original(material, request)

    monkeypatch.setattr(driver, "validated_explicit_material_update", capture)
    driver._prepare_explicit_material_points(material, state, max_batch_points=1, **options)
    assert len(captured) == 2

    def owner(array):
        while isinstance(array, np.ndarray):
            array = array.base
        return array

    assert isinstance(owner(captured[0]), bytes)
    assert owner(captured[0]) is owner(captured[1])
    for array in captured:
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_permuted_history_uses_cell_order_not_storage_order():
    from agentfem.constitutive._explicit_material_driver import _immutable_committed_history
    from test_explicit_material_quadrature import fixture

    _, state, _ = fixture()
    field = next(iter(state.committed.values()))
    indices = field._array_indices().copy()[::-1]
    field._array_indices_cache = indices
    field._identity_layout_cache = False
    field.function.x.array[:] = np.arange(field.function.x.array.size)
    expected = state.committed_state_vectors()
    actual = _immutable_committed_history(state)
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, field.function.x.array)


def test_explicit_response_revalidation_preserves_sliced_storage():
    from agentfem.constitutive._explicit_material import (
        ExplicitMaterialBatchOutput, _owned_validated_response,
    )
    from agentfem.constitutive.user_material import MaterialStateSchema, MaterialStateVariable
    schema = MaterialStateSchema("slice", (MaterialStateVariable("x"),))
    whole = ExplicitMaterialBatchOutput(np.zeros((4, 3, 3)), np.arange(4.).reshape(4, 1),
                                       schema, np.arange(4.))
    part = ExplicitMaterialBatchOutput(whole.cauchy_stress[1:3], whole.state_new[1:3],
                                      schema, whole.internal_work_density_increment[1:3])
    copied = _owned_validated_response(part)
    np.testing.assert_array_equal(copied.state_new, [[1.], [2.]])
    np.testing.assert_array_equal(copied.internal_work_density_increment, [1., 2.])
    assert np.shares_memory(copied.state_new, whole.state_new)
    part.state_new.shape = (2,)
    assert copied.state_new.shape == (2, 1)
