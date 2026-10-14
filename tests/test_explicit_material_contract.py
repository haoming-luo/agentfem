from dataclasses import replace

import numpy as np
import pytest

from agentfem.constitutive._explicit_material import (
    ExplicitMaterialBatchOutput,
    validated_explicit_material_update,
)
from agentfem.constitutive.material_array_batch import MaterialPointArrayBatchInput
from agentfem.constitutive.user_material import (
    MaterialStateSchema,
    MaterialStateVariable,
)


SCHEMA = MaterialStateSchema(
    "explicit_test", (MaterialStateVariable("history", shape=(2,)),)
)


def response(count=2):
    return ExplicitMaterialBatchOutput(
        np.tile(np.eye(3), (count, 1, 1)),
        np.zeros((count, 2)),
        SCHEMA,
        np.full(count, -0.1),
    )


def request():
    return MaterialPointArrayBatchInput(
        np.tile(np.eye(3), (2, 1, 1)),
        np.tile(np.eye(3), (2, 1, 1)),
        np.zeros((2, 2)),
        SCHEMA,
        0.0,
        0.1,
    )


def test_work_can_be_negative_and_does_not_become_stored_energy():
    result = response()
    assert result.stored_energy_density is None
    assert result.dissipation_density_increment is None
    assert not result.energy_contract()["stored_energy_available"]
    assert not result.energy_contract()["tangent_available"]
    assert not result.state_new.flags.writeable
    assert not result.internal_work_density_increment.flags.writeable


def test_owned_response_reuses_checked_bytes_and_detaches_metadata(monkeypatch):
    from agentfem.constitutive._explicit_material import _owned_validated_response
    original = response()
    def reject_rescan(*args, **kwargs):
        raise AssertionError("Unchanged immutable response should not be rescanned.")
    with monkeypatch.context() as context:
        context.setattr(np, "isfinite", reject_rescan)
        checked = _owned_validated_response(original)
    assert checked is not original
    assert np.shares_memory(checked.state_new, original.state_new)
    original.state_new.shape = (4,)
    assert checked.state_new.shape == (2, 2)
    with pytest.raises(ValueError):
        checked.state_new.setflags(write=True)
    with pytest.raises(ValueError):
        _owned_validated_response(original)


def test_owned_response_rechecks_replaced_array_and_dtype():
    from agentfem.constitutive._explicit_material import _owned_validated_response
    original = response()
    object.__setattr__(original, "state_new", np.full((2, 2), np.nan))
    with pytest.raises(ValueError):
        _owned_validated_response(original)
    original = response()
    original.state_new.dtype = np.int64
    # Revalidation may admit real finite converted values, but never reuse the
    # old float payload certificate after a dtype reinterpretation.
    expected = replace(original)
    actual = _owned_validated_response(original)
    np.testing.assert_array_equal(actual.state_new, expected.state_new)


def test_response_revalidation_reuses_immutable_bytes_not_array_metadata():
    original = response()
    checked = replace(original)
    assert np.shares_memory(original.state_new, checked.state_new)
    assert original.state_new is not checked.state_new
    original.state_new.shape = (4,)
    assert checked.state_new.shape == (2, 2)
    with pytest.raises(ValueError):
        checked.state_new.setflags(write=True)
    # Noncontiguous external storage is copied and normalized, never aliased.
    raw = np.arange(8., dtype=float).reshape(2, 4)[:, ::2]
    selected = replace(response(), state_new=raw)
    before = selected.state_new.copy()
    raw[:] = -1
    np.testing.assert_array_equal(selected.state_new, before)


def test_decomposition_is_optional_not_thermodynamic_certification():
    result = replace(
        response(),
        stored_energy_density=np.ones(2),
        dissipation_density_increment=np.full(2, -0.5),
    )
    assert result.energy_contract()["dissipation_available"]
    assert (
        result.energy_contract()["thermodynamic_validation"]
        == "not_established_by_response"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("state_new", np.zeros((2, 3))),
        ("cauchy_stress", np.full((2, 3, 3), np.nan)),
        ("internal_work_density_increment", np.ones(3)),
        ("stored_energy_density", np.array([1.0, np.inf])),
        ("requested_time_scale", np.array([0.0, 1.0])),
        ("requested_time_scale", np.array([1.0, 1.1])),
    ],
)
def test_bad_output_rejected(field, value):
    with pytest.raises(ValueError):
        replace(response(), **{field: value})


def test_unsymmetric_stress_and_empty_partition():
    stress = np.tile(np.eye(3), (2, 1, 1))
    stress[:, 0, 1] = 1
    with pytest.raises(ValueError):
        replace(response(), cauchy_stress=stress)
    assert response(0).point_count == 0


@pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)])
@pytest.mark.parametrize("scale", [1e-200, 1., 1e200])
def test_unique_pair_symmetry_matches_full_tensor_threshold(pair, scale):
    for perturbation in (0., np.nextafter(1e-12, 0.), 1e-12,
                         np.nextafter(1e-12, np.inf), 2e-12):
        stress = np.tile(np.eye(3) * scale, (7, 1, 1))
        stress[-1, pair[0], pair[1]] = perturbation * scale
        bound = 1e-12 * np.maximum(np.max(np.abs(stress), axis=(1, 2)),
                                  np.finfo(float).tiny)
        rejected = np.any(np.max(np.abs(stress-stress.transpose(0, 2, 1)),
                                 axis=(1, 2)) > bound)
        if rejected:
            with pytest.raises(ValueError, match="symmetric"):
                replace(response(7), cauchy_stress=stress)
        else:
            np.testing.assert_array_equal(
                replace(response(7), cauchy_stress=stress).cauchy_stress, stress)


def test_provider_boundary_revalidates_and_rejects_wrong_count():
    class Provider:
        state_schema = SCHEMA

        def update_explicit_array_batch(self, value):
            assert not value.state_old.flags.writeable
            return response()

    provider = Provider()
    assert validated_explicit_material_update(provider, request()).point_count == 2
    provider.update_explicit_array_batch = lambda value: response(1)
    with pytest.raises(ValueError, match="point count"):
        validated_explicit_material_update(provider, request())
    bad = response()
    object.__setattr__(bad, "internal_work_density_increment", np.array([1.0, np.nan]))
    provider.update_explicit_array_batch = lambda value: bad
    with pytest.raises(ValueError):
        validated_explicit_material_update(provider, request())
