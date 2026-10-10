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
