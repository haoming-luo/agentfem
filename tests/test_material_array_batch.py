# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import numpy as np
import pytest

from agentfem import constitutive
from agentfem.constitutive.material_array_batch import (
    MaterialPointArrayBatchInput,
    MaterialPointArrayBatchOutput,
    validated_material_array_batch_update,
)


def fixture(count=7):
    material = constitutive.finite_strain_j2_logarithmic(
        young=190000,
        poisson=0.3,
        yield_stress=280,
        hardening_modulus=1100,
    )
    gradient = np.tile(np.eye(3), (count, 1, 1))
    gradient[:, 0, 0] = np.linspace(1, 1.08, count)
    gradient[:, 1, 1] = gradient[:, 2, 2] = gradient[:, 0, 0] ** -0.5
    request = MaterialPointArrayBatchInput(
        deformation_gradient_old=np.tile(np.eye(3), (count, 1, 1)),
        deformation_gradient_new=gradient,
        state_old=np.tile(material.state_schema.initial_state(), (count, 1)),
        state_schema=material.state_schema,
        time=0,
        time_increment=0.1,
    )
    return material, request


@pytest.mark.parametrize("history", [False, True])
def test_direction_batched_tangent_crosses_chunks_and_matches_fixed_state_difference(
    history,
):
    material, request = fixture(count=1031)
    if history:
        first = material.update_array_batch(request)
        gradient = request.deformation_gradient_new.copy()
        gradient[:, 0, 1] += np.linspace(0.04, 0.06, request.point_count)
        angle = 0.7
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        request = replace(
            request,
            deformation_gradient_new=rotation @ gradient,
            deformation_gradient_old=request.deformation_gradient_new,
            state_old=first.state_new,
            time=0.1,
        )
    else:
        # Repeated principal stretches: exercise the divided-difference limit.
        request = replace(
            request, deformation_gradient_new=np.tile(1.03 * np.eye(3), (1031, 1, 1))
        )
    expected = replace(
        material, tangent_evaluation="central_difference", tangent_relative_step=2e-6
    ).update_array_batch(request)
    actual = material.update_array_batch(request)
    errors = np.linalg.norm(
        actual.consistent_tangent - expected.consistent_tangent, axis=(1, 2)
    ) / np.linalg.norm(expected.consistent_tangent, axis=(1, 2))
    assert np.max(errors) < 2e-5
    np.testing.assert_array_equal(actual.state_new, expected.state_new)
    np.testing.assert_array_equal(actual.cauchy_stress, expected.cauchy_stress)


@pytest.mark.parametrize("stage", ["loading", "unloading", "rotated"])
def test_columnar_matches_ordered_point_protocol(stage):
    material, request = fixture()
    if stage != "loading":
        first = validated_material_array_batch_update(material, request)
        gradient = request.deformation_gradient_new.copy()
        if stage == "unloading":
            gradient[:, 0, 0] *= 0.999
        else:
            angle = 0.7
            rotation = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            gradient = rotation @ gradient
        request = replace(
            request,
            deformation_gradient_old=request.deformation_gradient_new,
            deformation_gradient_new=gradient,
            state_old=first.state_new,
            time=0.1,
        )
    actual = validated_material_array_batch_update(material, request)
    points = tuple(
        constitutive.MaterialPointInput(
            deformation_gradient_old=request.deformation_gradient_old[i],
            deformation_gradient_new=request.deformation_gradient_new[i],
            state_old=request.state_old[i],
            state_schema=request.state_schema,
            properties=(),
            time=request.time,
            time_increment=request.time_increment,
        )
        for i in range(request.point_count)
    )
    expected = material.update_batch(constitutive.MaterialPointBatchInput(points))
    for name in (
        "cauchy_stress",
        "consistent_tangent",
        "state_new",
        "strain_energy_density",
        "dissipation_density_increment",
        "suggested_time_scale",
    ):
        np.testing.assert_allclose(
            getattr(actual, name),
            [getattr(point, name) for point in expected.responses],
            rtol=2e-12,
            atol=2e-12,
        )
    for name, values in actual.stored_energy_density_components.items():
        np.testing.assert_allclose(
            values,
            [r.stored_energy_density_components[name] for r in expected.responses],
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("deformation_gradient_old", np.zeros((7, 3, 3))),
        ("deformation_gradient_new", np.full((7, 3, 3), np.nan)),
        ("state_old", np.zeros((7, 1))),
        ("time_increment", 0),
        ("time", np.nan),
        ("temperature", np.zeros(6)),
        ("temperature_increment", np.full(7, np.inf)),
        ("field_variables", np.zeros((6, 2))),
        ("properties", [np.nan]),
    ],
)
def test_columnar_input_validation(field, value):
    _, request = fixture()
    with pytest.raises(ValueError):
        replace(request, **{field: value})


@pytest.mark.parametrize(
    "field",
    ["deformation_gradient_old", "deformation_gradient_new", "state_old", "properties"],
)
def test_columnar_input_rejects_complex_arrays_without_discarding_imaginary_part(field):
    _, request = fixture()
    values = np.asarray(getattr(request, field), dtype=complex)
    with pytest.raises(ValueError, match="must be real"):
        replace(request, **{field: values})


@pytest.mark.parametrize(
    "field",
    ["cauchy_stress", "consistent_tangent", "state_new", "strain_energy_density"],
)
def test_columnar_output_rejects_complex_arrays(field):
    material, request = fixture()
    response = validated_material_array_batch_update(material, request)
    values = np.asarray(getattr(response, field), dtype=complex)
    with pytest.raises(ValueError, match="must be real"):
        replace(response, **{field: values})


def test_arrays_are_owned_and_readonly():
    material, request = fixture()
    original = request.deformation_gradient_new.copy()
    copied = replace(request, deformation_gradient_new=original)
    original[:] = 0
    assert np.all(copied.deformation_gradient_new[:, 0, 0] > 0)
    with pytest.raises(ValueError):
        copied.state_old[0, 0] = 7
    result = material.update_array_batch(copied)
    with pytest.raises(ValueError):
        result.state_new[0, 0] = 8


@pytest.mark.parametrize(
    "field", ["stress_symmetry", "nan_tangent", "state_size", "component_sum", "scale"]
)
def test_columnar_output_validation(field):
    material, request = fixture()
    output = material.update_array_batch(request)
    if field == "stress_symmetry":
        stress = output.cauchy_stress.copy()
        stress[0, 0, 1] += 1
        change = dict(cauchy_stress=stress)
    elif field == "nan_tangent":
        tangent = output.consistent_tangent.copy()
        tangent[-1, 0, 0] = np.nan
        change = dict(consistent_tangent=tangent)
    elif field == "state_size":
        change = dict(state_new=output.state_new[:, :-1])
    elif field == "scale":
        change = dict(suggested_time_scale=np.zeros(request.point_count))
    else:
        change = dict(
            stored_energy_density_components={"ELENER": np.ones(request.point_count)}
        )
    with pytest.raises(ValueError):
        replace(output, **change)


@pytest.mark.parametrize("failure", ["count", "schema", "convention", "type"])
def test_boundary_rejects_changed_provider_contract(failure):
    material, request = fixture()

    class Provider:
        name = "invalid columnar provider"
        state_schema = material.state_schema
        tangent_convention = material.tangent_convention

        def update(self, point):
            raise AssertionError("A rejected array call must not fall back.")

        def update_array_batch(self, data):
            if failure == "type":
                return {}
            if failure == "count":
                data = replace(
                    data,
                    deformation_gradient_old=data.deformation_gradient_old[:-1],
                    deformation_gradient_new=data.deformation_gradient_new[:-1],
                    state_old=data.state_old[:-1],
                )
            result = material.update_array_batch(data)
            if failure == "schema":
                return replace(
                    result, state_schema=replace(result.state_schema, version="other")
                )
            if failure == "convention":
                return replace(
                    result,
                    tangent_convention=replace(
                        result.tangent_convention, symmetric=True
                    ),
                )
            return result

    with pytest.raises((TypeError, ValueError)):
        validated_material_array_batch_update(Provider(), request)


def test_optional_energy_channels_are_not_invented():
    material, request = fixture()
    source = material.update_array_batch(request)
    output = MaterialPointArrayBatchOutput(
        cauchy_stress=source.cauchy_stress,
        consistent_tangent=source.consistent_tangent,
        state_new=source.state_new,
        state_schema=source.state_schema,
        tangent_convention=source.tangent_convention,
    )
    assert output.strain_energy_density is None
    assert output.dissipation_density_increment is None
    np.testing.assert_array_equal(output.suggested_time_scale, 1)


def test_generic_columnar_provider_preserves_named_schema_and_optional_inputs():
    schema = constitutive.MaterialStateSchema(
        name="independent-columnar-provider",
        variables=(constitutive.MaterialStateVariable(name="history", shape=(2,)),),
    )
    convention = (
        constitutive.MaterialTangentConvention.first_piola_deformation_gradient()
    )
    request = MaterialPointArrayBatchInput(
        deformation_gradient_old=np.tile(np.eye(3), (2, 1, 1)),
        deformation_gradient_new=np.tile(np.eye(3), (2, 1, 1)),
        state_old=np.zeros((2, 2)),
        state_schema=schema,
        time=1,
        time_increment=0.2,
        properties=np.array([2, 3]),
        temperature=np.array([300, 310]),
        temperature_increment=np.array([1, 2]),
        field_variables=np.array([[4], [5]]),
    )

    class Provider:
        name = "transport-only independent test"
        state_schema = schema
        tangent_convention = convention

        def update(self, point):
            raise AssertionError("No scalar fallback expected")

        def update_array_batch(self, data):
            np.testing.assert_array_equal(data.properties, [2, 3])
            np.testing.assert_array_equal(data.temperature, [300, 310])
            np.testing.assert_array_equal(data.temperature_increment, [1, 2])
            np.testing.assert_array_equal(data.field_variables, [[4], [5]])
            return MaterialPointArrayBatchOutput(
                cauchy_stress=np.zeros((2, 3, 3)),
                consistent_tangent=np.zeros((2, 9, 9)),
                state_new=data.state_old + 1,
                state_schema=schema,
                tangent_convention=convention,
                suggested_time_scale=np.array([1, 0.5]),
            )

    result = validated_material_array_batch_update(Provider(), request)
    np.testing.assert_array_equal(result.state_new, np.ones((2, 2)))
    np.testing.assert_array_equal(request.state_old, np.zeros((2, 2)))
    np.testing.assert_array_equal(result.suggested_time_scale, [1, 0.5])
