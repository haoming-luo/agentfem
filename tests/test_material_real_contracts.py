"""Real material contracts must never silently discard imaginary components."""

from dataclasses import replace

import numpy as np
import pytest

from agentfem import constitutive as c


SCHEMA = c.MaterialStateSchema("real_state", (c.MaterialStateVariable("history"),))


@pytest.mark.parametrize(
    "field",
    [
        "deformation_gradient_old",
        "deformation_gradient_new",
        "properties",
        "state_old",
        "field_variables",
    ],
)
def test_finite_strain_point_rejects_complex_arrays(field):
    point = c.MaterialPointInput(
        np.eye(3), np.eye(3), 0.0, 0.1, [1.0], [0.0], SCHEMA, field_variables=[0.0]
    )
    with pytest.raises(ValueError, match="real"):
        replace(point, **{field: getattr(point, field).astype(complex) + 1j})


@pytest.mark.parametrize("field", ["cauchy_stress", "consistent_tangent", "state_new"])
def test_finite_strain_response_rejects_complex_arrays(field):
    response = c.MaterialPointOutput(
        np.eye(3),
        np.eye(9),
        [0.0],
        state_schema=SCHEMA,
        tangent_convention=c.MaterialTangentConvention.first_piola_deformation_gradient(),
    )
    with pytest.raises(ValueError, match="real"):
        replace(response, **{field: getattr(response, field).astype(complex) + 1j})


def test_state_initializer_and_declared_tangent_reject_complex():
    with pytest.raises(ValueError, match="real"):
        c.MaterialStateVariable(
            "tensor", shape=(3, 3), initial_value=np.eye(3, dtype=complex)
        )
    with pytest.raises(ValueError, match="real"):
        SCHEMA.validate(np.array([1j]))
    with pytest.raises(ValueError, match="real"):
        c.MaterialTangentConvention.first_piola_deformation_gradient().validate(
            np.eye(9, dtype=complex)
        )


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("field", ["cauchy_stress", "consistent_tangent", "state_new"])
def test_small_strain_response_rejects_complex_arrays(batch, field):
    if batch:
        response = c.SmallStrainMaterialPointBatchOutput(
            np.eye(3)[None], np.eye(6)[None], np.zeros((1, 1)), SCHEMA
        )
    else:
        response = c.SmallStrainMaterialPointOutput(np.eye(3), np.eye(6), [0.0], SCHEMA)
    with pytest.raises(ValueError, match="real"):
        replace(response, **{field: getattr(response, field).astype(complex) + 1j})


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("field", ["strain_old", "strain_new", "state_old"])
def test_small_strain_request_rejects_complex_arrays(batch, field):
    parameters = c.MaterialParameterSchema("empty", ())
    if batch:
        point = c.SmallStrainMaterialPointBatchInput(
            np.eye(3)[None],
            np.eye(3)[None],
            0.0,
            0.1,
            {},
            np.zeros((1, 1)),
            SCHEMA,
            parameters,
        )
    else:
        point = c.SmallStrainMaterialPointInput(
            np.eye(3), np.eye(3), 0.0, 0.1, {}, [0.0], SCHEMA, parameters
        )
    with pytest.raises(ValueError, match="real"):
        replace(point, **{field: getattr(point, field).astype(complex) + 1j})


@pytest.mark.parametrize(
    "field", ["time", "time_increment", "temperature", "temperature_increment"]
)
@pytest.mark.parametrize("value", [1j, np.complex128(1 + 1j), [0.1], np.nan, np.inf])
def test_point_scalar_channels_require_finite_real_scalars(field, value):
    finite = c.MaterialPointInput(np.eye(3), np.eye(3), 0.0, 0.1, [], [0.0], SCHEMA)
    small = c.SmallStrainMaterialPointInput(
        np.eye(3),
        np.eye(3),
        0.0,
        0.1,
        {},
        [0.0],
        SCHEMA,
        c.MaterialParameterSchema("empty", ()),
    )
    for point in (finite, small):
        with pytest.raises(ValueError, match="real"):
            replace(point, **{field: value})


@pytest.mark.parametrize("field", ["time", "time_increment"])
def test_columnar_time_is_real_scalar(field):
    point = c.MaterialPointArrayBatchInput(
        np.eye(3)[None],
        np.eye(3)[None],
        np.zeros((1, 1)),
        SCHEMA,
        0,
        0.1,
    )
    for value in (np.complex128(0.1 + 1j), [0.1], np.inf):
        with pytest.raises(ValueError, match="real"):
            replace(point, **{field: value})


@pytest.mark.parametrize(
    "field",
    ["strain_energy_density", "dissipation_density_increment", "suggested_time_scale"],
)
def test_response_scalar_channels_never_discard_imaginary_part(field):
    responses = (
        c.MaterialPointOutput(
            np.eye(3),
            np.eye(9),
            [0.0],
            state_schema=SCHEMA,
            tangent_convention=c.MaterialTangentConvention.first_piola_deformation_gradient(),
        ),
        c.SmallStrainMaterialPointOutput(np.eye(3), np.eye(6), [0.0], SCHEMA),
    )
    for response in responses:
        selected = (
            "stored_energy_density"
            if field == "strain_energy_density"
            and isinstance(response, c.SmallStrainMaterialPointOutput)
            else field
        )
        with pytest.raises(ValueError, match="real"):
            replace(response, **{selected: np.complex128(1 + 1j)})


def test_deformation_gradient_determinant_must_be_finite():
    with pytest.raises(ValueError, match="finite positive determinant"):
        c.MaterialPointInput(np.eye(3), 1e200 * np.eye(3), 0.0, 0.1, [], [0.0], SCHEMA)


@pytest.mark.parametrize(
    "field",
    [
        "deformation_gradient_new",
        "properties",
        "field_variables",
        "time",
        "time_increment",
    ],
)
def test_driver_complex_input_rejected_before_provider_and_preserves_state(field):
    from dolfinx import mesh
    from mpi4py import MPI

    law = c.finite_strain_j2_logarithmic(
        young=100, poisson=0.3, yield_stress=1, hardening_modulus=2
    )
    domain = mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    state = c.MaterialQuadratureState.create(domain, law.state_schema, degree=1)
    initial = state.committed_state_vectors().copy()

    class NeverCalled:
        name = "unreached provider"
        state_schema = law.state_schema
        tangent_convention = law.tangent_convention

        def update(self, point):
            raise AssertionError("Invalid input reached provider")

    arguments = dict(
        deformation_gradient_old=np.eye(3),
        deformation_gradient_new=np.eye(3),
        time=0.0,
        time_increment=0.1,
    )
    arguments[field] = (
        np.eye(3, dtype=complex)
        if field == "deformation_gradient_new"
        else np.complex128(1 + 1j)
    )
    with pytest.raises(RuntimeError, match="real"):
        c.update_material_points(NeverCalled(), state, **arguments)
    np.testing.assert_array_equal(state.committed_state_vectors(), initial)
    np.testing.assert_array_equal(state.trial_state_vectors(), initial)
