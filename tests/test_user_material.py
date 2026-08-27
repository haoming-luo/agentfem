from __future__ import annotations

import numpy as np
import pytest

from agentfem.constitutive.user_material import (
    AbaqusUserMaterialBridge,
    MaterialPointInput,
    MaterialPointOutput,
    MaterialStateSchema,
    MaterialStateVariable,
    MaterialTangentConvention,
    validated_material_update,
)


def test_material_point_contract_copies_and_validates_state():
    point = MaterialPointInput(
        deformation_gradient_old=np.eye(3),
        deformation_gradient_new=np.diag([1.1, 1.0, 1.0]),
        time=0.0,
        time_increment=0.1,
        properties=[1000.0, 0.3],
        state_old=[0.0],
    )
    response = MaterialPointOutput(
        cauchy_stress=np.zeros((3, 3)),
        consistent_tangent=np.eye(6),
        state_new=[0.2],
    )

    assert np.linalg.det(point.deformation_gradient_new) == pytest.approx(1.1)
    assert response.state_new.tolist() == [0.2]
    assert not response.global_newton_contract_complete
    with pytest.raises(ValueError, match="tangent_convention, state_schema"):
        response.require_global_newton_contract()


def test_material_point_contract_rejects_inverted_gradient():
    with pytest.raises(ValueError, match="positive determinant"):
        MaterialPointInput(
            deformation_gradient_old=np.eye(3),
            deformation_gradient_new=np.diag([-1.0, 1.0, 1.0]),
            time=0.0,
            time_increment=0.1,
            properties=[],
            state_old=[],
        )


def test_abaqus_bridge_is_explicitly_a_nonexecutable_specification():
    bridge = AbaqusUserMaterialBridge(
        kind="umat",
        source="material.for",
        material_name="matrix",
        property_count=2,
        state_variable_count=6,
    )

    assert bridge.kind == "UMAT"
    assert bridge.executable is False
    assert "quadrature state driver" in bridge.summary()["required_runtime"]
    assert bridge.summary()["tangent_convention"]["stress_measure"] == "kirchhoff"


def test_named_material_state_schema_initializes_validates_and_unpacks():
    schema = MaterialStateSchema(
        "ductile_state",
        (
            MaterialStateVariable("equivalent_plastic_strain", unit="1"),
            MaterialStateVariable(
                "plastic_deformation_gradient",
                shape=(3, 3),
                initial_value=np.eye(3),
                unit="1",
            ),
        ),
        version="1.0.0",
    )

    state = schema.initial_state()
    assert schema.size == 10
    assert schema.identity == "ductile_state@1.0.0"
    unpacked = schema.unpack(state)
    assert unpacked["equivalent_plastic_strain"] == pytest.approx(0.0)
    np.testing.assert_allclose(
        unpacked["plastic_deformation_gradient"],
        np.eye(3),
    )
    assert schema.summary()["variables"][1]["initial_value"] == np.eye(3).tolist()
    with pytest.raises(ValueError, match="requires 10"):
        schema.validate(np.zeros(9))
    with pytest.raises(ValueError, match="declared shape"):
        MaterialStateVariable(
            "invalid_tensor",
            shape=(3, 3),
            initial_value=np.zeros((2, 2)),
        )


def test_tangent_convention_declares_measure_storage_and_objective_rate():
    umat = MaterialTangentConvention.abaqus_umat()
    first_piola = MaterialTangentConvention.first_piola_deformation_gradient()

    assert umat.array_shape == (6, 6)
    assert umat.shear_convention == "engineering"
    assert umat.objective_rate == "abaqus_umat_corotational"
    assert first_piola.array_shape == (9, 9)
    assert first_piola.stress_measure == "first_piola"
    with pytest.raises(ValueError, match="objective-rate"):
        MaterialTangentConvention(
            stress_measure="cauchy",
            kinematic_measure="rate_of_deformation",
            configuration="current",
            storage="matrix_6x6",
            component_order=("11", "22", "33", "12", "13", "23"),
        )


def test_validated_material_update_fails_closed_on_contract_drift():
    schema = MaterialStateSchema(
        "scalar_damage",
        (MaterialStateVariable("damage", unit="1"),),
    )
    tangent = MaterialTangentConvention.abaqus_umat()

    class Material:
        name = "test material"
        state_schema = schema
        tangent_convention = tangent

        def update(self, point):
            return MaterialPointOutput(
                cauchy_stress=np.zeros((3, 3)),
                consistent_tangent=np.eye(6),
                state_new=point.state_old + 0.1,
                tangent_convention=self.tangent_convention,
                state_schema=self.state_schema,
            )

    point = MaterialPointInput(
        deformation_gradient_old=np.eye(3),
        deformation_gradient_new=np.diag([1.1, 1.0, 1.0]),
        time=0.0,
        time_increment=0.1,
        properties=[],
        state_old=[0.0],
        state_schema=schema,
    )
    response = validated_material_update(Material(), point)
    assert response.global_newton_contract_complete
    assert response.state_new[0] == pytest.approx(0.1)

    class DriftingMaterial(Material):
        def update(self, point):
            return MaterialPointOutput(
                cauchy_stress=np.zeros((3, 3)),
                consistent_tangent=np.eye(6),
                state_new=[],
                tangent_convention=self.tangent_convention,
                state_schema=MaterialStateSchema("other"),
            )

    with pytest.raises(ValueError, match="changed the declared state schema"):
        validated_material_update(DriftingMaterial(), point)
