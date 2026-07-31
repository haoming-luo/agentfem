from __future__ import annotations

import numpy as np
import pytest

from agentfem.constitutive.user_material import (
    AbaqusUserMaterialBridge,
    MaterialPointInput,
    MaterialPointOutput,
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
