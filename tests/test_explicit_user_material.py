# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from agentfem import constitutive
from agentfem.constitutive.user_material import MaterialStateSchema, MaterialStateVariable
from agentfem.constitutive.material_array_batch import MaterialPointArrayBatchInput


def fixture():
    schema = MaterialStateSchema("arbitrary", (MaterialStateVariable("a", shape=(2,)),
                                               MaterialStateVariable("b", shape=(3,))))
    f = np.tile(np.eye(3), (4, 1, 1))
    request = MaterialPointArrayBatchInput(f, f, np.zeros((4, 5)), schema, 0., .1)
    def update(r):
        return constitutive.ExplicitMaterialBatchOutput(np.zeros((4, 3, 3)),
            r.state_old + 1, schema, np.zeros(4))
    return schema, request, update


def test_arbitrary_history_layout_and_detached_identity():
    schema, request, update = fixture()
    identity = {"revision": "v1", "parameters": [1., 2.]}
    material = constitutive.explicit_user_material(state_schema=schema, update=update,
                                                   identity=identity)
    identity["parameters"][0] = 99
    material.identity["parameters"][0] = 101
    assert material.summary()["identity"]["parameters"] == [1., 2.]
    response = material.evaluate(request)
    np.testing.assert_array_equal(response.state_new, np.ones((4, 5)))
    np.testing.assert_array_equal(request.state_old, np.zeros((4, 5)))
    assert not response.state_new.flags.writeable
    assert not material.summary()["tangent_available"]


def test_limit_before_callback():
    schema, request, _ = fixture()
    called = []
    material = constitutive.explicit_user_material(state_schema=schema,
        update=lambda r: called.append(r), identity={"revision":"v1"},
        increment_limit=lambda r: constitutive.ExplicitMaterialIncrementLimit(
            np.full(r.point_count, .01), "test-limit"), increment_policy_identity="test-limit")
    with pytest.raises(constitutive.ExplicitMaterialIncrementExceeded):
        material.evaluate(request)
    assert not called


def test_reject_invalid_response_and_incomplete_policy():
    schema, request, update = fixture()
    with pytest.raises(ValueError, match="together"):
        constitutive.explicit_user_material(state_schema=schema, update=update,
            identity={"revision":"v1"}, increment_policy_identity="missing")
    material = constitutive.explicit_user_material(state_schema=schema,
        update=lambda r: None, identity={"revision":"v1"})
    with pytest.raises(TypeError, match="ExplicitMaterialBatchOutput"):
        material.evaluate(request)
