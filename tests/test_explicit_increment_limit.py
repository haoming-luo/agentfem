# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import numpy as np
import pytest

from agentfem.constitutive._explicit_material import (
    ExplicitMaterialIncrementLimit,
    ExplicitMaterialIncrementExceeded,
    explicit_increment_policy_identity,
    validated_explicit_material_update,
)
from test_explicit_material_contract import SCHEMA, request, response


class Limited:
    state_schema = SCHEMA
    explicit_increment_policy_identity = "test-material-resolution.v1"

    def __init__(self, limit=0.1):
        self.limit = limit
        self.calls = 0

    def explicit_increment_limit(self, value):
        return ExplicitMaterialIncrementLimit(
            np.full(value.point_count, self.limit),
            self.explicit_increment_policy_identity,
        )

    def update_explicit_array_batch(self, value):
        self.calls += 1
        return response(value.point_count)


@pytest.mark.parametrize("index", [67, 100, 1000, 10000, 100000])
def test_limit_equality_uses_clock_roundoff_not_only_increment_roundoff(index):
    nominal = 1e-6
    old_time, new_time = (index - 1) * nominal, index * nominal
    provider = Limited(nominal)
    value = replace(request(), time=old_time, time_increment=new_time - old_time)
    validated_explicit_material_update(provider, value)
    assert provider.calls == 1
    with pytest.raises(ExplicitMaterialIncrementExceeded):
        validated_explicit_material_update(
            provider, replace(value, time_increment=nominal * (1 + 1e-8))
        )
    assert provider.calls == 1


def test_unresolvable_absolute_clock_cannot_relax_material_limit():
    provider = Limited(1e-6)
    with pytest.raises(ValueError, match="cannot resolve"):
        validated_explicit_material_update(
            provider, replace(request(), time=1e20, time_increment=2e-6)
        )
    assert provider.calls == 0


def test_limit_consumed_before_response_and_does_not_invent_tangent():
    provider = Limited(0.05)
    with pytest.raises(ExplicitMaterialIncrementExceeded) as failure:
        validated_explicit_material_update(provider, request())
    assert provider.calls == 0
    assert failure.value.code == "explicit_material_increment_exceeded"
    assert failure.value.allowed_time_increment == 0.05
    assert failure.value.point_index == 0
    result = validated_explicit_material_update(
        provider, replace(request(), time_increment=0.05)
    )
    assert provider.calls == 1
    assert (
        result.energy_contract()["explicit_stability"]
        == "requires_independent_procedure_contract"
    )


@pytest.mark.parametrize("values", [[0], [-1], [np.nan], [np.inf], [1j], [[1]], [True]])
def test_invalid_limit_rejected(values):
    with pytest.raises(ValueError):
        ExplicitMaterialIncrementLimit(values, "test")


def test_limits_are_copied_and_readonly_and_empty_batch_is_supported():
    raw = np.array([1.0])
    limit = ExplicitMaterialIncrementLimit(raw, "test")
    raw[0] = 2
    assert limit.maximum_time_increment[0] == 1
    assert not limit.maximum_time_increment.flags.writeable
    assert (
        ExplicitMaterialIncrementLimit(np.zeros(0), "test").maximum_time_increment.size
        == 0
    )


@pytest.mark.parametrize("mode", ["count", "identity", "nan", "type", "changed"])
def test_provider_policy_revalidated(mode):
    provider = Limited()

    def query(value):
        limit = ExplicitMaterialIncrementLimit(
            np.ones(2), provider.explicit_increment_policy_identity
        )
        if mode == "count":
            return replace(limit, maximum_time_increment=np.ones(1))
        if mode == "identity":
            return replace(limit, policy_identity="different")
        if mode == "type":
            return np.ones(2)
        if mode == "changed":
            provider.explicit_increment_policy_identity = "changed"
        else:
            object.__setattr__(limit, "maximum_time_increment", np.full(2, np.nan))
        return limit

    provider.explicit_increment_limit = query
    with pytest.raises((ValueError, TypeError)):
        validated_explicit_material_update(provider, request())
    assert provider.calls == 0


def test_partial_or_mutating_policy_identity_rejected():
    provider = Limited()
    provider.explicit_increment_policy_identity = None
    with pytest.raises(ValueError):
        explicit_increment_policy_identity(provider)
    provider.explicit_increment_policy_identity = "test"

    def update(value):
        provider.explicit_increment_policy_identity = "modified"
        return response()

    provider.update_explicit_array_batch = update
    with pytest.raises(ValueError, match="changed during"):
        validated_explicit_material_update(provider, request())


def test_late_chunk_limit_has_global_point_index_and_preserves_fields():
    from test_explicit_material_quadrature import fixture
    from agentfem.constitutive._explicit_material_driver import (
        update_explicit_material_points,
    )

    material, state, options = fixture()
    material.explicit_increment_policy_identity = "state-dependent.v1"
    options["deformation_gradient_new"][1, 0, 0] = 1.2

    def query(value):
        limits = np.where(value.deformation_gradient_new[:, 0, 0] > 1.1, 0.01, 0.2)
        return ExplicitMaterialIncrementLimit(
            limits, material.explicit_increment_policy_identity
        )

    material.explicit_increment_limit = query
    trial = state.trial_state_vectors().copy()
    committed = state.committed_state_vectors().copy()
    with pytest.raises(ExplicitMaterialIncrementExceeded) as failure:
        update_explicit_material_points(material, state, **options, max_batch_points=1)
    assert failure.value.point_index == 1
    assert material.calls == 1
    np.testing.assert_array_equal(state.trial_state_vectors(), trial)
    np.testing.assert_array_equal(state.committed_state_vectors(), committed)


def test_step_rejection_restores_nodal_and_material_state():
    from test_response_hex_explicit import make_step

    step = make_step()
    body = step.residual
    body.material.explicit_increment_policy_identity = "limit.v1"
    body.material.explicit_increment_limit = lambda request: (
        ExplicitMaterialIncrementLimit(np.full(request.point_count, 0.5e-4), "limit.v1")
    )
    # Reconstruct to admit the policy into checkpoint identity before stepping.
    from agentfem.mechanics._response_hex_explicit import ResponseHexExplicitResidual

    replacement = ResponseHexExplicitResidual(
        body.internal,
        body.material,
        time_increment_ceiling=body.ceiling,
        material_identity={"test": "limited"},
        stress_free_initial_state=True,
        require_material_increment_limit=True,
    )
    step.residual = replacement
    before = step.state.snapshot()
    accepted = replacement.snapshot()
    with pytest.raises(ExplicitMaterialIncrementExceeded):
        step.run()
    assert replacement.snapshot() == accepted
    for name, values in before["fields"].items():
        np.testing.assert_array_equal(step.state.snapshot()["fields"][name], values)
    assert replacement.material.calls == 0


def test_required_policy_and_restart_identity():
    from test_response_hex_explicit import make_step
    from agentfem.mechanics._response_hex_explicit import ResponseHexExplicitResidual

    body = make_step().residual
    options = dict(
        time_increment_ceiling=body.ceiling,
        material_identity={"test": "limit"},
        stress_free_initial_state=True,
        require_material_increment_limit=True,
    )
    with pytest.raises(ValueError, match="requires a material"):
        ResponseHexExplicitResidual(body.internal, body.material, **options)
    body.material.explicit_increment_policy_identity = "v1"
    body.material.explicit_increment_limit = lambda request: (
        ExplicitMaterialIncrementLimit(
            np.full(request.point_count, 0.1),
            body.material.explicit_increment_policy_identity,
        )
    )
    first = ResponseHexExplicitResidual(body.internal, body.material, **options)
    record = first.snapshot()
    body.material.explicit_increment_policy_identity = "v2"
    second = ResponseHexExplicitResidual(body.internal, body.material, **options)
    with pytest.raises(ValueError, match="identity"):
        second.restore(record)
    assert first.identity != second.identity


def make_limited_step(*, failing_time=None):
    from test_response_hex_explicit import make_step
    from agentfem.mechanics._response_hex_explicit import ResponseHexExplicitResidual

    step = make_step()
    old = step.residual
    material = old.material
    material.explicit_increment_policy_identity = f"bounded-test.v1:{failing_time}"

    def limit(request):
        value = 1e-4 if failing_time is None or request.time < failing_time else 5e-5
        return ExplicitMaterialIncrementLimit(
            np.full(request.point_count, value),
            material.explicit_increment_policy_identity,
        )

    material.explicit_increment_limit = limit
    step.residual = ResponseHexExplicitResidual(
        old.internal,
        material,
        time_increment_ceiling=old.ceiling,
        material_identity={"test": "bounded-test"},
        stress_free_initial_state=True,
        require_material_increment_limit=True,
    )
    return step


def test_nonzero_limited_step_and_disk_restart_are_identical(tmp_path):
    full = make_limited_step()
    full.run()
    assert np.max(abs(full.state.u.value.x.array)) > 0
    partial = make_limited_step()
    partial.run(until_step=3)
    checkpoint = partial.save_checkpoint(tmp_path / "material_limit")
    resumed = make_limited_step()
    resumed.load_checkpoint(checkpoint)
    resumed.run()
    assert full.residual.snapshot() == resumed.residual.snapshot()
    assert full.history_records == resumed.history_records
    for name, values in full.state.snapshot()["fields"].items():
        np.testing.assert_array_equal(resumed.state.snapshot()["fields"][name], values)


def test_later_state_dependent_limit_keeps_last_accepted_station():
    step = make_limited_step(failing_time=2e-4)
    step.run(until_step=2)
    state = step.state.snapshot()
    material = step.residual.snapshot()
    with pytest.raises(ExplicitMaterialIncrementExceeded):
        step.run()
    assert step.completed_steps == 2
    assert step.residual.snapshot() == material
    for name, values in state["fields"].items():
        np.testing.assert_array_equal(step.state.snapshot()["fields"][name], values)


def test_roundoff_at_limit_is_not_physical_permission_to_exceed_it():
    provider = Limited(0.1)
    validated_explicit_material_update(
        provider, replace(request(), time_increment=np.nextafter(0.1, 1))
    )
    with pytest.raises(ExplicitMaterialIncrementExceeded):
        validated_explicit_material_update(
            provider, replace(request(), time_increment=0.1 * (1 + 1e-10))
        )
