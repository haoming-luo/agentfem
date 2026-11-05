"""Independent transformation rules, declared history frames and negative controls."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from agentfem import constitutive as c

spec = importlib.util.spec_from_file_location(
    "rotation_history_example",
    Path(__file__).parents[1] / "examples/history_material/material.py",
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def rotation(angle, axis=2):
    axes = [index for index in range(3) if index != axis]
    Q = np.eye(3)
    Q[np.ix_(axes, axes)] = (
        (np.cos(angle), -np.sin(angle)),
        (np.sin(angle), np.cos(angle)),
    )
    return Q


def request(material, F=None, **kwargs):
    return c.MaterialPointInput(
        np.eye(3),
        np.diag((1.12, 0.96, 1.03)) if F is None else F,
        0.1,
        0.1,
        [],
        material.state_schema.initial_state(),
        material.state_schema,
        **kwargs,
    )


@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_documented_local_verification_snippet(mechanism, monkeypatch):
    import sys

    readme = Path(__file__).parents[1] / "examples/history_material/README.md"
    code = readme.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    code = code.replace('mechanism="viscoelastic"', f'mechanism="{mechanism}"')
    monkeypatch.setitem(sys.modules, "material", example)
    namespace = {}
    exec(compile(code, str(readme), "exec"), namespace)
    assert namespace["derivative"].accepted
    assert namespace["rotated"].accepted


@pytest.mark.parametrize("law", ["viscoelastic", "damage"])
@pytest.mark.parametrize("scale", [1e-200, 1, 1e200])
def test_reference_history_during_loading_unloading_hold_and_changing_observer(
    law, scale
):
    material = example.HistoryMaterial(
        mechanism=law, lame=100 * scale, shear=50 * scale, branch=30 * scale
    )
    point = request(material)
    frames = {v.name: "invariant" for v in material.state_schema.variables}
    for i, stretch in enumerate((1.1, 1.2, 1.2, 1.05, 0.97, 1.08)):
        point = replace(
            point,
            deformation_gradient_new=np.array(
                ((stretch, 0.04, 0), (0.01, 0.98, 0), (0, 0, 1.01))
            ),
            time=(i + 1) * 0.1,
        )
        before = point.state_old.copy()
        check = c.check_material_rotation(
            material,
            point,
            rotation(0.4 * (i + 1)),
            old_rotation=rotation(0.3 * i, 0),
            state_frames=frames,
        )
        assert check.accepted, check.summary()
        assert max(dict(check.errors).values()) < 1e-12
        assert json.loads(json.dumps(check.summary()))["energy_assessed"] is False
        np.testing.assert_array_equal(point.state_old, before)
        response = material.update(point)
        point = replace(
            point,
            deformation_gradient_old=point.deformation_gradient_new,
            state_old=response.state_new,
        )


@pytest.mark.parametrize("law", ["viscoelastic", "damage"])
def test_near_zero_responses_use_explicit_physical_scales(law):
    material = example.HistoryMaterial(mechanism=law)
    check = c.check_material_rotation(
        material,
        request(material, np.eye(3)),
        rotation(1.1),
        old_rotation=rotation(0.4, 0),
        state_frames={v.name: "invariant" for v in material.state_schema.variables},
        stress_scale=100,
        state_scales={v.name: 1 for v in material.state_schema.variables},
    )
    assert check.accepted
    assert check.summary()["stress_scale"] == 100


class SpatialHistory:
    name = "objective nominal-linear law with convected history"
    tangent_convention = c.MaterialTangentConvention.first_piola_deformation_gradient()
    state_schema = c.MaterialStateSchema(
        "transport",
        (
            c.MaterialStateVariable(
                "tensor", shape=(3, 3), initial_value=np.diag((1.0, 2.0, 3.0))
            ),
            c.MaterialStateVariable(
                "vector", shape=(3,), initial_value=(1.0, 2.0, -1.0)
            ),
            c.MaterialStateVariable("age"),
        ),
    )

    def update(self, point):
        F = point.deformation_gradient_new
        H = F @ np.linalg.inv(point.deformation_gradient_old)
        old = self.state_schema.unpack(point.state_old)
        state = np.r_[
            (H @ old["tensor"] @ H.T).ravel(),
            H @ old["vector"],
            old["age"] + point.time_increment,
        ]
        return c.MaterialPointOutput(
            F @ F.T / np.linalg.det(F),
            np.eye(9),
            state,
            state_schema=self.state_schema,
            tangent_convention=self.tangent_convention,
        )


FRAMES = {"tensor": "spatial_tensor", "vector": "spatial_vector", "age": "invariant"}


def test_spatial_history_uses_both_endpoint_frames():
    material = SpatialHistory()
    point = request(material)
    check = c.check_material_rotation(
        material,
        point,
        rotation(0.7),
        old_rotation=rotation(-0.8, 0),
        state_frames=FRAMES,
    )
    assert check.accepted
    # Merely treating every 3x3 history as reference data is not equivalent.
    wrong = c.check_material_rotation(
        material,
        point,
        rotation(0.7),
        old_rotation=rotation(-0.8, 0),
        state_frames={name: "invariant" for name in FRAMES},
    )
    assert not wrong.accepted
    assert dict(wrong.errors)["state:tensor"] > 0.1


@pytest.mark.parametrize("broken", ["stress", "tangent", "history"])
def test_nonobjective_response_is_rejected(broken):
    class Broken(SpatialHistory):
        def update(self, point):
            response = super().update(point)
            if broken == "stress":
                return replace(
                    response,
                    cauchy_stress=response.cauchy_stress + np.diag((0.1, 0, 0)),
                )
            if broken == "tangent":
                A = response.consistent_tangent.copy()
                A[0, 0] += 0.1
                return replace(response, consistent_tangent=A)
            state = response.state_new.copy()
            state[-1] += point.deformation_gradient_new[0, 0]
            return replace(response, state_new=state)

    material = Broken()
    check = c.check_material_rotation(
        material, request(material), rotation(0.7), state_frames=FRAMES
    )
    assert not check.accepted


def test_covariance_is_not_a_derivative_check():
    class WrongDerivative(SpatialHistory):
        def update(self, point):
            response = super().update(point)
            return replace(
                response, consistent_tangent=1.2 * response.consistent_tangent
            )

    material = WrongDerivative()
    point = request(material)
    assert c.check_material_rotation(
        material, point, rotation(0.8), state_frames=FRAMES
    ).accepted
    assert not c.check_material_tangent(material, point).accepted


def test_declared_component_permutation_and_reference_anisotropy():
    class Anisotropic:
        name = "reference anisotropy P=F D"
        state_schema = c.MaterialStateSchema("empty")
        order = (8, 0, 3, 2, 7, 5, 4, 1, 6)
        tangent_convention = replace(
            c.MaterialTangentConvention.first_piola_deformation_gradient(),
            component_order=tuple(
                c.MaterialTangentConvention.first_piola_deformation_gradient().component_order[
                    i
                ]
                for i in order
            ),
        )

        def update(self, point):
            F = point.deformation_gradient_new
            D = np.diag((2.0, 3.0, 5.0))
            A = np.einsum("ik,jl->ijkl", np.eye(3), D).reshape(9, 9)
            return c.MaterialPointOutput(
                F @ D @ F.T / np.linalg.det(F),
                A[np.ix_(self.order, self.order)],
                [],
                state_schema=self.state_schema,
                tangent_convention=self.tangent_convention,
            )

    material = Anisotropic()
    assert c.check_material_rotation(
        material, request(material), rotation(0.6), state_frames={}
    ).accepted
    assert c.check_material_tangent(material, request(material)).accepted


@pytest.mark.parametrize(
    "bad",
    [
        np.diag((-1.0, 1.0, 1.0)),
        np.eye(3) * 2,
        np.eye(2),
        np.full((3, 3), np.nan),
        np.eye(3).astype(complex),
    ],
)
def test_invalid_rotations_rejected_before_material_call(bad):
    material = SpatialHistory()
    point = request(material)
    material.update = lambda _: pytest.fail("Must reject before update")
    with pytest.raises(ValueError, match="proper orthogonal"):
        c.check_material_rotation(material, point, bad, state_frames=FRAMES)


@pytest.mark.parametrize(
    "frames",
    [
        {},
        {**FRAMES, "other": "invariant"},
        {**FRAMES, "age": "spatial_tensor"},
        {**FRAMES, "vector": "unknown"},
    ],
)
def test_history_frames_are_not_inferred(frames):
    material = SpatialHistory()
    point = request(material)
    material.update = lambda _: pytest.fail("Must reject before update")
    with pytest.raises(ValueError, match="state_frames|frame/shape"):
        c.check_material_rotation(material, point, rotation(0.2), state_frames=frames)


def test_external_fields_need_their_own_frame_contract():
    material = SpatialHistory()
    point = request(material, field_variables=[1.0, 0.0, 0.0])
    with pytest.raises(NotImplementedError, match="external-field frames"):
        c.check_material_rotation(material, point, rotation(0.4), state_frames=FRAMES)


def test_mutating_provider_does_not_modify_input_or_create_evidence():
    class Mutating(SpatialHistory):
        def update(self, point):
            point.state_old[:] = 0
            return super().update(point)

    material = Mutating()
    point = request(material)
    old = point.state_old.copy()
    with pytest.raises(ValueError, match="AFM-MATERIAL-INPUT-MUTATION"):
        c.check_material_rotation(material, point, rotation(0.4), state_frames=FRAMES)
    np.testing.assert_array_equal(point.state_old, old)


@pytest.mark.parametrize(
    "options",
    [
        {"tolerance": True},
        {"stress_scale": 0},
        {"state_scales": {"age": float("inf")}},
        {"state_scales": {"other": 1}},
    ],
)
def test_invalid_scales_fail_before_updates(options):
    material = SpatialHistory()
    point = request(material)
    material.update = lambda _: pytest.fail("Must reject before update")
    with pytest.raises(ValueError):
        c.check_material_rotation(
            material, point, rotation(0.4), state_frames=FRAMES, **options
        )


@pytest.mark.parametrize("tangent", ["analytic_spectral", "central_difference"])
def test_existing_j2_material_loaded_and_reversed_history(tangent):
    material = c.finite_strain_j2_logarithmic(
        young=190000,
        poisson=0.3,
        yield_stress=280,
        hardening_modulus=1100,
        tangent_evaluation=tangent,
    )
    point = request(material)
    frames = {v.name: "invariant" for v in material.state_schema.variables}
    for index, stretch in enumerate((1.01, 1.03, 1.015, 0.99, 1.02)):
        F = np.diag((stretch, stretch**-0.5, stretch**-0.5))
        F[0, 1] = 0.01 * index
        point = replace(point, deformation_gradient_new=F, time=0.1 * (index + 1))
        check = c.check_material_rotation(
            material,
            point,
            rotation(0.4 * (index + 1)),
            old_rotation=rotation(0.2 * index, 0),
            state_frames=frames,
            tolerance=1e-6,
        )
        assert check.accepted, check.summary()
        assert c.check_material_tangent(material, point, tolerance=1e-5).accepted
        response = material.update(point)
        point = replace(point, deformation_gradient_old=F, state_old=response.state_new)
    assert (
        material.state_schema.unpack(point.state_old)["equivalent_plastic_strain"] > 0
    )


def test_large_age_cannot_hide_a_small_spatial_history_error():
    class BrokenTinyVector(SpatialHistory):
        def update(self, point):
            response = super().update(point)
            state = response.state_new.copy()
            state[9] += 1e-201  # Fixed spatial direction is deliberately nonobjective.
            return replace(response, state_new=state)

    material = BrokenTinyVector()
    point = request(material)
    old = point.state_old.copy()
    old[9:12] *= 1e-200
    old[-1] = 1e200
    point = replace(point, state_old=old)
    check = c.check_material_rotation(
        material, point, rotation(0.9), state_frames=FRAMES
    )
    assert not check.accepted
    assert dict(check.errors)["state:vector"] > 0.01
    assert dict(check.errors)["state:age"] == 0


@pytest.mark.parametrize("parameter", ["stress_scale", "tolerance"])
@pytest.mark.parametrize("value", ["1", np.array([1.0]), complex(1), float("nan"), -1])
def test_numeric_policy_is_explicitly_scalar_and_real(parameter, value):
    material = SpatialHistory()
    point = request(material)
    with pytest.raises(ValueError, match="finite and positive"):
        c.check_material_rotation(
            material, point, rotation(0.4), state_frames=FRAMES, **{parameter: value}
        )


def test_identity_observer_and_zero_width_external_fields():
    material = SpatialHistory()
    check = c.check_material_rotation(
        material, request(material, field_variables=[]), np.eye(3), state_frames=FRAMES
    )
    assert check.accepted
    assert all(error == 0 for _, error in check.errors)


def test_reference_rotation_does_not_mutate_temperature_or_point_arrays():
    material = example.HistoryMaterial()
    point = request(material, temperature=300.0, temperature_increment=5.0)
    original = {
        key: getattr(point, key).copy()
        for key in (
            "deformation_gradient_old",
            "deformation_gradient_new",
            "state_old",
            "properties",
        )
    }
    check = c.check_material_rotation(
        material,
        point,
        rotation(0.7),
        state_frames={"Q": "invariant", "AGE": "invariant"},
    )
    assert check.accepted
    assert point.temperature == 300 and point.temperature_increment == 5
    for name, value in original.items():
        np.testing.assert_array_equal(getattr(point, name), value)
