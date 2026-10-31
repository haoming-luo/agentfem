"""Verification must not certify a provider which edits its trial request."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from agentfem import constitutive as c

spec = importlib.util.spec_from_file_location(
    "verification_history_example",
    Path(__file__).parents[1] / "examples/history_material/material.py",
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)

ARRAYS = (
    "deformation_gradient_old",
    "deformation_gradient_new",
    "state_old",
    "properties",
    "field_variables",
)


def request(material):
    return c.MaterialPointInput(
        np.eye(3),
        np.diag((1.1, 0.98, 1.0)),
        0.1,
        0.1,
        [3.0],
        material.state_schema.initial_state(),
        material.state_schema,
        field_variables=[2.0],
    )


def check(material, point, reduced):
    if reduced:
        return c.check_plane_stress_tangent(
            material,
            point,
            reduction=c.PlaneStressReduction(symmetry="reflection_symmetric"),
        )
    return c.check_material_tangent(material, point)


@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("attribute", ARRAYS)
@pytest.mark.parametrize("at_call", [1, 2])
@pytest.mark.parametrize("provider_failure", [False, True])
def test_verification_rejects_input_mutation_without_touching_caller(
    reduced, attribute, at_call, provider_failure
):
    class MutatingMaterial(example.HistoryMaterial):
        calls = 0

        def update(self, point):
            self.calls += 1
            response = super().update(point)
            if self.calls == at_call:
                getattr(point, attribute).flat[0] += 0.125
                if provider_failure:
                    raise RuntimeError("local material failure")
            return response

    material = MutatingMaterial()
    point = request(material)
    before = {name: getattr(point, name).copy() for name in ARRAYS}
    try:
        with pytest.raises(
            ValueError, match=f"AFM-MATERIAL-INPUT-MUTATION.*{attribute}"
        ):
            check(material, point, reduced)
    finally:
        for name, original in before.items():
            np.testing.assert_array_equal(getattr(point, name), original)
    assert material.calls == at_call  # Contract errors are not local cutbacks.


@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("mechanism", ["viscoelastic", "damage"])
def test_repeated_verification_keeps_identical_old_history(reduced, mechanism):
    material = example.HistoryMaterial(mechanism=mechanism)
    point = request(material)
    first = material.update(point)
    point = replace(
        point,
        state_old=first.state_new,
        deformation_gradient_old=point.deformation_gradient_new,
        deformation_gradient_new=np.diag((1.05, 1.0, 1.0)),
        time=0.2,
    )
    before = {name: getattr(point, name).copy() for name in ARRAYS}
    a, b = check(material, point, reduced), check(material, point, reduced)
    assert a.accepted and b.accepted
    assert a.summary() == b.summary()
    for name, original in before.items():
        np.testing.assert_array_equal(getattr(point, name), original)


class SmallLinear:
    name = "small linear verification reference"
    state_schema = c.MaterialStateSchema(
        "small history", (c.MaterialStateVariable("age"),)
    )
    parameter_schema = c.MaterialParameterSchema("no parameters", ())

    def __init__(self, scale=1.0, correct=True, shear="tensor"):
        self.scale, self.correct = scale, correct
        self.tangent_convention = c.small_strain_tangent_convention(
            shear_convention=shear
        )

    def update(self, point):
        diagonal = [1, 1, 1] + [
            0.5 if self.tangent_convention.shear_convention == "engineering" else 1
        ] * 3
        return c.SmallStrainMaterialPointOutput(
            self.scale * point.strain_new,
            self.scale * np.diag(diagonal) * (1 if self.correct else 1.2),
            point.state_old + point.time_increment,
            self.state_schema,
            self.tangent_convention,
        )


def small_request(material):
    return c.SmallStrainMaterialPointInput(
        np.zeros((3, 3)),
        np.diag((0.01, 0.02, 0.03)),
        0.1,
        0.1,
        {},
        [0.0],
        material.state_schema,
        material.parameter_schema,
    )


@pytest.mark.parametrize("scale", [1e-200, 1e-9, 1, 1e200])
@pytest.mark.parametrize("correct", [False, True])
@pytest.mark.parametrize("shear", ["tensor", "engineering"])
def test_small_strain_tangent_evidence_is_unit_robust(scale, correct, shear):
    material = SmallLinear(scale, correct, shear)
    evidence = c.check_small_strain_material_tangent(material, small_request(material))
    assert evidence.accepted == correct
    assert np.isfinite(evidence.relative_error)
    assert evidence.relative_error == pytest.approx(0 if correct else 0.2, abs=1e-9)


@pytest.mark.parametrize("attribute", ["strain_old", "strain_new", "state_old"])
@pytest.mark.parametrize("at_call", [1, 2])
@pytest.mark.parametrize("provider_failure", [False, True])
def test_small_strain_checker_detects_mutation_and_preserves_caller(
    attribute, at_call, provider_failure
):
    class Mutating(SmallLinear):
        calls = 0

        def update(self, point):
            self.calls += 1
            response = super().update(point)
            if self.calls == at_call:
                getattr(point, attribute).flat[0] += 0.1
                if provider_failure:
                    raise RuntimeError("local material failure")
            return response

    material = Mutating()
    point = small_request(material)
    before = [point.strain_old.copy(), point.strain_new.copy(), point.state_old.copy()]
    with pytest.raises(ValueError, match=f"AFM-MATERIAL-INPUT-MUTATION.*{attribute}"):
        c.check_small_strain_material_tangent(material, point)
    for actual, expected in zip(
        (point.strain_old, point.strain_new, point.state_old), before
    ):
        np.testing.assert_array_equal(actual, expected)
    assert material.calls == at_call


def test_small_strain_checker_rejects_unrepresentable_difference():
    material = SmallLinear()
    with pytest.raises(ValueError, match="not representable"):
        c.check_small_strain_material_tangent(
            material, small_request(material), relative_step=1e-30
        )


def response_with_stress(kind, stress):
    schema = c.MaterialStateSchema("empty")
    if kind == "finite":
        return c.MaterialPointOutput(stress, np.eye(6), [])
    if kind == "finite_batch":
        return c.MaterialPointArrayBatchOutput(
            stress[None],
            np.eye(9)[None],
            np.empty((1, 0)),
            c.MaterialTangentConvention.first_piola_deformation_gradient(),
            schema,
        )
    if kind == "small":
        return c.SmallStrainMaterialPointOutput(stress, np.eye(6), [], schema)
    return c.SmallStrainMaterialPointBatchOutput(
        stress[None],
        np.eye(6)[None],
        np.empty((1, 0)),
        schema,
    )


@pytest.mark.parametrize("kind", ["finite", "finite_batch", "small", "small_batch"])
@pytest.mark.parametrize("scale", [1e-300, 1e-200, 1e-9, 1, 1e200, 1e300, 1e308])
@pytest.mark.parametrize("skew", [0, 1e-12, 0.1])
def test_stress_symmetry_decision_is_unit_robust(kind, scale, skew):
    stress = scale * np.array(((1, 0.2 + skew, 0), (0.2, 0.5, 0), (0, 0, 0.1)))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if skew == 0.1:
            with pytest.raises(ValueError, match="symmetric"):
                response_with_stress(kind, stress)
        else:
            response = response_with_stress(kind, stress)
            assert np.isfinite(response.cauchy_stress).all()
            if skew == 0:
                np.testing.assert_array_equal(
                    response.cauchy_stress.reshape(3, 3), stress
                )


@pytest.mark.parametrize("kind", ["finite", "finite_batch", "small", "small_batch"])
@pytest.mark.parametrize("scale", [0, np.nextafter(0.0, 1.0)])
def test_zero_and_subnormal_symmetric_stress_are_preserved(kind, scale):
    stress = scale * np.eye(3)
    response = response_with_stress(kind, stress)
    np.testing.assert_array_equal(response.cauchy_stress.reshape(3, 3), stress)


def test_symmetry_normalization_preserves_ordinary_scale_tolerance():
    from agentfem.constitutive.user_material import _symmetric_tensors

    rng = np.random.default_rng(918)
    for factor in (0.8, 0.99, 1.01, 1.2):
        for _ in range(50):
            original = rng.normal(size=(3, 3))
            stress = original + original.T
            stress[0, 1] += factor * 1e-10 * np.linalg.norm(stress)
            expected = np.max(np.abs(stress - stress.T)) <= 1e-10 * np.linalg.norm(
                stress
            )
            assert _symmetric_tensors(stress) == expected


@pytest.mark.parametrize("kind", ["finite_batch", "small_batch"])
def test_batch_symmetry_uses_each_points_scale(kind):
    # A large neighbor must not conceal a small point's nonsymmetric stress.
    stress = np.stack((np.eye(3) * 1e200, np.eye(3) * 1e-200))
    stress[1, 0, 1] = 1e-201
    schema = c.MaterialStateSchema("empty")
    with pytest.raises(ValueError, match="symmetric"):
        if kind == "finite_batch":
            c.MaterialPointArrayBatchOutput(
                stress,
                np.tile(np.eye(9), (2, 1, 1)),
                np.empty((2, 0)),
                c.MaterialTangentConvention.first_piola_deformation_gradient(),
                schema,
            )
        else:
            c.SmallStrainMaterialPointBatchOutput(
                stress,
                np.tile(np.eye(6), (2, 1, 1)),
                np.empty((2, 0)),
                schema,
            )
