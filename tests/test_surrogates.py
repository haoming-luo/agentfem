from __future__ import annotations

import numpy as np
import pytest

from agentfem import campaigns, datasets, surrogates


def _linear_dataset(count=30):
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("x", -2.0, 2.0),
        campaigns.RealParameter("z", 0.0, 1.0),
    )
    samples = []
    for index in range(count):
        x = -2.0 + 4.0 * index / (count - 1)
        z = ((index * 7) % count) / (count - 1)
        samples.append(
            datasets.Sample(
                case_id=f"case-{index}",
                inputs={"x": x, "z": z},
                outputs={
                    "qoi": 3.0 * x - 2.0 * z + 1.0,
                    "field": np.asarray((x + z, 2.0 * x, -z)),
                },
            )
        )
    return datasets.ScientificDataset(
        parameter_space=space,
        quantities=(
            datasets.Quantity("qoi", unit="m"),
            datasets.Quantity(
                "field",
                shape=(3,),
                kind="sampled_field",
                unit="Pa",
                field_encoding={"representation": "fixed_vector_samples"},
            ),
        ),
        samples=tuple(samples),
        name="linear_reference",
    )


def test_ridge_fit_validate_predict_and_artifact_round_trip(tmp_path):
    dataset = _linear_dataset()
    trained = surrogates.RidgeSurrogate(alpha=1.0e-12).fit(dataset)
    report = trained.validate(
        dataset,
        thresholds={"max_relative_l2": 1.0e-8, "min_r2": 0.999999},
    )
    prediction = trained.predict_with_uncertainty({"x": 0.2, "z": 0.4})
    trained.write(tmp_path / "ridge")
    restored = surrogates.TrainedRidge.read(tmp_path / "ridge")

    assert report.accepted is True
    assert prediction.source == "ridge"
    assert prediction.uncertainty is not None
    assert trained.coefficients.flags.writeable is False
    np.testing.assert_allclose(
        restored.predict_matrix({"x": 0.2, "z": 0.4}),
        trained.predict_matrix({"x": 0.2, "z": 0.4}),
    )


def test_training_workflow_keeps_split_validation_and_guard_together():
    dataset = _linear_dataset()
    training = surrogates.train(
        dataset,
        estimator=surrogates.RidgeSurrogate(alpha=1.0e-12),
        validation_fraction=0.2,
        seed=2026,
        thresholds={"max_relative_l2": 1.0e-8},
    )
    guarded = training.guard()

    assert training.accepted is True
    assert len(training.split.train.samples) == 24
    assert len(training.split.validation.samples) == 6
    assert training.summary()["split"]["seed"] == 2026
    assert guarded.predict({"x": 0.0, "z": 0.5}).in_domain is True


def test_training_workflow_rejects_too_little_independent_evidence():
    dataset = _linear_dataset(count=2)

    with pytest.raises(ValueError, match="at least three"):
        surrogates.train(dataset)


def test_pod_ridge_reconstructs_field_and_persists(tmp_path):
    dataset = _linear_dataset()
    trained = surrogates.PODRidgeSurrogate(
        max_modes=4,
        energy=1.0,
        alpha=1.0e-12,
    ).fit(dataset)
    report = trained.validate(dataset, thresholds={"max_relative_l2": 1.0e-8})
    trained.write(tmp_path / "pod")
    restored = surrogates.TrainedPODRidge.read(tmp_path / "pod")

    assert report.accepted is True
    assert trained.mode_count <= 4
    assert trained.basis.flags.writeable is False
    np.testing.assert_allclose(
        restored.predict_matrix({"x": -0.5, "z": 0.25}),
        trained.predict_matrix({"x": -0.5, "z": 0.25}),
    )


def test_applicability_guard_uses_high_fidelity_fallback():
    dataset = _linear_dataset()
    interior = dataset.subset(range(5, 25))
    trained = surrogates.RidgeSurrogate().fit(interior)
    domain = surrogates.BoxApplicabilityDomain.from_dataset(interior)
    calls = {"count": 0}

    def fallback(values):
        calls["count"] += 1
        return {
            "qoi": 3.0 * values["x"] - 2.0 * values["z"] + 1.0,
            "field": [values["x"] + values["z"], 2.0 * values["x"], -values["z"]],
        }

    guarded = surrogates.GuardedSurrogate(trained, domain, fallback=fallback)
    inside = guarded.predict(interior.samples[5].inputs)
    outside = guarded.predict({"x": -2.0, "z": 0.0})

    assert inside.in_domain is True
    assert outside.in_domain is False
    assert outside.source == "high_fidelity_fallback"
    assert calls["count"] == 1


def test_guard_rejects_extrapolation_without_fallback():
    dataset = _linear_dataset()
    interior = dataset.subset(range(5, 25))
    guarded = surrogates.GuardedSurrogate(
        surrogates.RidgeSurrogate().fit(interior),
        surrogates.BoxApplicabilityDomain.from_dataset(interior),
    )

    with pytest.raises(surrogates.OutOfDomainError):
        guarded.predict({"x": -2.0, "z": 0.0})


def test_guard_validates_high_fidelity_fallback_schema():
    dataset = _linear_dataset()
    interior = dataset.subset(range(5, 25))
    guarded = surrogates.GuardedSurrogate(
        surrogates.RidgeSurrogate().fit(interior),
        surrogates.BoxApplicabilityDomain.from_dataset(interior),
        fallback=lambda _values: {"qoi": 1.0},
    )

    with pytest.raises(ValueError, match="fallback outputs"):
        guarded.predict({"x": -2.0, "z": 0.0})


def test_applicability_domain_rejects_an_unseen_category():
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("x", 0.0, 1.0),
        campaigns.ChoiceParameter("material_family", ("steel", "aluminum")),
    )
    dataset = datasets.ScientificDataset(
        parameter_space=space,
        quantities=(datasets.Quantity("response"),),
        samples=tuple(
            datasets.Sample(
                case_id=str(index),
                inputs={"x": index / 4.0, "material_family": "steel"},
                outputs={"response": float(index)},
            )
            for index in range(5)
        ),
    )
    domain = surrogates.BoxApplicabilityDomain.from_dataset(dataset)

    assert domain.contains({"x": 0.5, "material_family": "steel"}) is True
    assert domain.contains({"x": 0.5, "material_family": "aluminum"}) is False


def test_physics_learning_contracts_are_explicit():
    temperature = surrogates.FieldEncoding(
        name="temperature",
        role="output",
        unit="K",
        representation="point_samples",
        mesh_policy="mesh_independent_coordinates",
    )
    residual = surrogates.PhysicsResidual(
        name="heat_balance",
        equation="rho*c*dT_dt - div(k*grad(T)) - Q = 0",
        form="strong",
        dependent_fields=("temperature",),
        independent_variables=("x", "y", "t"),
        unit="W/m^3",
        implementation="agentfem.physics.heat_strong_v1",
    )
    condition = surrogates.PhysicsCondition(
        name="initial_temperature",
        kind="initial",
        target="temperature",
        location="t=0",
        value=293.15,
    )
    spec = surrogates.PINNSpec(
        fields=(temperature,),
        residuals=(residual,),
        conditions=(condition,),
    )

    assert spec.summary()["status"] == "contract_only"
    assert spec.summary()["residuals"][0]["form"] == "strong"


def test_fno_contract_rejects_unstructured_encoding():
    field = surrogates.FieldEncoding(
        name="material",
        role="input",
        unit="Pa",
        representation="mesh_dofs",
    )
    output = surrogates.FieldEncoding(
        name="displacement",
        role="output",
        unit="m",
        representation="mesh_dofs",
    )

    with pytest.raises(ValueError, match="structured_grid"):
        surrogates.NeuralOperatorSpec(
            architecture="fno",
            inputs=(field,),
            outputs=(output,),
            boundary_encoding="mask_and_values",
        )


def test_torch_pinn_adapter_executes_explicit_autodiff_contract():
    torch = pytest.importorskip("torch")
    field = surrogates.FieldEncoding(
        name="u",
        role="output",
        unit=None,
        representation="point_samples",
        mesh_policy="mesh_independent_coordinates",
    )
    spec = surrogates.PINNSpec(
        fields=(field,),
        residuals=(
            surrogates.PhysicsResidual(
                name="unit_gradient",
                equation="du_dx - 1 = 0",
                form="strong",
                dependent_fields=("u",),
                independent_variables=("x",),
            ),
        ),
        conditions=(
            surrogates.PhysicsCondition(
                name="origin",
                kind="boundary",
                target="u",
                location="x=0",
                value=0.0,
            ),
        ),
    )

    class Exact(torch.nn.Module):
        def forward(self, x):
            return x

    def residual(module, points):
        values = module(points)
        gradient = torch.autograd.grad(
            values,
            points,
            grad_outputs=torch.ones_like(values),
            create_graph=True,
        )[0]
        return gradient - 1.0

    adapter = surrogates.TorchPINNAdapter(
        spec,
        residual_functions={"unit_gradient": residual},
        condition_functions={"origin": lambda module, points: module(points)},
    )
    collocation = torch.linspace(0.0, 1.0, 8).reshape(-1, 1).requires_grad_(True)
    boundary = torch.zeros((1, 1))
    loss, diagnostics = adapter.loss(
        Exact(),
        residual_points={"unit_gradient": collocation},
        condition_points={"origin": boundary},
    )

    assert float(loss) == pytest.approx(0.0)
    assert diagnostics["total"] == pytest.approx(0.0)
    assert adapter.summary()["automatic_ufl_translation"] is False
