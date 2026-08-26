from __future__ import annotations

import pytest

from agentfem import learning, models, public_api, results, studies, surrogates


def _displacement_field():
    return learning.FieldEncoding(
        name="displacement",
        role="output",
        unit="m",
        representation="point_samples",
        mesh_policy="mesh_independent_coordinates",
        components=("U1", "U2"),
    )


def test_learning_is_an_additive_public_umbrella():
    assert "learning" in public_api("advanced")
    assert learning.RidgeSurrogate is surrogates.RidgeSurrogate
    assert learning.PINNSpec is surrogates.PINNSpec


def test_neural_field_energy_keeps_physical_sign_separate_from_loss_weight():
    field = _displacement_field()
    strain_energy = learning.ObjectiveTerm(
        name="strain_energy",
        kind="energy",
        expression="integral(psi(F), domain)",
        dependent_fields=(field.name,),
        form="variational",
        measure="domain",
        unit="J",
    )
    external_work = learning.ObjectiveTerm(
        name="external_work",
        kind="energy",
        expression="integral(t dot u, boundary)",
        dependent_fields=(field.name,),
        form="variational",
        measure="boundary",
        unit="J",
        coefficient=-1.0,
        weight=0.25,
    )
    spec = learning.NeuralFieldSpec(
        fields=(field,),
        objectives=(strain_energy, external_work),
        conditions=(
            learning.ConditionSpec(
                name="fixed_left",
                kind="boundary",
                target=field.name,
                on="regions.left",
                value=(0.0, 0.0),
                enforcement="hard",
            ),
        ),
        representations=(
            learning.NeuralRepresentation(
                name="displacement_network",
                fields=(field.name,),
                architecture="xdem:kan",
                features=("coordinates", "xdem:crack_function"),
                enrichments=("xdem:williams_tip",),
            ),
        ),
        sampling=(
            learning.SamplingPlan(
                name="energy_quadrature",
                on="domain",
                strategy="quadrature",
            ),
        ),
    )

    summary = spec.summary()
    assert summary["objective_kinds"] == ("energy",)
    assert summary["objectives"][1]["coefficient"] == -1.0
    assert summary["objectives"][1]["weight"] == 0.25
    assert summary["representations"][0]["enrichments"] == (
        "xdem:williams_tip",
    )
    assert summary["status"] == "declarative"


def test_neural_field_rejects_unknown_field_references():
    with pytest.raises(ValueError, match="unknown fields"):
        learning.NeuralFieldSpec(
            fields=(_displacement_field(),),
            objectives=(
                learning.ObjectiveTerm(
                    name="bad",
                    kind="residual",
                    expression="div(sigma) = 0",
                    dependent_fields=("temperature",),
                    form="strong",
                    measure="domain",
                ),
            ),
            conditions=(
                learning.ConditionSpec(
                    name="fixed_left",
                    kind="boundary",
                    target="displacement",
                    on="regions.left",
                ),
            ),
            representations=(
                learning.NeuralRepresentation(
                    name="displacement_network",
                    fields=("displacement",),
                ),
            ),
        )


def test_sampling_plan_accepts_namespaced_provider_strategy():
    plan = learning.SamplingPlan(
        name="tip_points",
        on="domain",
        strategy="xdem:crack_tip_adaptive",
        count=512,
        seed=2026,
    )

    assert plan.summary()["strategy"] == "xdem:crack_tip_adaptive"


def test_integration_plan_requires_explicitly_held_out_validation_points():
    training = learning.IntegrationRule(
        name="train_points",
        role="training",
        strategy="xdem:tensor_midpoint",
        count=256,
    )
    with pytest.raises(ValueError, match="declare independence"):
        learning.IntegrationPlan(
            training=training,
            validation=learning.IntegrationRule(
                name="validation_points",
                role="validation",
                strategy="xdem:tensor_midpoint",
                count=512,
            ),
        )


def test_integration_consistency_distinguishes_loss_from_held_out_evidence():
    plan = learning.IntegrationPlan(
        training=learning.IntegrationRule(
            name="train_points",
            role="training",
            strategy="xdem:tensor_midpoint",
            count=256,
        ),
        validation=learning.IntegrationRule(
            name="validation_points",
            role="validation",
            strategy="xdem:tensor_midpoint",
            count=512,
            independent_of=("train_points",),
        ),
        refinements=(
            learning.IntegrationRule(
                name="refined_points",
                role="refinement",
                strategy="xdem:tensor_midpoint",
                count=1024,
                independent_of=("train_points", "validation_points"),
            ),
        ),
    )
    accepted = learning.integration_consistency_check(
        plan,
        training_value=10.0,
        validation_value=10.1,
        refinement_values=(10.12,),
        relative_tolerance=0.03,
    )
    suspicious = learning.integration_consistency_check(
        plan,
        training_value=6.0,
        validation_value=10.0,
        refinement_values=(10.1,),
        relative_tolerance=0.03,
    )

    assert accepted.status == "accepted"
    assert suspicious.status == "uncertain"
    assert "possible_training_quadrature_exploitation" in suspicious.findings
    assert len(plan.fingerprint) == 64


def test_condition_values_are_machine_shaped_not_live_callables():
    with pytest.raises(TypeError, match="JSON-shaped"):
        learning.ConditionSpec(
            name="unsafe",
            kind="boundary",
            target="displacement",
            on="regions.left",
            value=lambda _x: 0.0,
        )


def test_legacy_pinn_contract_lifts_to_neural_field_contract():
    field = learning.FieldEncoding(
        name="temperature",
        role="output",
        unit="K",
        representation="point_samples",
        mesh_policy="mesh_independent_coordinates",
    )
    legacy = learning.PINNSpec(
        fields=(field,),
        residuals=(
            learning.PhysicsResidual(
                name="heat_balance",
                equation="rho*c*dT_dt - div(k*grad(T)) - Q = 0",
                form="strong",
                dependent_fields=(field.name,),
                independent_variables=("x", "y", "t"),
            ),
        ),
        conditions=(
            learning.PhysicsCondition(
                name="initial_temperature",
                kind="initial",
                target=field.name,
                location="t=0",
                value=293.15,
            ),
        ),
        collocation_policy={"strategy": "sobol"},
    )

    general = learning.NeuralFieldSpec.from_pinn(legacy)

    assert general.objective_kinds == ("residual",)
    assert general.conditions[0].enforcement == "penalty"
    assert general.metadata["legacy_collocation_policy"] == {"strategy": "sobol"}


def test_trainable_parameters_require_an_inverse_or_hybrid_purpose():
    field = _displacement_field()
    parameter = learning.TrainableParameter(
        name="young_modulus",
        role="material",
        initial=200.0e9,
        bounds=(100.0e9, 300.0e9),
        unit="Pa",
        transform="log",
    )
    objective = learning.ObjectiveTerm(
        name="equilibrium",
        kind="residual",
        expression="div(sigma(u, E)) = 0",
        dependent_fields=(field.name,),
        form="strong",
        measure="domain",
    )
    condition = learning.ConditionSpec(
        name="measured_displacement",
        kind="observation",
        target=field.name,
        on="observations.extensometer",
        enforcement="data",
    )

    with pytest.raises(ValueError, match="purpose='inverse'"):
        learning.NeuralFieldSpec(
            fields=(field,),
            objectives=(objective,),
            conditions=(condition,),
            representations=(
                learning.NeuralRepresentation(
                    name="displacement_network",
                    fields=(field.name,),
                ),
            ),
            parameters=(parameter,),
        )

    inverse = learning.NeuralFieldSpec(
        fields=(field,),
        objectives=(objective,),
        conditions=(condition,),
        representations=(
            learning.NeuralRepresentation(
                name="displacement_network",
                fields=(field.name,),
            ),
        ),
        parameters=(parameter,),
        purpose="inverse",
    )
    assert inverse.parameters[0].unit == "Pa"


def test_parameter_transform_validates_its_physical_domain():
    with pytest.raises(ValueError, match="positive bounds"):
        learning.TrainableParameter(
            name="conductivity",
            role="material",
            initial=1.0,
            bounds=(-1.0, 2.0),
            transform="log",
        )
    with pytest.raises(ValueError, match="requires finite bounds"):
        learning.TrainableParameter(
            name="damage",
            role="state",
            initial=0.2,
            transform="logit",
        )


def test_neural_representations_cover_each_field_once():
    field = _displacement_field()
    objective = learning.ObjectiveTerm(
        name="equilibrium",
        kind="residual",
        expression="div(sigma) = 0",
        dependent_fields=(field.name,),
        form="strong",
        measure="domain",
    )
    condition = learning.ConditionSpec(
        name="fixed_left",
        kind="boundary",
        target=field.name,
        on="regions.left",
    )
    representation = learning.NeuralRepresentation(
        name="first",
        fields=(field.name,),
    )

    with pytest.raises(ValueError, match="exactly once"):
        learning.NeuralFieldSpec(
            fields=(field,),
            objectives=(objective,),
            conditions=(condition,),
            representations=(
                representation,
                learning.NeuralRepresentation(
                    name="second",
                    fields=(field.name,),
                ),
            ),
        )


def _minimal_neural_field_spec():
    field = _displacement_field()
    return learning.NeuralFieldSpec(
        fields=(field,),
        objectives=(
            learning.ObjectiveTerm(
                name="equilibrium",
                kind="residual",
                expression="div(sigma) = 0",
                dependent_fields=(field.name,),
                form="strong",
                measure="domain",
            ),
        ),
        conditions=(
            learning.ConditionSpec(
                name="fixed_left",
                kind="boundary",
                target=field.name,
                on="regions.left",
            ),
        ),
        representations=(
            learning.NeuralRepresentation(
                name="user_network",
                fields=(field.name,),
            ),
        ),
    )


def test_user_owned_neural_field_executor_enters_common_step_lifecycle(tmp_path):
    captured = []

    class UserModel:
        executor_name = "laboratory.custom_neural_field"
        executor_version = "2026.08"

        def solve(self, request):
            captured.append(request)
            result = results.SimulationResult(request.name)
            result.add_quantity("loss", request.option("loss", 0.0), kind="optimization")
            return result

    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain"),
        name="byom_model",
    )
    step = model.step(
        target=_minimal_neural_field_spec(),
        executor=UserModel(),
        executor_options={"loss": 1.0e-4},
        output=tmp_path / "run",
        name="user_neural_field",
    )
    result = step.solve_result()

    assert result.quantity("loss") == pytest.approx(1.0e-4)
    assert captured[0].model is model
    assert captured[0].options == {"loss": 1.0e-4}
    assert captured[0].summary()["managed_output"] is True
    assert str(tmp_path) not in str(captured[0].summary())
    assert result.metadata["learning_execution"]["executor"] == {
        "name": "laboratory.custom_neural_field",
        "version": "2026.08",
    }
    assert (tmp_path / "run" / "result.json").is_file()
    assert step.solve_result() is result


def test_neural_field_executor_must_return_simulation_result():
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain")
    )
    step = model.step(
        target=_minimal_neural_field_spec(),
        executor=lambda request: {"loss": 0.0},
    )

    with pytest.raises(TypeError, match="must return results.SimulationResult"):
        step.solve_result()


def test_neural_field_executor_options_are_explicit_and_immutable():
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_strain")
    )

    with pytest.raises(TypeError, match="executor_options must be a mapping"):
        model.step(
            target=_minimal_neural_field_spec(),
            executor=lambda request: results.SimulationResult(request.name),
            executor_options=("epochs", 10),
        )
