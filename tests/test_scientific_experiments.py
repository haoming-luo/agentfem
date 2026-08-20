from __future__ import annotations

import json

import numpy as np
import pytest

from agentfem import amplitudes, campaigns, datasets, events, responses, results


def test_amplitude_basis_is_readable_differentiable_and_serializable():
    modes = amplitudes.basis(
        amplitudes.sine(amplitude=1.0, frequency=1.0, name="fast"),
        amplitudes.smooth_step(0.0, 1.0, name="slow"),
        name="actuators",
        coefficient_names=("a_fast", "a_slow"),
        value_unit="mm",
    )
    history = modes.combine({"a_fast": 2.0, "a_slow": -0.5}, name="load")

    time = 0.25
    expected = 2.0 * modes.components[0](time) - 0.5 * modes.components[1](time)
    assert history(time) == pytest.approx(expected)
    assert history.velocity(time) == pytest.approx(
        2.0 * modes.components[0].velocity(time)
        - 0.5 * modes.components[1].velocity(time)
    )
    assert modes.dimension == 2
    assert modes.fingerprint.startswith("sha256:")
    assert history.fingerprint.startswith("sha256:")
    assert json.dumps(history.to_dict())

    audit = history.audit(0.0, 1.0, samples=33)
    assert audit.finite is True
    assert audit.summary()["derivatives"]["2"]["finite"] is True

    transformed = modes.components[0].time_shifted(0.1).time_scaled(2.0).scaled(3.0)
    assert transformed(0.5) == pytest.approx(3.0 * modes.components[0](0.15))
    assert transformed.velocity(0.5) == pytest.approx(
        1.5 * modes.components[0].velocity(0.15)
    )
    assert transformed.fingerprint.startswith("sha256:")


def test_custom_callable_remains_an_explicit_nonserializable_escape_hatch():
    history = amplitudes.as_amplitude(lambda time: time**2)
    assert history.serializable is False
    assert history.fingerprint is None
    assert history.derivative(2.0) == pytest.approx(4.0, rel=1.0e-5)
    with pytest.raises(ValueError, match="cannot be serialized"):
        history.to_dict()


def test_first_passage_interpolates_and_preserves_censoring_semantics():
    history = results.HistoryResult(
        "opening",
        abscissa=(0.0, 1.0, 2.0),
        values=(0.0, 0.4, 1.2),
        unit="mm",
    )
    event = events.first_passage(history, threshold=0.8)
    censored = events.first_passage(history, threshold=2.0)

    assert event.status == "observed"
    assert event.coordinate == pytest.approx(1.5)
    assert event.bracket == (1.0, 2.0)
    assert event.value_name == "opening"
    assert event.value_unit == "mm"
    assert censored.status == "right_censored"
    assert censored.coordinate is None


def test_response_operator_reuses_campaign_and_recovers_vector_jacobian(tmp_path):
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("x", -10.0, 10.0),
        campaigns.RealParameter("y", -10.0, 10.0),
    )
    operator = responses.finite_difference(
        parameter_space=space,
        baseline={"x": 1.0, "y": 2.0},
        outputs=(
            datasets.Quantity("f"),
            datasets.Quantity("g", shape=(2,)),
        ),
        perturbation={"x": 0.1, "y": 0.2},
        step_mode="absolute",
        scheme="central",
        name="linear_response",
    )

    def evaluate(parameters):
        x = parameters["x"]
        y = parameters["y"]
        return {
            "f": x**2 + 3.0 * y,
            "g": np.asarray((x + y, 2.0 * x - y)),
        }

    campaign_report, response = operator.run(
        evaluate=evaluate,
        output_directory=tmp_path / "response",
    )

    assert campaign_report.completed == 5
    assert response.complete is True
    np.testing.assert_allclose(
        response.jacobian,
        np.asarray(((2.0, 3.0), (1.0, 1.0), (2.0, -1.0))),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert response.rank == 2
    assert response.conditioning_basis == "unscaled_homogeneous_units"
    assert response.derivative_units["f wrt x"] is None
    assert response.nonlinearity["y"] == pytest.approx(0.0, abs=1.0e-12)
    assert response.nonlinearity_by_output["x"]["f"] > 0.0
    assert response.nonlinearity_by_output["x"]["g"] == pytest.approx(
        0.0, abs=1.0e-12
    )
    assert (tmp_path / "response" / "response.json").is_file()

    reused_report, reused_response = operator.run(
        evaluate=evaluate,
        output_directory=tmp_path / "response",
    )
    assert all(record.reused for record in reused_report.records)
    np.testing.assert_allclose(reused_response.jacobian, response.jacobian)


def test_response_operator_keeps_failed_perturbations_visible():
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("x", -1.0, 1.0),
    )
    operator = responses.finite_difference(
        parameter_space=space,
        baseline={"x": 0.0},
        outputs=(datasets.Quantity("f"),),
        perturbation=0.1,
        step_mode="absolute",
    )

    def evaluate(parameters):
        if parameters["x"] > 0.0:
            raise RuntimeError("planted solver failure")
        return {"f": parameters["x"]}

    campaign_report, response = operator.run(evaluate=evaluate)
    assert campaign_report.failed == 1
    assert response.status == "incomplete"
    assert response.missing_cases == ("x:plus",)
    assert response.jacobian is None


def test_response_operator_does_not_hide_log_coordinates_or_mixed_unit_conditioning():
    log_space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("rate", 1.0, 100.0, scale="log"),
    )
    with pytest.raises(ValueError, match="logarithmic coordinates"):
        responses.finite_difference(
            parameter_space=log_space,
            baseline={"rate": 10.0},
            outputs=(datasets.Quantity("response"),),
        )

    mixed_space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("length", 1.0, 3.0, unit="m"),
        campaigns.RealParameter("time", 1.0, 3.0, unit="s"),
    )
    operator = responses.finite_difference(
        parameter_space=mixed_space,
        baseline={"length": 2.0, "time": 2.0},
        outputs=(datasets.Quantity("response", unit="Pa"),),
        perturbation=0.1,
        step_mode="absolute",
    )
    _, report = operator.run(
        evaluate=lambda values: {"response": values["length"] + values["time"]}
    )
    assert report.rank == 1
    assert report.singular_values is None
    assert report.condition_number is None
    assert report.conditioning_basis == "mixed_units_require_explicit_scaling"
