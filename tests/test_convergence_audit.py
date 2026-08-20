from __future__ import annotations

import json

import pytest

from agentfem import campaigns, convergence, datasets


def _space():
    return campaigns.ParameterSpace.create(
        campaigns.RealParameter("h", 0.025, 0.2, unit="m"),
        campaigns.RealParameter("dt", 0.025, 0.1, unit="s"),
    )


def _sampling(space):
    return campaigns.explicit(
        space,
        (
            {"h": 0.2, "dt": 0.05},
            {"h": 0.1, "dt": 0.05},
            {"h": 0.05, "dt": 0.05},
            {"h": 0.05, "dt": 0.1},
            {"h": 0.05, "dt": 0.025},
        ),
    )


def _evaluate(values):
    return campaigns.CaseOutcome(
        outputs={"response": 1.0 + values["h"] ** 2 + values["dt"] ** 2},
        provenance={"events": {"order": ("yield", "peak")}},
    )


def test_multi_axis_convergence_certificate_is_explicit_and_reproducible(tmp_path):
    space = _space()
    campaign = campaigns.create(
        name="multi_axis",
        parameter_space=space,
        outputs=(datasets.Quantity("response"),),
        evaluate=_evaluate,
    )
    report = campaign.run(_sampling(space))
    output = tmp_path / "convergence.json"
    certificate = convergence.audit(
        report,
        axes=(
            convergence.axis("h", fixed={"dt": 0.05}),
            convergence.axis(
                "dt",
                fixed={"h": 0.05},
                discretization="time_step",
            ),
        ),
        observables=(
            convergence.observable(
                "response",
                tolerance=0.01,
                minimum_observed_order=1.9,
            ),
            convergence.observable(
                "event_order",
                comparison="exact",
                source="provenance",
                path="events.order",
            ),
        ),
        output=output,
    )

    assert certificate.passed is True
    assert len(certificate.checks) == 4
    numeric = tuple(
        check for check in certificate.checks if check.observable == "response"
    )
    assert all(check.observed_order == pytest.approx(2.0) for check in numeric)
    assert all(check.metric <= check.tolerance for check in numeric)
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["status"] == "passed"
    assert stored["certificate_id"].startswith("sha256:")


def test_convergence_failure_or_ambiguous_slice_is_not_hidden():
    space = _space()

    def evaluate(values):
        if values == {"h": 0.1, "dt": 0.05}:
            raise RuntimeError("planted refinement failure")
        return _evaluate(values)

    campaign = campaigns.create(
        name="incomplete_axis",
        parameter_space=space,
        outputs=(datasets.Quantity("response"),),
        evaluate=evaluate,
    )
    report = campaign.run(_sampling(space))
    certificate = convergence.audit(
        report,
        axes=(convergence.axis("h", fixed={"dt": 0.05}),),
        observables=(convergence.observable("response", tolerance=0.01),),
    )
    complete_report = campaigns.create(
        name="ambiguous_axis",
        parameter_space=space,
        outputs=(datasets.Quantity("response"),),
        evaluate=_evaluate,
    ).run(_sampling(space))
    ambiguous = convergence.audit(
        complete_report,
        axes=(convergence.axis("h"),),
        observables=(convergence.observable("response", tolerance=0.01),),
    )

    assert certificate.status == "inconclusive"
    assert len(certificate.checks[0].failed_case_ids) == 1
    assert ambiguous.status == "inconclusive"
    assert "uncontrolled parameters ('dt',)" in ambiguous.checks[0].message


def test_inverse_characteristic_supports_resolution_counts():
    space = campaigns.ParameterSpace.create(
        campaigns.IntegerParameter("elements", 10, 40),
    )
    campaign = campaigns.create(
        name="resolution_count",
        parameter_space=space,
        outputs=(datasets.Quantity("response"),),
        evaluate=lambda values: {"response": 1.0 + 1.0 / values["elements"] ** 2},
    )
    report = campaign.run(
        campaigns.explicit(
            space,
            ({"elements": 10}, {"elements": 20}, {"elements": 40}),
        )
    )
    certificate = convergence.audit(
        report,
        axes=(convergence.axis("elements", characteristic="inverse"),),
        observables=(
            convergence.observable(
                "response",
                tolerance=0.002,
                minimum_observed_order=1.9,
            ),
        ),
    )

    assert certificate.passed
    assert certificate.checks[0].observed_order == pytest.approx(2.0)


def test_correlated_unfixed_parameter_cannot_masquerade_as_refinement():
    space = _space()
    campaign = campaigns.create(
        name="confounded_refinement",
        parameter_space=space,
        outputs=(datasets.Quantity("response"),),
        evaluate=_evaluate,
    )
    report = campaign.run(
        campaigns.explicit(
            space,
            (
                {"h": 0.2, "dt": 0.1},
                {"h": 0.1, "dt": 0.05},
                {"h": 0.05, "dt": 0.025},
            ),
        )
    )

    certificate = convergence.audit(
        report,
        axes=(convergence.axis("h"),),
        observables=(convergence.observable("response", tolerance=0.01),),
    )

    assert certificate.status == "inconclusive"
    assert "uncontrolled parameters ('dt',)" in certificate.checks[0].message
