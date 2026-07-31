"""Run a declarative parameter campaign with a trusted Python evaluator."""

from __future__ import annotations

from pathlib import Path

from agentfem import campaigns


def evaluate(parameters):
    """Replace this algebraic stand-in with a model-build/solve/result function."""

    return {
        "response": parameters["load"] / parameters["stiffness"],
    }


def main() -> None:
    spec_path = Path(__file__).with_name("campaign_from_json.spec.json")
    specification = campaigns.load_specification(spec_path)
    campaign = specification.create_campaign(evaluate=evaluate)
    report = campaign.run(
        specification.sampling,
        output_directory=Path("examples_output") / "campaign_from_json",
    )
    print(report.summary())


if __name__ == "__main__":
    main()
