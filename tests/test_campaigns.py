from __future__ import annotations

import json

import numpy as np
import pytest

from agentfem import campaigns, datasets, results, verification


def _space():
    return campaigns.ParameterSpace.create(
        campaigns.RealParameter("young", 100.0, 300.0, unit="GPa"),
        campaigns.IntegerParameter("layers", 1, 4),
        campaigns.ChoiceParameter("support", ("fixed", "pinned")),
        name="design",
    )


def test_sampling_is_reproducible_valid_and_case_ids_are_stable():
    space = _space()
    first = campaigns.latin_hypercube(space, 8, seed=42)
    second = campaigns.latin_hypercube(space, 8, seed=42)

    assert first.samples == second.samples
    assert all(space.validate(sample) == sample for sample in first.samples)
    assert campaigns.case_id("beam", first.samples[0]) == campaigns.case_id(
        "beam", second.samples[0]
    )


def test_plan_shards_are_disjoint_and_complete():
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("x", 0.0, 1.0)
    )
    campaign = campaigns.create(
        name="shards",
        parameter_space=space,
        outputs=(datasets.Quantity("y"),),
        evaluate=lambda values: {"y": values["x"]},
    )
    plan = campaign.plan(campaigns.random(space, 9, seed=7))
    shards = [campaign.plan(campaigns.explicit(space, [case.parameters])) for case in plan.cases]

    assert len({shard.cases[0].case_id for shard in shards}) == len(plan.cases)
    shard0 = plan.shard(0, 2)
    shard1 = plan.shard(1, 2)
    assert {case.case_id for case in shard0.cases}.isdisjoint(
        {case.case_id for case in shard1.cases}
    )
    assert len(shard0.cases) + len(shard1.cases) == len(plan.cases)


def test_campaign_records_failures_and_resumes_completed_cases(tmp_path):
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("x", 0.0, 1.0)
    )
    calls = {"count": 0}

    def evaluate(values):
        calls["count"] += 1
        if np.isclose(values["x"], 0.5):
            raise RuntimeError("planted failure")
        return campaigns.CaseOutcome(
            outputs={"response": values["x"] ** 2},
            provenance={"solver": "unit_reference"},
        )

    campaign = campaigns.create(
        name="resume",
        parameter_space=space,
        outputs=(datasets.Quantity("response", unit="m"),),
        evaluate=evaluate,
    )
    sampling = campaigns.explicit(
        space,
        ({"x": 0.0}, {"x": 0.5}, {"x": 1.0}),
    )
    first = campaign.run(sampling, output_directory=tmp_path)
    second = campaign.run(sampling, output_directory=tmp_path)

    assert first.completed == 2
    assert first.failed == 1
    assert second.completed == 2
    assert second.failed == 1
    assert calls["count"] == 4
    assert sum(record.reused for record in second.records) == 2
    assert second.dataset is not None
    assert len(second.dataset.samples) == 2
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["failed"] == 1

    with pytest.raises(RuntimeError, match="failed case"):
        second.require_dataset()
    accepted = second.require_dataset(allow_partial=True)
    assert len(accepted.samples) == 2
    decision = accepted.metadata["partial_campaign_acceptance"]
    assert decision["accepted"] is True
    assert decision["failed_case_count"] == 1
    assert len(decision["failed_case_ids"]) == 1


def test_campaign_captures_model_ir_from_built_case():
    class Case:
        def __init__(self, parameters):
            self.parameters = parameters

        def to_ir(self, metadata=None):
            return {"kind": "test_model", "metadata": metadata}

    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("x", 0.0, 1.0)
    )
    campaign = campaigns.create(
        name="provenance",
        parameter_space=space,
        outputs=(datasets.Quantity("y"),),
        build=Case,
        evaluate=lambda case: {"y": case.parameters["x"]},
    )
    report = campaign.run(campaigns.explicit(space, ({"x": 0.25},)))

    provenance = report.dataset.samples[0].provenance
    assert provenance["model_ir"]["kind"] == "test_model"


def test_campaign_does_not_hide_a_nonroot_mpi_failure():
    class FakeComm:
        rank = 0

        def barrier(self):
            return None

        def bcast(self, value, root=0):
            assert root == 0
            return value

        def allgather(self, local):
            return (
                local,
                {
                    "rank": 1,
                    "successful": False,
                    "error_type": "RuntimeError",
                    "error_message": "nonroot failure",
                },
            )

    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("x", 0.0, 1.0)
    )
    campaign = campaigns.create(
        name="mpi_failure",
        parameter_space=space,
        outputs=(datasets.Quantity("y"),),
        evaluate=lambda values: {"y": values["x"]},
    )
    report = campaign.run(
        campaigns.explicit(space, ({"x": 0.25},)),
        comm=FakeComm(),
    )

    assert report.failed == 1
    assert report.dataset is None
    assert report.records[0].error_type == "agentfem.campaigns.MPIRankFailure"
    assert "rank 1" in report.records[0].error_message


def test_json_specification_drives_sampling_execution_and_dataset(tmp_path):
    spec_path = tmp_path / "campaign.json"
    spec_path.write_text(
        json.dumps(
            {
                "name": "configured",
                "parameters": [
                    {
                        "kind": "real",
                        "name": "x",
                        "lower": 1.0,
                        "upper": 3.0,
                        "unit": "m",
                    },
                    {
                        "kind": "choice",
                        "name": "mode",
                        "choices": ["a", "b"],
                    },
                ],
                "sampling": {
                    "method": "explicit",
                    "samples": [
                        {"x": 1.0, "mode": "a"},
                        {"x": 3.0, "mode": "b"},
                    ],
                },
                "outputs": [{"name": "y", "unit": "m"}],
                "execution": {"resume": True},
            }
        ),
        encoding="utf-8",
    )
    specification = campaigns.load_specification(spec_path)
    campaign = specification.create_campaign(
        evaluate=lambda values: {"y": values["x"] * 2.0}
    )

    report = campaign.run(
        specification.sampling,
        output_directory=tmp_path / "run",
    )

    assert report.completed == 2
    assert specification.parameter_space.names == ("x", "mode")
    np.testing.assert_allclose(report.dataset.y_matrix(), [[2.0], [6.0]])


def test_campaign_accepts_simulation_result_without_serializing_live_fields(tmp_path):
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("load", 1.0, 2.0, unit="N")
    )

    def evaluate(values):
        result = results.SimulationResult(
            "beam",
            metadata={"backend": "test"},
        )
        result.add_quantity("tip_displacement", values["load"] * 0.1, unit="m")
        result.add_field("displacement", artifact="beam.xdmf")
        result.add_artifact("fields", "beam.xdmf")
        return result

    campaign = campaigns.create(
        name="result_bridge",
        parameter_space=space,
        outputs=(datasets.Quantity("tip_displacement", unit="m"),),
        evaluate=evaluate,
    )
    report = campaign.run(
        campaigns.explicit(space, ({"load": 1.5},)),
        output_directory=tmp_path,
    )

    sample = report.dataset.samples[0]
    assert sample.outputs["tip_displacement"] == pytest.approx(0.15)
    assert sample.artifacts == {"fields": "beam.xdmf"}
    assert sample.provenance["simulation_result"]["fields"] == ("displacement",)


def test_learning_dataset_can_require_verified_simulation_evidence():
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("load", 1.0, 2.0)
    )

    def evaluate(values):
        result = results.SimulationResult("beam")
        result.add_quantity("response", values["load"])
        if values["load"] < 1.5:
            claim = verification.VerificationClaim.compare(
                name="reference",
                observable="response",
                actual=values["load"],
                expected=values["load"],
                reference="analytical",
            )
            result.add_verification(verification.report(claim))
        return result

    campaign = campaigns.create(
        name="trust_gate",
        parameter_space=space,
        outputs=(datasets.Quantity("response"),),
        evaluate=evaluate,
    )
    report = campaign.run(
        campaigns.explicit(space, ({"load": 1.25}, {"load": 1.75}))
    )

    with pytest.raises(RuntimeError, match="below required trust level"):
        report.require_dataset(minimum_trust_level="verified")
    assert report.require_dataset(minimum_trust_level="computed") is report.dataset
