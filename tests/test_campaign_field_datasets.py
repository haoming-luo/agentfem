from __future__ import annotations

import numpy as np
import pytest

from agentfem import campaigns, datasets, learning


def _field_campaign(*, fail_at: float | None = None):
    space = campaigns.ParameterSpace.create(
        campaigns.RealParameter("load", 0.0, 2.0, unit="N")
    )

    def evaluate(values):
        load = float(values["load"])
        if fail_at is not None and np.isclose(load, fail_at):
            raise RuntimeError("planted field-case failure")
        return campaigns.CaseOutcome(
            outputs={"peak": 2.0 * load},
            provenance={"solver": "manufactured_field_reference"},
            artifacts={"field_source": f"memory://load/{load:g}"},
        )

    campaign = campaigns.create(
        name="field_bridge",
        parameter_space=space,
        outputs=(datasets.Quantity("peak", unit="m"),),
        evaluate=evaluate,
        scientific_inputs={"mesh": "two_point_reference"},
    )
    sampling = campaigns.explicit(
        space,
        ({"load": 0.5}, {"load": 1.0}, {"load": 1.5}),
    )
    return campaign, sampling


def _assembler(*, broken: bool = False):
    forcing = learning.FieldEncoding(
        name="forcing",
        role="input",
        unit="N",
        representation="point_samples",
        shape=(2,),
        mesh_policy="mesh_independent_coordinates",
    )
    displacement = learning.FieldEncoding(
        name="displacement",
        role="output",
        unit="m",
        representation="point_samples",
        shape=(2,),
        mesh_policy="mesh_independent_coordinates",
    )

    def extract(case, outcome):
        load = float(case.parameters["load"])
        fields = {
            "forcing": np.asarray([load, load]),
            "displacement": np.asarray([load, 2.0 * load]),
        }
        if broken and np.isclose(load, 1.5):
            fields["displacement"] = np.asarray([load])
        return datasets.FieldCaseData(
            fields=fields,
            coordinates={"points": np.asarray([[0.0], [1.0]])},
            metadata={"peak_from_outcome": outcome.outputs["peak"]},
        )

    return datasets.FieldDatasetAssembler(
        encodings=(forcing, displacement),
        extract=extract,
        parameter_names=("load",),
        name="field_bridge_dataset",
        metadata={"purpose": "campaign_field_bridge_test"},
    )


def test_campaign_report_assembles_quality_gated_field_dataset(tmp_path):
    campaign, sampling = _field_campaign()
    report = campaign.run(sampling, output_directory=tmp_path / "campaign")

    field_dataset = report.require_field_dataset(_assembler())
    manifest = field_dataset.write(tmp_path / "fields")
    restored = datasets.ScientificFieldDataset.read(manifest)

    assert field_dataset.case_ids == tuple(
        record.case.case_id for record in report.records
    )
    assert field_dataset.input_names == ("forcing",)
    assert field_dataset.output_names == ("displacement",)
    assert field_dataset.fields["displacement"].shape == (3, 2)
    np.testing.assert_allclose(field_dataset.parameters["load"], [0.5, 1.0, 1.5])
    np.testing.assert_allclose(
        field_dataset.coordinates["points"],
        np.broadcast_to(np.asarray([[0.0], [1.0]]), (3, 2, 1)),
    )
    assert field_dataset.case_metadata[0]["campaign_provenance"]["solver"] == (
        "manufactured_field_reference"
    )
    assert field_dataset.case_metadata[0]["campaign_artifacts"]["field_source"].startswith(
        "memory://"
    )
    assert field_dataset.metadata["campaign_dataset_metadata"]["campaign"] == (
        "field_bridge"
    )
    assert restored.fingerprint == field_dataset.fingerprint


def test_field_dataset_bridge_inherits_partial_campaign_gate(tmp_path):
    campaign, sampling = _field_campaign(fail_at=1.0)
    report = campaign.run(sampling, output_directory=tmp_path / "partial")

    with pytest.raises(RuntimeError, match="failed case"):
        report.require_field_dataset(_assembler())

    accepted = report.require_field_dataset(_assembler(), allow_partial=True)
    assert accepted.case_count == 2
    decision = accepted.metadata["campaign_dataset_metadata"][
        "partial_campaign_acceptance"
    ]
    assert decision["accepted"] is True
    assert decision["failed_case_count"] == 1


def test_field_dataset_bridge_rejects_inconsistent_case_shapes(tmp_path):
    campaign, sampling = _field_campaign()
    report = campaign.run(sampling, output_directory=tmp_path / "shape")

    with pytest.raises(ValueError, match="shapes must match"):
        report.require_field_dataset(_assembler(broken=True))

