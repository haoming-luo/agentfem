from __future__ import annotations

from hashlib import sha256
import json
from zipfile import ZipFile

import numpy as np
import pytest

from agentfem import datasets


def _manifest_for(path, *, roles=("observable",)):
    payload = path.read_bytes()
    return datasets.ExternalDatasetManifest.from_mapping(
        {
            "schema": "agentfem.external-scientific-dataset",
            "schema_version": "0.1.0",
            "dataset_id": "synthetic-evidence-v1",
            "title": "Synthetic evidence",
            "doi": "10.0000/example",
            "version_id": "1",
            "license": "CC0-1.0",
            "landing_page": "https://example.invalid/dataset",
            "default_required_roles": ["observable"],
            "files": [
                {
                    "path": path.name,
                    "size": len(payload),
                    "sha256": sha256(payload).hexdigest(),
                    "roles": list(roles),
                }
            ],
        }
    )


def test_science_supershear_manifest_pins_public_dryad_version_and_roles():
    manifest = datasets.science_supershear_dryad_manifest()

    assert manifest.dataset_id == "science-2023-supershear-dryad-v7"
    assert manifest.version_id == "235603"
    assert manifest.license == "CC0-1.0"
    assert len(manifest.files) == 26
    assert len(manifest.files_for_roles(("crack_speed",))) == 1
    assert len(manifest.files_for_roles(("sed_ked_field",))) == 4
    assert manifest.summary()["total_size"] == sum(item.size for item in manifest.files)

    task = datasets.science_supershear_v5_research_task()
    assert task["status"] == "ready_for_research_execution"
    assert [item["id"] for item in task["work_packages"]] == [
        "V5-A", "V5-B", "V5-C", "V5-D", "V5-E"
    ]
    with pytest.raises(ValueError, match="no files for roles"):
        manifest.files_for_roles(("not_a_scientific_role",))


def test_external_dataset_audit_detects_missing_size_and_digest(tmp_path):
    source = tmp_path / "observable.bin"
    source.write_bytes(b"public scientific evidence")
    manifest = _manifest_for(source)

    assert manifest.audit(tmp_path).require().accepted
    altered = bytearray(source.read_bytes())
    altered[0] ^= 1
    source.write_bytes(altered)
    audit = manifest.audit(tmp_path)
    assert not audit.accepted
    assert audit.digest_mismatches == (source.name,)

    source.write_bytes(b"short")
    audit = manifest.audit(tmp_path)
    assert audit.size_mismatches == (source.name,)

    source.unlink()
    audit = manifest.audit(tmp_path)
    assert audit.missing_files == (source.name,)
    with pytest.raises(RuntimeError, match="audit failed"):
        audit.require()


def test_external_manifest_rejects_unknown_schema(tmp_path):
    source = tmp_path / "evidence.bin"
    source.write_bytes(b"evidence")
    data = json.loads(
        json.dumps(
            {
                "schema": "some-other-schema",
                "schema_version": "0.1.0",
                "dataset_id": "bad",
                "title": "bad",
                "doi": "bad",
                "version_id": "1",
                "license": "CC0",
                "landing_page": "https://example.invalid",
                "files": [],
            }
        )
    )
    with pytest.raises(ValueError, match="Unsupported external dataset schema"):
        datasets.ExternalDatasetManifest.from_mapping(data)


def test_dependency_free_xlsx_reader_handles_strings_numbers_boolean_and_formula(tmp_path):
    path = tmp_path / "observations.xlsx"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
              xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
              <sheets><sheet name="crack speed" sheetId="1" r:id="rId1"/></sheets>
            </workbook>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rId1" Target="worksheets/sheet1.xml"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>
            </Relationships>""",
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <si><t>time</t></si><si><t>speed</t></si>
            </sst>""",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <sheetData>
                <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
                <row r="2"><c r="A2"><v>0</v></c><c r="B2"><v>1.25</v></c><c r="C2" t="b"><v>1</v></c></row>
                <row r="3"><c r="A3"><v>2</v></c><c r="B3"><f>A3*2</f><v>4</v></c></row>
              </sheetData>
            </worksheet>""",
        )

    workbook = datasets.read_xlsx_workbook(path)
    sheet = workbook.sheet("crack speed")

    assert workbook.summary()["sheets"][0]["rows"] == 3
    assert sheet.rows[0] == ("time", "speed")
    assert sheet.rows[1] == (0, 1.25, True)
    assert sheet.rows[2] == (2, 4)
    np.testing.assert_allclose(
        sheet.numeric_block(start_row=1, start_column=0, stop_column=2),
        [[0.0, 1.25], [2.0, 4.0]],
    )
