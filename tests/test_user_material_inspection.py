from __future__ import annotations

import json

from agentfem import cli
from agentfem.constitutive import user_material


def test_inspects_restricted_uhyper_candidate_without_claiming_execution(tmp_path):
    source = tmp_path / "rubber.for"
    source.write_text(
        """      SUBROUTINE UHYPER(BI1,BI2,AJ,U,UI1,UI2,UI3,TEMP,\n"
        "     1 NOEL,CMNAME,INCMPFLAG,NUMSTATEV,STATEV,NUMFIELDV,\n"
        "     2 FIELDV,FIELDVINC,NUMPROPS,PROPS)\n"
        "      INCLUDE 'ABA_PARAM.INC'\n"
        "      U(1)=PROPS(1)*(BI1-3.D0)\n"
        "      RETURN\n"
        "      END\n""",
        encoding="utf-8",
    )

    report = user_material.inspect_abaqus_user_material(source)

    assert report.status == "adapter_candidate"
    assert report.interface == "UHYPER"
    assert report.route == "restricted_uhyper_energy_adapter"
    assert report.summary()["executable"] is False
    assert report.source_sha256


def test_inspection_routes_umat_with_abaqus_utility_to_manual_adaptation(tmp_path):
    source = tmp_path / "legacy_umat.f"
    source.write_text(
        """      SUBROUTINE UMAT(STRESS,STATEV,DDSDDE,SSE,SPD,SCD)\n"
        "      CALL ROTSIG(STRESS,DROT,ROTATED,1,3,3)\n"
        "      RETURN\n"
        "      END\n""",
        encoding="utf-8",
    )

    report = user_material.inspect_abaqus_user_material(source)

    assert report.status == "manual_adaptation_required"
    assert report.interface == "UMAT"
    assert report.abaqus_utility_calls == ("ROTSIG",)
    assert "AFM-USERMAT-SOURCE-003" in {
        item.code for item in report.findings
    }


def test_cli_emits_portable_user_material_inspection(tmp_path, capsys):
    source = tmp_path / "material.f90"
    source.write_text(
        "subroutine umat(stress,statev,ddsdde,stran,dstran,time,dtime,props,nprops)\n"
        "end subroutine umat\n",
        encoding="utf-8",
    )
    destination = tmp_path / "inspection.json"

    assert cli.main(
        [
            "inspect-user-material",
            str(source),
            "--write",
            str(destination),
            "--json",
        ]
    ) == 0

    emitted = json.loads(capsys.readouterr().out)
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert emitted["schema"] == "agentfem.abaqus-user-material-inspection"
    assert emitted["interface"] == "UMAT"
    assert saved["source_sha256"] == emitted["source_sha256"]


def test_inspection_fingerprints_local_include_graph_and_runtime_header(tmp_path):
    source = tmp_path / "umat.f90"
    helper = tmp_path / "hardening.inc"
    helper.write_text(
        "subroutine hardening(statev)\nreal*8 statev(*)\nend subroutine\n",
        encoding="utf-8",
    )
    source.write_text(
        "subroutine umat(stress,statev,ddsdde,stran,dstran,time,dtime,props,nprops)\n"
        "include 'aba_param.inc'\n"
        "include 'hardening.inc'\n"
        "call hardening(statev)\n"
        "end subroutine umat\n",
        encoding="utf-8",
    )

    first = user_material.inspect_abaqus_user_material(source)
    assert first.status == "adapter_candidate"
    assert first.source_graph.complete is True
    assert len(first.source_graph.files) == 2
    assert first.source_graph.runtime_includes == ("ABA_PARAM.INC",)
    assert first.project_calls == ("HARDENING",)
    assert first.external_calls == ()
    assert {item.status for item in first.source_graph.edges} == {
        "resolved",
        "runtime_provided",
    }

    helper.write_text(
        "subroutine hardening(statev)\nreal*8 statev(*)\nstatev(1)=0d0\n"
        "end subroutine\n",
        encoding="utf-8",
    )
    second = user_material.inspect_abaqus_user_material(source)
    assert second.source_sha256 == first.source_sha256
    assert second.source_graph.fingerprint != first.source_graph.fingerprint


def test_missing_project_include_is_an_addressable_blocker(tmp_path):
    source = tmp_path / "umat.f90"
    source.write_text(
        "subroutine umat(stress,statev,ddsdde,stran,dstran,time,dtime,props,nprops)\n"
        "include 'missing-project.inc'\n"
        "end subroutine umat\n",
        encoding="utf-8",
    )

    report = user_material.inspect_abaqus_user_material(source)

    assert report.status == "manual_adaptation_required"
    assert report.source_graph.complete is False
    assert "AFM-USERMAT-INCLUDE-001" in {
        item.code for item in report.findings
    }


def test_recursive_project_include_is_not_silently_flattened(tmp_path):
    source = tmp_path / "umat.f90"
    helper = tmp_path / "recursive.inc"
    source.write_text(
        "subroutine umat(stress,statev,ddsdde,stran,dstran,time,dtime,props,nprops)\n"
        "include 'recursive.inc'\n"
        "end subroutine umat\n",
        encoding="utf-8",
    )
    helper.write_text("include 'umat.f90'\n", encoding="utf-8")

    report = user_material.inspect_abaqus_user_material(source)

    assert report.status == "manual_adaptation_required"
    assert any(item.status == "cycle" for item in report.source_graph.edges)
    assert "AFM-USERMAT-INCLUDE-002" in {
        item.code for item in report.findings
    }


def test_migration_binds_deck_material_to_fingerprinted_source_graph(
    tmp_path, capsys
):
    deck = tmp_path / "model.inp"
    deck.write_text(
        "*Material, name=LEGACY_STEEL\n"
        "*User Material, constants=1\n210000.\n"
        "*Depvar\n1\n",
        encoding="utf-8",
    )
    source_directory = tmp_path / "legacy_material"
    source_directory.mkdir()
    source = source_directory / "umat.f90"
    helper = source_directory / "hardening.inc"
    source.write_text(
        "subroutine umat(stress,statev,ddsdde,stran,dstran,time,dtime,props,nprops)\n"
        "include 'aba_param.inc'\n"
        "include 'hardening.inc'\n"
        "call hardening(statev)\n"
        "end subroutine umat\n",
        encoding="utf-8",
    )
    helper.write_text(
        "subroutine hardening(statev)\nreal*8 statev(*)\nend subroutine\n",
        encoding="utf-8",
    )
    target = tmp_path / "migrated"

    assert cli.main(
        [
            "migrate-abaqus",
            str(deck),
            str(target),
            "--user-material",
            f"LEGACY_STEEL={source}",
            "--json",
        ]
    ) == 0

    emitted = json.loads(capsys.readouterr().out)
    migration = json.loads((target / "migration.json").read_text(encoding="utf-8"))
    association = migration["user_materials"][0]
    assert emitted["user_materials"][0]["material"] == "LEGACY_STEEL"
    assert association["material"] == "LEGACY_STEEL"
    assert association["status"] == "adapter_candidate"
    assert association["executable"] is False
    assert association["inspection"]["source_graph"]["complete"] is True
    assert len(association["bundled_sources"]) == 2
    assert (target / association["source_entrypoint"]).is_file()
    assert (
        target
        / "materials"
        / "user_materials"
        / "LEGACY_STEEL"
        / "source"
        / "hardening.inc"
    ).is_file()
    assert "User-material source assets" in (
        target / "migration.md"
    ).read_text(encoding="utf-8")


def test_migration_rejects_unmatched_user_material_name_atomically(
    tmp_path, capsys
):
    deck = tmp_path / "model.inp"
    deck.write_text(
        "*Material, name=DECK_NAME\n*User Material\n1.0\n",
        encoding="utf-8",
    )
    source = tmp_path / "umat.f90"
    source.write_text(
        "subroutine umat(stress,statev,ddsdde,stran,dstran,time,dtime,props,nprops)\n"
        "end subroutine umat\n",
        encoding="utf-8",
    )
    target = tmp_path / "migrated"

    assert cli.main(
        [
            "migrate-abaqus",
            str(deck),
            str(target),
            "--user-material",
            f"WRONG_NAME={source}",
            "--json",
        ]
    ) == 2

    failure = json.loads(capsys.readouterr().out)
    assert failure["status"] == "failed"
    assert "does not match a deck material" in failure["error"]["message"]
    assert not target.exists()
