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
