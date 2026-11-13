import hashlib
import json

import numpy as np
import pytest

from tools import p92_bam_source
from tools.p92_bam_source import read_source


def test_partial_download_cannot_be_admitted_as_experiment(tmp_path):
    path = tmp_path / "Data_RF.lis"
    path.write_bytes(b"BAM 5.2 *Data_RF*.LIS\r\n[Data]\r\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_source(path, protocol="RF")


def test_protocol_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="protocol"):
        read_source(tmp_path / "missing", protocol="pooled")


def fixture_file(tmp_path, monkeypatch, rows, units="s\tNo\tMPa\t%\t°C"):
    raw = ("Operator\tPrivate Name\nSpecimen Designation (Internal)\tfixture\n"
           "Cycles to Failure, Nf\t10%\t20\nTotal No. of Cycles, N\t25\n"
           "[Data]\nTime\tCycle\tStress\tStrain\tTemperature\n"
           + units + "\n" + rows).encode("latin-1")
    path = tmp_path / "synthetic.lis"
    path.write_bytes(raw)
    monkeypatch.setitem(p92_bam_source.CHECKSUMS, "RF",
                        hashlib.md5(raw, usedforsecurity=False).hexdigest())
    return path


def test_units_and_missing_cycles_are_preserved(tmp_path, monkeypatch):
    path = fixture_file(tmp_path, monkeypatch,
                        "0\t1\t100\t0.2\t620\n736\t3\t-50\t-0.2\t620\n")
    data = read_source(path, protocol="RF")
    np.testing.assert_array_equal(data["recorded_cycles"], [1, 3])
    np.testing.assert_allclose(data["strain"], [.002, -.002])
    np.testing.assert_allclose(data["stress_Pa"], [1e8, -5e7])
    np.testing.assert_allclose(data["temperature_K"], [893.15, 893.15])
    assert not data["engineering_qualified"]
    assert data["metadata"] == {"specimen_id": "fixture", "failure_cycles": "20",
                                 "total_test_cycles": "25"}


@pytest.mark.parametrize("rows,reason", [
    ("1 1 100 .2 620\n0 1 100 .2 620\n", "monotone"),
    ("0 1.5 100 .2 620\n", "integers"),
    ("0 1 nan .2 620\n", "values"),
])
def test_invalid_history_rejected(tmp_path, monkeypatch, rows, reason):
    path = fixture_file(tmp_path, monkeypatch, rows)
    with pytest.raises(ValueError, match=reason):
        read_source(path, protocol="RF")


def test_unexpected_units_are_not_guessed(tmp_path, monkeypatch):
    path = fixture_file(tmp_path, monkeypatch, "0 1 100 .2 620\n",
                        units="s No Pa % °C")
    with pytest.raises(ValueError, match="units"):
        read_source(path, protocol="RF")


def test_audit_integrates_only_recorded_path_without_closing_it(tmp_path, monkeypatch):
    raw = ("Specimen Designation (Internal)\tfixture\nCycles to Failure, Nf\t2\n"
           "Total No. of Cycles, N\t3\n"
           "[MAX-MIN]\nCycle Stress-Max Stress-Min Strain-Max Strain-Min Temperature-Max Temperature-Min\n"
           "No MPa MPa % % °C °C\n1 2 0 1 0 620 620\n3 2 0 1 0 620 620\n"
           "[Data]\nTime Cycle Stress Strain Temperature\ns No MPa % °C\n"
           "0 1 0 0 620\n1 1 2 1 620\n100 3 2 1 620\n101 3 0 0 620\n").encode("latin-1")
    path = tmp_path / "fixture.lis"
    path.write_bytes(raw)
    monkeypatch.setitem(p92_bam_source.CHECKSUMS, "RF",
                        hashlib.md5(raw, usedforsecurity=False).hexdigest())
    report = p92_bam_source.audit_source(path, protocol="RF")
    json.dumps(report, allow_nan=False)
    assert [r["cycle"] for r in report["records"]] == [1, 3]
    assert [r["sampled_path_work_J_per_m3"] for r in report["records"]] == [10000, -10000]
    assert report["records"][0]["endpoint_strain_difference"] == .01
    assert report["life_observation"]["life"] == 2


def test_protocols_are_separate_life_cohorts_not_cycle_replicates():
    from agentfem.assessments import life_survival

    data = {"metadata": {"specimen_id": "fixture", "failure_cycles": "20",
                          "total_test_cycles": "25"}, "sha256": "fixture"}
    rf = p92_bam_source.life_observation(dict(data, protocol="RF"))
    slr = p92_bam_source.life_observation(dict(data, protocol="SLR",
        metadata=dict(data["metadata"], specimen_id="other")))
    assert rf.life == 20
    assert rf.cohort_key != slr.cohort_key
    with pytest.raises(ValueError, match="Mixed population"):
        life_survival([rf, slr], independent_specimens=True, noninformative_censoring=True)
