import hashlib

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
