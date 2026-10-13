# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Packaging admission independent of DOLFINx or a native compiler."""
import importlib.util
from pathlib import Path
import zipfile

from packaging.tags import Tag
import pytest

spec = importlib.util.spec_from_file_location(
    "platform_wheel_build", Path(__file__).parents[1] / "tools/build_platform_wheel.py",
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def wheel(tmp_path, platform, *, body=True, q1=False):
    path = tmp_path / f"agentfem-0.4.2-cp311-cp311-{platform}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("agentfem/_hex_native.cpython-311.so", b"fixture, not executable")
        if body:
            archive.writestr("agentfem/_body_pipeline.cpython-311.so", b"fixture")
        if q1:
            archive.writestr("agentfem/_q1_native.cpython-311.so", b"fixture")
    return path


def test_native_silence_and_wrong_platform_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "sys_tags", lambda: [Tag("cp311", "cp311", "linux_x86_64")])
    with pytest.raises(ValueError, match="omitted required"):
        builder.inspect_wheel(wheel(tmp_path, "linux_x86_64", body=False))
    with pytest.raises(ValueError, match="compatible"):
        builder.inspect_wheel(wheel(tmp_path, "macosx_11_0_arm64"))


def test_raw_linux_tag_cannot_pass_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "sys_tags", lambda: [Tag("cp311", "cp311", "linux_x86_64")])
    with pytest.raises(ValueError, match="portable platform tag"):
        builder.inspect_wheel(wheel(tmp_path, "linux_x86_64"), require_portable_linux=True)


def test_portable_candidate_records_optional_geometry_truthfully(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "sys_tags", lambda: [Tag("cp311", "cp311", "manylinux_2_28_x86_64")])
    record = builder.inspect_wheel(wheel(tmp_path, "manylinux_2_28_x86_64"), require_portable_linux=True)
    assert record["packaged_native_modules"] == {
        "_hex_native": True, "_body_pipeline": True, "_q1_native": False,
    }


def test_source_selection_and_candidate_overwrite_are_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        builder.select_source(tmp_path)
    (tmp_path / "agentfem-0.4.2.tar.gz").write_bytes(b"source fixture")
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "retained"
    sentinel.write_bytes(b"old evidence")
    with pytest.raises(FileExistsError):
        builder.build(tmp_path, output)
    assert sentinel.read_bytes() == b"old evidence"
    (tmp_path / "agentfem-0.4.3.tar.gz").write_bytes(b"another source")
    with pytest.raises(ValueError, match="exactly one"):
        builder.select_source(tmp_path)
