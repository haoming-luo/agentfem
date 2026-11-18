# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from agentfem._viewer_files import ResultDirectory


def test_lists_only_directories_and_supported_results(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "result.vtu").touch()
    (tmp_path / "case.py").touch()
    (tmp_path / ".hidden.vtu").touch()
    directory = ResultDirectory(tmp_path)
    assert [entry["value"] for entry in directory.entries()] == ["frames", "result.vtu"]
    assert directory.result("result.vtu") == tmp_path / "result.vtu"


def test_reject_parent_and_symlink_escape(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.vtu"
    outside.touch()
    (root / "link.vtu").symlink_to(outside)
    directory = ResultDirectory(root)
    assert directory.entries() == []
    for name in ("../outside.vtu", "link.vtu", str(outside)):
        with pytest.raises(ValueError, match="inside"):
            directory.result(name)


def test_xdmf_checks_companions_and_preserves_relative_continuation(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "joined").mkdir()
    (tmp_path / "frames/data.h5").touch()
    index = tmp_path / "joined/results.xdmf"
    index.write_text('<Xdmf><Domain><DataItem Format="HDF">../frames/data.h5:/u</DataItem></Domain></Xdmf>')
    directory = ResultDirectory(tmp_path)
    assert directory.result(index) == index
    (tmp_path / "frames/data.h5").unlink()
    with pytest.raises(FileNotFoundError):
        directory.result(index)


def test_reject_xdmf_companion_escape(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (tmp_path / "outside.h5").touch()
    index = root / "result.xdmf"
    index.write_text('<Xdmf><DataItem Format="HDF">../outside.h5:/u</DataItem></Xdmf>')
    with pytest.raises(ValueError, match="inside"):
        ResultDirectory(root).result(index)


def test_pvd_checks_every_frame(tmp_path):
    (tmp_path / "first.vtu").touch()
    path = tmp_path / "series.pvd"
    path.write_text('<VTKFile><Collection><DataSet file="first.vtu"/><DataSet file="missing.vtu"/></Collection></VTKFile>')
    with pytest.raises(FileNotFoundError):
        ResultDirectory(tmp_path).result(path)
    (tmp_path / "missing.vtu").touch()
    assert ResultDirectory(tmp_path).result(path) == path


def test_no_xml_includes_or_python_files(tmp_path):
    index = tmp_path / "result.xdmf"
    index.write_text('<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude"><xi:include href="other.xml"/></Xdmf>')
    with pytest.raises(ValueError, match="includes"):
        ResultDirectory(tmp_path).result(index)
    code = tmp_path / "case.py"
    code.write_text('raise AssertionError("must never execute")')
    with pytest.raises(ValueError, match="Select XDMF"):
        ResultDirectory(tmp_path).result(code)


def test_module_help_needs_no_viewer_dependencies(capsys):
    from agentfem.viewer import main
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "directory" in capsys.readouterr().out
