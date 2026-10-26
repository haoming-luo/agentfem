"""Discovery must work even when the numerical runtime cannot initialize."""

import subprocess
import sys

import pytest

from agentfem import __version__


@pytest.mark.parametrize("arguments,expected_code,expected_text", [
    (["--help"], 0, "usage: agentfem"),
    (["--version"], 0, __version__),
    (["init", "--help"], 0, "--template"),
    (["run", "--help"], 0, "--name"),
    (["doctor", "--help"], 0, "--json"),
    (["unknown-command"], 2, "invalid choice"),
])
def test_cli_help_version_and_usage_do_not_import_numerical_runtime(
    arguments, expected_code, expected_text,
):
    code = '''
import importlib.abc
import sys
class DenyNumericalImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mpi4py", "dolfinx", "petsc4py"}:
            raise AssertionError("Discovery imported numerical runtime: " + fullname)
sys.meta_path.insert(0, DenyNumericalImport())
from agentfem.cli import main
raise SystemExit(main())
'''
    result = subprocess.run([sys.executable, "-c", code, *arguments],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == expected_code, result.stdout + result.stderr
    assert expected_text in result.stdout + result.stderr


def test_deferred_mpi_uses_real_module_without_guessing_rank(monkeypatch):
    from types import ModuleType, SimpleNamespace
    from agentfem import cli

    package = ModuleType("mpi4py")
    communicator = object()
    runtime = SimpleNamespace(COMM_WORLD=communicator)
    package.MPI = runtime
    monkeypatch.setitem(sys.modules, "mpi4py", package)
    monkeypatch.setattr(cli, "MPI", cli._DeferredMPI())
    assert cli.MPI.COMM_WORLD is communicator
    assert cli.MPI is runtime
