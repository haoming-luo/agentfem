from __future__ import annotations

import subprocess
import sys

import pytest

from agentfem import __version__, dependencies, mesh, platforms


def test_platform_support_does_not_overclaim_native_windows():
    native = platforms.support_for("Windows")
    wsl = platforms.support_for("Linux", wsl=True)
    wsl1 = platforms.support_for("Linux", wsl=True, wsl_version=1)

    assert native.level == "experimental"
    assert native.recommended is False
    assert any("PETSc" in item for item in native.limitations)
    assert any("dolfinx_mpc" in item for item in native.limitations)
    assert wsl.route == "Windows via WSL2/Linux"
    assert wsl.recommended is True
    assert wsl1.route == "Windows via WSL1/Linux"
    assert wsl1.level == "unsupported"


def test_optional_dependency_error_names_capability_and_install_extra(monkeypatch):
    def missing(_name):
        raise ModuleNotFoundError("planted missing package")

    monkeypatch.setattr(dependencies, "import_module", missing)

    with pytest.raises(dependencies.OptionalDependencyError) as error:
        dependencies.require(
            "gmsh",
            extra="gmsh",
            capability="Gmsh mesh import",
        )

    assert error.value.package == "gmsh"
    assert "agentfem[gmsh]" in str(error.value)
    assert "Gmsh mesh import" in str(error.value)


def test_mesh_namespace_keeps_external_gmsh_import_lazy():
    # This test is meaningful even in a development environment where Gmsh is
    # installed.  Use a fresh interpreter so the assertion is independent of
    # whether another mesh test has already exercised the optional backend.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import agentfem.mesh; "
            "raise SystemExit('gmsh' in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    capabilities = {item.package: item for item in mesh.optional_mesh_capabilities()}
    assert capabilities["gmsh"].extra == "gmsh"
    assert capabilities["meshio"].extra == "mesh-formats"


def test_runtime_report_is_serializable_and_names_optional_integrations():
    report = platforms.runtime_report().summary()
    assert report["schema"] == "agentfem.runtime-report"

    assert report["python"]
    assert report["platform"]["level"]
    assert report["operating_system"]["system"]
    assert report["packages"]["agentfem"] == __version__
    assert report["mpi"]["vendor"]
    assert report["mpi"]["family"] in {
        "mpich",
        "openmpi",
        "intelmpi",
        "msmpi",
        "unknown",
    }
    assert report["mpi"]["rank_count"] >= 1
    assert report["numerics"]["numpy_default_float"] == "float64"
    assert report["numerics"]["petsc_scalar_type"]
    assert "path_mismatch" in report["mpi"]
    assert report["mpi"]["code"].startswith("AFM-MPI-LAUNCHER-")
    assert isinstance(report["mpi"]["compatible"], bool)
    assert report["execution"]["python_executable"] == sys.executable
    assert report["execution"]["imported_package"]
    assert report["execution"]["mode"] in {
        "source_checkout",
        "installed_distribution",
    }
    assert isinstance(report["execution"]["distribution_mismatch"], bool)
    assert isinstance(report["execution"]["version_mismatch"], bool)
    assert isinstance(report["execution"]["path_mismatch"], bool)
    assert report["execution"]["environment_consistent"] is not report["execution"]["distribution_mismatch"]
    assert report["execution"]["runtime_version"] == __version__
    assert "source" in report["execution"]
    assert "distribution" in report["execution"]
    if report["execution"]["mode"] == "source_checkout":
        assert report["execution"]["source"]["commit"]
        assert isinstance(report["execution"]["source"]["tracked_dirty"], bool)
        assert len(report["execution"]["source"]["package_tree_sha256"]) == 64
    assert {item["package"] for item in report["optional"]} >= {
        "gmsh",
        "meshio",
        "torch",
    }


def test_runtime_identity_detects_stale_distribution_version(monkeypatch):
    monkeypatch.setattr(
        platforms,
        "_distribution_identity",
        lambda: {
            "version": "0.1.0",
            "installer": "pip",
            "direct_url": None,
            "record_sha256": None,
        },
    )

    identity = platforms._execution_identity(runtime_version="0.2.5")

    assert identity["version_mismatch"] is True
    assert identity["distribution_mismatch"] is True
    assert identity["environment_consistent"] is False
