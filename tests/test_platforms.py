from __future__ import annotations

import sys

import pytest

from agentfem import __version__, dependencies, mesh, platforms


def test_platform_support_does_not_overclaim_native_windows():
    native = platforms.support_for("Windows")
    wsl = platforms.support_for("Linux", wsl=True)

    assert native.level == "experimental"
    assert native.recommended is False
    assert any("PETSc" in item for item in native.limitations)
    assert any("dolfinx_mpc" in item for item in native.limitations)
    assert wsl.route == "Windows via WSL2/Linux"
    assert wsl.recommended is True


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
    # installed: importing agentfem.mesh must not import the external API.
    assert "gmsh" not in sys.modules
    capabilities = {item.package: item for item in mesh.optional_mesh_capabilities()}
    assert capabilities["gmsh"].extra == "gmsh"
    assert capabilities["meshio"].extra == "mesh-formats"


def test_runtime_report_is_serializable_and_names_optional_integrations():
    report = platforms.runtime_report().summary()

    assert report["python"]
    assert report["platform"]["level"]
    assert report["packages"]["agentfem"] == __version__
    assert report["mpi"]["vendor"]
    assert "path_mismatch" in report["mpi"]
    assert {item["package"] for item in report["optional"]} >= {
        "gmsh",
        "meshio",
        "torch",
    }
