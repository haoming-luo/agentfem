from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
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
    assert isinstance(report["solver_ready"], bool)
    assert report["required"]["scipy"]["available"] is True

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
    assert isinstance(report["toolchain"]["jit_ready"], bool)
    assert "c_compiler" in report["toolchain"]
    assert report["toolchain"]["purpose"] == "compile uncached FFCx/CFFI forms"
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
    assert set(report["execution"]["accelerators"]) == {
        "hex_arithmetic", "material_body_pipeline", "q1_geometry"
    }
    assert report["workspace"]["schema"] == "agentfem.workspace"
    assert isinstance(
        report["workspace"]["protected_from_distribution_removal"], bool
    )
    if report["execution"]["mode"] == "source_checkout":
        assert report["execution"]["source"]["commit"]
        assert isinstance(report["execution"]["source"]["tracked_dirty"], bool)
        assert len(report["execution"]["source"]["package_tree_sha256"]) == 64
        assert len(report["execution"]["source"]["scientific_runtime_sha256"]) == 64
        assert report["execution"]["source"]["scientific_runtime_exclusions"] == (
            "knowledge/benchmarks/**",
        )
    assert {item["package"] for item in report["optional"]} >= {
        "gmsh",
        "meshio",
        "torch",
        "matplotlib",
    }


def test_native_inventory_distinguishes_missing_broken_and_loadable(monkeypatch):
    seen = []
    def importer(name):
        seen.append(name)
        if name == "agentfem._hex_native":
            raise ModuleNotFoundError("optional extension absent", name=name)
        if name == "agentfem._q1_native":
            raise ImportError("GMP shared library unavailable")
        def forbidden(*args):
            raise AssertionError("Inventory must not execute an accelerator")
        return SimpleNamespace(__file__="installed/_body_pipeline.so", run=forbidden)
    monkeypatch.setattr(platforms, "import_module", importer)
    records = platforms._native_accelerators()
    json.dumps(records, allow_nan=False)
    assert len(seen) == 3
    assert records["hex_arithmetic"]["status"] == "not_built"
    assert records["material_body_pipeline"]["available"] is True
    assert records["q1_geometry"]["status"] == "load_failed"
    assert records["q1_geometry"]["error"]["message"] == "GMP shared library unavailable"


@pytest.mark.parametrize("policy", [None, "wrong-policy"])
def test_native_inventory_rejects_incomplete_modules_and_wrong_q1_policy(monkeypatch, policy):
    monkeypatch.setattr(platforms, "import_module",
                        lambda name: SimpleNamespace(__file__=name, projection_policy=policy))
    assert all(item["status"] == "load_failed" for item in platforms._native_accelerators().values())


def test_native_inventory_does_not_mislabel_a_missing_dependency(monkeypatch):
    def broken(name):
        raise ModuleNotFoundError("dependency unavailable", name="external_dependency")
    monkeypatch.setattr(platforms, "import_module", broken)
    assert all(item["status"] == "load_failed" for item in platforms._native_accelerators().values())


def test_runtime_report_fails_closed_when_dolfinx_has_no_jit_compiler(monkeypatch):
    monkeypatch.setattr(
        platforms,
        "_compiler_runtime",
        lambda: {
            "c_compiler": None,
            "cxx_compiler": None,
            "jit_ready": False,
            "purpose": "compile uncached FFCx/CFFI forms",
        },
    )

    report = platforms.runtime_report()

    assert report.packages["fenics-dolfinx"]
    assert report.solver_ready is False
    assert report.summary()["solver_ready"] is False
    assert "Solver readiness: not ready" in report.format()


@pytest.mark.parametrize("failure", [ModuleNotFoundError("No module named 'scipy'"),
                                     OSError("incompatible SciPy shared library")])
def test_doctor_rejects_missing_or_broken_scipy_despite_installed_metadata(monkeypatch, failure):
    original = platforms.import_module

    def importing(name):
        if name == "scipy.sparse.linalg":
            raise failure
        return original(name)

    monkeypatch.setattr(platforms, "import_module", importing)
    report = platforms.runtime_report()
    assert report.packages["scipy"]  # Metadata alone must not imply readiness.
    assert not report.solver_ready
    assert report.required["scipy"]["code"] == "AFM-RUNTIME-DEPENDENCY-UNAVAILABLE"
    assert str(failure) in report.format()
    json.dumps(report.summary(), allow_nan=False)


def test_doctor_missing_dolfinx_never_reports_solver_ready():
    from dataclasses import replace

    report = platforms.runtime_report()
    missing = replace(report, packages={**report.packages, "fenics-dolfinx": None})
    assert not missing.solver_ready


def test_optional_default_tools_do_not_block_existing_mesh_solves(monkeypatch):
    original = dependencies.status

    def unavailable(package, **kwargs):
        if package in {"gmsh", "matplotlib"}:
            return dependencies.DependencyStatus(package, kwargs["extra"], kwargs["capability"], False)
        return original(package, **kwargs)

    monkeypatch.setattr(dependencies, "status", unavailable)
    report = platforms.runtime_report()
    assert report.solver_ready
    assert not any(item.available for item in report.optional if item.package in {"gmsh", "matplotlib"})


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


def test_scientific_runtime_identity_excludes_only_benchmark_declarations(tmp_path):
    package = tmp_path / "agentfem"
    solver = package / "solver.py"
    benchmark = package / "knowledge" / "benchmarks" / "example.json"
    card = package / "knowledge" / "cards" / "example.json"
    benchmark.parent.mkdir(parents=True)
    card.parent.mkdir(parents=True)
    solver.write_text("VALUE = 1\n", encoding="utf-8")
    benchmark.write_text('{"status": "experimental"}\n', encoding="utf-8")
    card.write_text('{"statement": "runtime input"}\n', encoding="utf-8")

    exact_before = platforms._source_tree_digest(package)
    runtime_before = platforms._source_tree_digest(
        package,
        excluded_subtrees=(Path("knowledge/benchmarks"),),
    )
    benchmark.write_text('{"status": "accepted"}\n', encoding="utf-8")

    assert platforms._source_tree_digest(package) != exact_before
    assert (
        platforms._source_tree_digest(
            package,
            excluded_subtrees=(Path("knowledge/benchmarks"),),
        )
        == runtime_before
    )

    card.write_text('{"statement": "changed runtime input"}\n', encoding="utf-8")
    runtime_after_card = platforms._source_tree_digest(
        package,
        excluded_subtrees=(Path("knowledge/benchmarks"),),
    )
    assert runtime_after_card != runtime_before

    solver.write_text("VALUE = 2\n", encoding="utf-8")
    assert (
        platforms._source_tree_digest(
            package,
            excluded_subtrees=(Path("knowledge/benchmarks"),),
        )
        != runtime_after_card
    )
