from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from mpi4py import MPI

import agentfem
from agentfem import (
    constitutive,
    fields,
    fracture,
    integrations,
    mechanics,
    mesh,
    models,
    platforms,
    solvers,
    studies,
)
from agentfem.backends import runtime
from agentfem.constitutive import elasticity
from agentfem.step_providers import StepRequest


def _force_native(monkeypatch):
    monkeypatch.setenv("AGENTFEM_RUNTIME", "fenicsx-native-serial")
    selected = runtime.current_runtime()
    assert selected.name == "fenicsx-native-serial"
    assert selected.supports("linear_solve")
    return selected


def test_native_runtime_capability_contract(monkeypatch):
    selected = _force_native(monkeypatch)
    assert selected.supports("matrix_assembly", "vector_assembly")
    assert not selected.supports("petsc_nonlinear_solve")
    assert not selected.distributed


def test_runtime_selection_rejects_unknown_name(monkeypatch):
    monkeypatch.setenv("AGENTFEM_RUNTIME", "native-windowz")
    with np.testing.assert_raises(runtime.RuntimeSelectionError):
        runtime.current_runtime()


def test_runtime_probe_rejects_present_but_unimportable_module(monkeypatch):
    monkeypatch.setattr(runtime, "find_spec", lambda _module: object())

    def broken_import(_module):
        raise OSError("missing compiled runtime DLL")

    monkeypatch.setattr(runtime, "import_module", broken_import)
    assert runtime._available("petsc4py") is False


def test_explicit_petsc_selection_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setenv("AGENTFEM_RUNTIME", "fenicsx-petsc")
    availability = {
        "dolfinx": True,
        "petsc4py": False,
        "dolfinx.fem.petsc": False,
        "scipy": True,
        "pyamg": True,
        "dolfinx_mpc": False,
    }
    monkeypatch.setattr(runtime, "_available", availability.__getitem__)
    with np.testing.assert_raises(runtime.RuntimeSelectionError):
        runtime.current_runtime()


def test_explicit_runtime_selection_requires_dolfinx(monkeypatch):
    monkeypatch.setenv("AGENTFEM_RUNTIME", "fenicsx-native-serial")
    monkeypatch.setattr(runtime, "_available", lambda module: module != "dolfinx")
    with pytest.raises(runtime.RuntimeSelectionError, match="DOLFINx is unavailable"):
        runtime.current_runtime()


def test_native_jacobi_policy_is_executed(monkeypatch):
    _force_native(monkeypatch)
    matrix = csr_matrix(np.diag([2.0, 4.0, 8.0]))
    rhs = np.array([2.0, 8.0, 24.0])
    values, info = solvers._solve_native_matrix(
        matrix,
        rhs,
        solvers.LinearSolverOptions(
            ksp_type="cg",
            pc_type="jacobi",
            rtol=1.0e-12,
            atol=1.0e-14,
        ),
    )
    assert info.converged
    assert info.backend == "scipy_cg"
    assert info.preconditioner == "jacobi"
    assert values == pytest.approx((1.0, 2.0, 3.0))


def test_native_direct_solver_rejects_singular_system(monkeypatch):
    _force_native(monkeypatch)
    matrix = csr_matrix(np.array([[1.0, 1.0], [2.0, 2.0]]))
    with pytest.raises(RuntimeError, match="did not converge"):
        solvers._solve_native_matrix(
            matrix,
            np.array([1.0, 2.0]),
            solvers.direct_solver(),
        )


def test_native_iterative_solver_rejects_unmapped_preconditioner(monkeypatch):
    _force_native(monkeypatch)
    matrix = csr_matrix(np.eye(2))
    with pytest.raises(ValueError, match="supports pc_type"):
        solvers._solve_native_matrix(
            matrix,
            np.ones(2),
            solvers.LinearSolverOptions(ksp_type="cg", pc_type="ilu"),
        )


def test_native_provider_rejects_distributed_mesh_before_lowering(monkeypatch):
    _force_native(monkeypatch)
    registry = models.StepProviderRegistry()
    registry.register(
        models.StepProvider(
            name="fake_linear",
            analyses=("linear_static",),
            accepts=lambda _model, _request: True,
            lower=lambda _model, _request: object(),
            requires=("linear_solve",),
        )
    )
    fake_model = SimpleNamespace(
        mesh=SimpleNamespace(comm=SimpleNamespace(size=2)),
        materials=(),
    )
    request = StepRequest(
        analysis="linear_static",
        target=None,
        options={},
    )
    with pytest.raises(runtime.RuntimeCapabilityError) as caught:
        registry.resolve(fake_model, request)
    assert caught.value.missing == ("mpi_distributed_mesh",)


def test_native_public_namespaces_import_without_petsc(monkeypatch):
    selected = _force_native(monkeypatch)
    assert fracture.__name__ == "agentfem.fracture"
    assert mechanics.__name__ == "agentfem.mechanics"
    assert solvers.__name__ == "agentfem.solvers"
    assert integrations.pdeagent_bench.BENCHMARK_NAME == "PDEAgent-Bench"
    assert platforms.runtime_report().solver_runtime["name"] == selected.name


def test_native_public_api_inventory_is_importable(monkeypatch):
    _force_native(monkeypatch)
    imported = tuple(
        importlib.import_module(f"agentfem.{name}")
        for name in agentfem.PUBLIC_WORKFLOW_MODULES
    )
    assert len(imported) == len(agentfem.PUBLIC_WORKFLOW_MODULES)


def test_native_static_solid_uses_same_public_workflow(monkeypatch):
    _force_native(monkeypatch)
    domain = mesh.rectangle(
        (0.0, 0.0), (1.0, 0.2), (4, 1),
        comm=MPI.COMM_SELF, cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.static_solid(dimension=2, assumption="plane_stress"),
        mesh=domain,
        name="windows_native_patch",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        elasticity.isotropic_elastic(young=1.0e6, poisson=0.3, density=1.0)
    )
    model.clamp(
        displacement,
        on=mesh.face(domain, axis="x", value=0.0, name="fixed", tag=1),
    )
    model.traction(
        (100.0, 0.0),
        on=mesh.face(domain, axis="x", value=1.0, name="loaded", tag=2),
    )
    result = model.step(target=displacement).solve_result()
    solve = result.metadata["step"]["problem"]["last_solve"]
    assert solve["converged"] is True
    assert solve["backend"] == "scipy_spsolve"
    assert solve["preconditioner"] == "superlu"
    assert np.max(np.abs(displacement.value.x.array)) > 0.0


def test_native_newmark_uses_prepared_scipy_system(monkeypatch, tmp_path):
    _force_native(monkeypatch)
    domain = mesh.rectangle(
        (0.0, 0.0), (1.0, 0.2), (2, 1),
        comm=MPI.COMM_SELF, cell_type="triangle",
    )
    model = models.create(
        study=studies.implicit_dynamics(
            physics="solid_mechanics",
            dimension=2,
            assumption="plane_stress",
            method="newmark",
        ),
        mesh=domain,
        name="windows_native_newmark",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        elasticity.isotropic_elastic(
            young=2.0e5,
            poisson=0.3,
            density=1.0e3,
        )
    )
    model.clamp(
        displacement,
        on=mesh.face(domain, axis="x", value=0.0, name="fixed", tag=1),
    )
    model.traction(
        (10.0, 0.0),
        on=mesh.face(domain, axis="x", value=1.0, name="loaded", tag=2),
    )

    step = model.step(
        target=displacement,
        dt=1.0e-3,
        steps=2,
        progress=False,
    )
    result = step.solve_result(output=tmp_path / "native-newmark.xdmf")

    assert result.status == "completed"
    assert step.procedure.algorithm == "newmark"
    assert step.problem.last_solve_info.backend == "scipy_spsolve"
    assert np.max(np.abs(step.state.u.value.x.array)) > 0.0


def test_native_runtime_rejects_nonlinear_provider_before_lowering(monkeypatch):
    _force_native(monkeypatch)
    domain = mesh.rectangle(
        (0.0, 0.0), (1.0, 0.2), (2, 1),
        comm=MPI.COMM_SELF, cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics", dimension=2, assumption="plane_strain"
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    model.material(constitutive.neo_hookean(young=1.0e6, poisson=0.3))
    capability = models.step_capability(model, target=displacement)
    report = model.validate(target=displacement)

    assert capability["supported"] is False
    assert capability["runtime"]["name"] == "fenicsx-native-serial"
    assert capability["runtime_compatible"] is False
    assert capability["missing_capabilities"] == ("petsc_nonlinear_solve",)
    assert any(item.code == "AFM-STUDY-002" for item in report.errors)
    try:
        model.step(target=displacement)
    except runtime.RuntimeCapabilityError as error:
        assert error.code == "AFM-BACKEND-CAPABILITY-001"
        assert "petsc_nonlinear_solve" in error.missing
    else:  # pragma: no cover
        raise AssertionError("The native runtime silently accepted a PETSc-only step.")
