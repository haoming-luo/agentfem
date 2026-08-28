from __future__ import annotations

import importlib

import numpy as np
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
