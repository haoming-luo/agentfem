"""Geometry and exact-affine contracts for a true periodic void cell."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from agentfem import (
    benchmarks,
    checkpointing,
    constitutive,
    fields,
    models,
    provenance,
    results,
    solvers,
    steps,
    studies,
)

from periodic_void_fixture import (
    PERIODIC_VOID_GMSH_OPTIONS,
    periodic_spherical_void_cell,
)


BENCHMARK_ID = "agentfem.benchmark.finite_strain_j2_periodic_void"
_REFERENCE_SOLUTION = None


def _benchmark_card() -> dict[str, object]:
    path = (
        Path(benchmarks.__file__).resolve().parents[1]
        / "knowledge"
        / "benchmarks"
        / "finite_strain_j2_periodic_void.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _run_real_void_j2(*, mesh_size: float, increments: int):
    pytest.importorskip("gmsh")
    reference = _benchmark_card()["regression_identity"]["scientific_input"]
    geometry = reference["geometry"]
    material_input = reference["material"]
    macro_deformation = np.asarray(
        reference["macroscopic_deformation_gradient"],
        dtype=float,
    )
    if not np.allclose(
        macro_deformation,
        np.diag(np.diag(macro_deformation)),
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError("The reference fixture requires a diagonal Fbar.")
    stretch = float(macro_deformation[0, 0])
    expected_macro_deformation = np.diag(
        (stretch, 1.0 / np.sqrt(stretch), 1.0 / np.sqrt(stretch))
    )
    np.testing.assert_allclose(
        macro_deformation,
        expected_macro_deformation,
        rtol=0.0,
        atol=2.0e-15,
    )
    fixture = periodic_spherical_void_cell(
        MPI.COMM_SELF,
        side_length=float(geometry["side_length"]),
        void_radius=float(geometry["void_radius"]),
        mesh_size=mesh_size,
        stretch=stretch,
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="finite_strain_j2_real_void_smoke",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=float(material_input["young"]),
            poisson=float(material_input["poisson"]),
            yield_stress=float(material_input["yield_stress"]),
            hardening_modulus=float(material_input["hardening_modulus"]),
            tangent_relative_step=float(
                material_input["tangent_relative_step"]
            ),
        )
    )
    periodicity = model.constraint(fixture.constraint(displacement))
    newton_input = reference["newton"]
    solver_options = solvers.newton(
        relative_tolerance=float(newton_input["relative_tolerance"]),
        absolute_tolerance=float(newton_input["absolute_tolerance"]),
        maximum_iterations=int(newton_input["maximum_iterations"]),
        line_search=str(newton_input["line_search"]),
        linear_solver=solvers.direct_solver(),
    )
    quadrature_degree = int(reference["quadrature_degree"])
    step = model.step(
        target=displacement,
        material=material,
        constraints=periodicity,
        incrementation=steps.fixed(increments),
        solver_options=solver_options,
        quadrature_degree=quadrature_degree,
        output_every=1,
        progress=False,
    )
    result = step.solve_result()
    frames = results.homogenize_periodic_path(
        step.snapshots,
        step.state_transaction.material,
        constraint=periodicity,
    )
    hill_mandel = results.hill_mandel_periodic_path(
        step.snapshots,
        step.state_transaction.material,
        constraint=periodicity,
        frames=frames,
    )
    minimum_j = min(
        float(np.min(np.linalg.det(snapshot.fields["F"].owned_values)))
        for snapshot in step.snapshots
    )
    peeq = step.response.state.committed["equivalent_plastic_strain"]
    statistics = peeq.weighted_statistics(quantiles=(0.95, 0.99))
    energy_closure = max(
        float(
            np.max(
                np.abs(
                    snapshot.fields["SENER"].owned_values
                    - snapshot.fields["ELENER"].owned_values
                    - snapshot.fields["HARDENER"].owned_values
                ),
                initial=0.0,
            )
        )
        for snapshot in step.snapshots
    )
    exact_fraction = fixture.exact_solid_volume / fixture.cell_reference_volume
    solver_summary = solver_options.summary()
    linear_summary = solver_summary["linear_solver"]
    scientific_input = {
        "geometry": {
            "side_length": fixture.side_length,
            "void_radius": fixture.void_radius,
            "mesh_size": float(mesh_size),
            "exact_solid_volume": fixture.exact_solid_volume,
        },
        "material": {
            name: float(material_input[name])
            for name in (
                "young",
                "poisson",
                "yield_stress",
                "hardening_modulus",
                "tangent_relative_step",
            )
        },
        "macroscopic_deformation_gradient": (
            fixture.deformation_gradient.tolist()
        ),
        "constraint_fingerprint": periodicity.scientific_identity()[
            "fingerprint"
        ],
        "equation_count": int(
            periodicity.scientific_identity()["equation_count"]
        ),
        "increments": int(increments),
        "accepted_load_factors": [
            float(snapshot.load_factor) for snapshot in step.snapshots
        ],
        "quadrature_degree": quadrature_degree,
        "newton": {
            "relative_tolerance": float(
                solver_summary["relative_tolerance"]
            ),
            "absolute_tolerance": float(
                solver_summary["absolute_tolerance"]
            ),
            "maximum_iterations": int(
                solver_summary["maximum_iterations"]
            ),
            "line_search": solver_summary["line_search"],
            "linear_solver": (
                f"{linear_summary['ksp_type']}_{linear_summary['pc_type']}"
            ),
        },
    }
    return {
        "fixture": fixture,
        "periodicity": periodicity,
        "step": step,
        "result": result,
        "frame": frames[-1],
        "statistics": statistics,
        "mesh_identity": checkpointing.mesh_portable_identity(fixture.domain),
        "global_cells": int(
            fixture.domain.topology.index_map(3).size_global
        ),
        "scientific_input": scientific_input,
        "scientific_input_fingerprint": provenance.content_fingerprint(
            scientific_input
        ),
        "minimum_j": minimum_j,
        "maximum_peeq": statistics.maximum,
        "maximum_hill_mandel_relative_error": max(
            item.relative_error for item in hill_mandel
        ),
        "energy_closure": energy_closure,
        "geometry_relative_error": abs(
            frames[-1].solid_reference_fraction - exact_fraction
        )
        / exact_fraction,
    }


@pytest.fixture(scope="module")
def real_void_j2_solution():
    return _reference_solution()


def _reference_solution():
    global _REFERENCE_SOLUTION
    if _REFERENCE_SOLUTION is None:
        reference = _benchmark_card()["regression_identity"]["scientific_input"]
        _REFERENCE_SOLUTION = _run_real_void_j2(
            mesh_size=float(reference["geometry"]["mesh_size"]),
            increments=int(reference["increments"]),
        )
    return _REFERENCE_SOLUTION


def test_gmsh_spherical_void_mesh_has_exact_periodic_source_equations():
    pytest.importorskip("gmsh")
    fixture = periodic_spherical_void_cell(MPI.COMM_SELF)
    displacement = fields.displacement(fixture.domain)
    periodicity = fixture.constraint(displacement)
    reduction = periodicity.reduction()

    global_cells = fixture.domain.topology.index_map(3).size_global
    global_vertices = fixture.domain.topology.index_map(0).size_global
    solid_volume = fem.assemble_scalar(
        fem.form(ufl.as_ufl(1.0) * ufl.dx(domain=fixture.domain))
    )

    assert global_cells > 0
    assert reduction.full_size == 3 * global_vertices
    assert 0 < reduction.reduced_size < reduction.full_size
    assert fixture.periodic_pairing_error < 1.0e-13
    assert len(fixture.equations.equations) > 0
    assert periodicity.reference_cell_volume == pytest.approx(1.0)
    assert solid_volume == pytest.approx(fixture.exact_solid_volume, rel=8.0e-3)
    assert set(np.unique(fixture.cell_tags.values)) == {1}
    assert set(np.unique(fixture.facet_tags.values)) == {10, 20}

    periodicity.apply_affine_increment(0.0, 1.0)
    assert periodicity.mismatch() < 1.0e-12
    np.testing.assert_allclose(
        periodicity.measured_deformation_gradient(displacement),
        fixture.deformation_gradient,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_real_void_finite_strain_j2_uses_public_result_lifecycle(
    real_void_j2_solution,
):
    evidence = real_void_j2_solution
    fixture = evidence["fixture"]
    result = evidence["result"]
    final = evidence["frame"]
    _assert_physical_gates(evidence)
    assert 0.95 < final.solid_reference_fraction < 0.99
    assert np.isfinite(final.first_piola_stress).all()
    assert final.first_piola_stress[0, 0] > 0.0
    assert {
        "Displacement",
        "F",
        "P",
        "S",
        "MISES",
        "SENER",
        "ELENER",
        "HARDENER",
        "FP",
        "PEEQ",
    } <= set(result.fields)
    assert fixture.periodic_pairing_error < 1.0e-12 * fixture.side_length


def test_periodic_void_golden_contract_is_machine_readable():
    card = _benchmark_card()
    golden = benchmarks.golden_benchmark(BENCHMARK_ID)

    assert card["status"] == (
        "automated_fixed_stack_regression_experimental_science"
    )
    assert card["regression_identity"]["reference_source"]["clean_tree"]
    assert card["regression_identity"]["scientific_input_fingerprint"] == (
        provenance.content_fingerprint(
            card["regression_identity"]["scientific_input"]
        )
    )
    assert card["regression_identity"]["fixed_stack"]["gmsh"]["options"] == (
        PERIODIC_VOID_GMSH_OPTIONS
    )
    assert golden.reference_version == "periodic-void-j2-ef3268c-f202c023-v1"
    assert {item.name for item in golden.quantities} == {
        "homogenized_first_piola_stress",
        "peeq_mean",
        "peeq_p95",
        "peeq_p99",
        "solid_reference_fraction",
    }
    assert "peeq_maximum_diagnostic" not in {
        item.name for item in golden.quantities
    }


def test_fixed_stack_applicability_does_not_gate_candidate_agentfem_version(
    monkeypatch,
):
    card = _benchmark_card()
    before = _fixed_stack_mismatches(card)
    manifest = provenance.runtime_manifest()
    manifest["identity"]["packages"]["agentfem"] = "future-candidate"
    monkeypatch.setattr(provenance, "runtime_manifest", lambda: manifest)

    after = _fixed_stack_mismatches(card)
    assert after == before
    assert "agentfem" not in after


def test_periodic_void_fixed_stack_matches_versioned_golden():
    card = _benchmark_card()
    mismatches = _fixed_stack_mismatches(card)
    if mismatches:
        message = "Fixed-stack Golden is inapplicable: " + "; ".join(mismatches)
        if os.environ.get("AGENTFEM_REQUIRE_RVE_GOLDEN") == "1":
            pytest.fail(message)
        pytest.skip(message)

    evidence = _reference_solution()
    _assert_physical_gates(evidence)
    expected_identity = card["regression_identity"]["mesh_identity"]
    assert evidence["mesh_identity"] == expected_identity
    assert evidence["scientific_input"] == card["regression_identity"][
        "scientific_input"
    ]
    assert evidence["scientific_input_fingerprint"] == card[
        "regression_identity"
    ]["scientific_input_fingerprint"]
    assert (
        evidence["periodicity"].scientific_identity()["fingerprint"]
        == card["regression_identity"]["scientific_input"][
            "constraint_fingerprint"
        ]
    )
    statistics = evidence["statistics"]
    actual = {
        "homogenized_first_piola_stress": (
            evidence["frame"].first_piola_stress.reshape(-1)
        ),
        "peeq_mean": statistics.mean,
        "peeq_p95": statistics.quantiles[0.95],
        "peeq_p99": statistics.quantiles[0.99],
        "solid_reference_fraction": evidence["frame"].solid_reference_fraction,
    }
    golden = benchmarks.golden_benchmark(BENCHMARK_ID)
    for quantity in golden.quantities:
        quantity.assert_accepts(actual[quantity.name])
    assert [
        float(snapshot.load_factor) for snapshot in evidence["step"].snapshots
    ] == [0.0, 0.5, 1.0]
    assert evidence["statistics"].sample_count == 493
    assert evidence["statistics"].total_weight == pytest.approx(
        evidence["frame"].solid_reference_fraction,
        rel=1.0e-12,
        abs=1.0e-12,
    )


@pytest.mark.skipif(
    os.environ.get("AGENTFEM_RUN_RVE_CONVERGENCE") != "1",
    reason="set AGENTFEM_RUN_RVE_CONVERGENCE=1 for the RVE refinement certificate",
)
def test_periodic_void_successive_refinement_certificate(real_void_j2_solution):
    coarse_two = real_void_j2_solution
    coarse_four = _run_real_void_j2(mesh_size=0.25, increments=4)
    medium = _run_real_void_j2(mesh_size=0.18, increments=4)
    fine = _run_real_void_j2(mesh_size=0.14, increments=4)

    for evidence in (coarse_two, coarse_four, medium, fine):
        _assert_physical_gates(evidence)
    assert coarse_two["mesh_identity"] == coarse_four["mesh_identity"]
    assert (
        coarse_two["global_cells"]
        < medium["global_cells"]
        < fine["global_cells"]
    )
    assert len(
        {
            coarse_two["mesh_identity"]["geometry_connectivity_hash"],
            medium["mesh_identity"]["geometry_connectivity_hash"],
            fine["mesh_identity"]["geometry_connectivity_hash"],
        }
    ) == 3
    assert (
        coarse_two["geometry_relative_error"]
        > medium["geometry_relative_error"]
        > fine["geometry_relative_error"]
    )

    increment_stress_change = _tensor_relative_change(coarse_two, coarse_four)
    increment_mean_change = _scalar_relative_change(
        coarse_two["statistics"].mean,
        coarse_four["statistics"].mean,
    )
    increment_p95_change = _scalar_relative_change(
        coarse_two["statistics"].quantiles[0.95],
        coarse_four["statistics"].quantiles[0.95],
    )
    mesh_stress_change = _tensor_relative_change(medium, fine)
    mesh_mean_change = _scalar_relative_change(
        medium["statistics"].mean,
        fine["statistics"].mean,
    )
    mesh_p95_change = _scalar_relative_change(
        medium["statistics"].quantiles[0.95],
        fine["statistics"].quantiles[0.95],
    )
    certificate = {
        "schema": "agentfem.rve-successive-refinement.v1",
        "global_cells": [
            coarse_two["global_cells"],
            medium["global_cells"],
            fine["global_cells"],
        ],
        "increment_relative_changes": {
            "macro_first_piola": increment_stress_change,
            "peeq_mean": increment_mean_change,
            "peeq_p95": increment_p95_change,
        },
        "mesh_relative_changes": {
            "macro_first_piola": mesh_stress_change,
            "peeq_mean": mesh_mean_change,
            "peeq_p95": mesh_p95_change,
        },
        "geometry_relative_errors": [
            coarse_two["geometry_relative_error"],
            medium["geometry_relative_error"],
            fine["geometry_relative_error"],
        ],
    }
    print(json.dumps(certificate, indent=2, sort_keys=True))
    observed = _benchmark_card()["refinement_certificate"]["observed"]
    assert certificate["global_cells"] == observed["global_cells"]
    for group in (
        "increment_relative_changes",
        "mesh_relative_changes",
    ):
        for name, expected in observed[group].items():
            assert certificate[group][name] == pytest.approx(
                expected,
                rel=1.0e-6,
                abs=1.0e-12,
            )
    np.testing.assert_allclose(
        certificate["geometry_relative_errors"],
        observed["geometry_relative_errors"],
        rtol=1.0e-8,
        atol=1.0e-12,
    )

    assert increment_stress_change < 2.0e-3
    assert increment_mean_change < 2.0e-3
    assert increment_p95_change < 1.0e-2
    assert mesh_stress_change < 2.0e-2
    assert mesh_mean_change < 1.0e-2
    assert mesh_p95_change < 5.0e-2
    assert fine["geometry_relative_error"] < 7.0e-3


def _assert_physical_gates(evidence) -> None:
    fixture = evidence["fixture"]
    periodicity = evidence["periodicity"]
    step = evidence["step"]
    result = evidence["result"]
    statistics = evidence["statistics"]
    assert result.status == "completed"
    assert step.last_solve_info.converged
    assert step.accepted_load_factor == pytest.approx(1.0)
    accepted = [float(item.load_factor) for item in step.snapshots]
    assert accepted[0] == pytest.approx(0.0)
    assert accepted[-1] == pytest.approx(1.0)
    assert np.all(np.diff(accepted) > 0.0)
    assert periodicity.mismatch() < 1.0e-10 * fixture.side_length
    assert fixture.periodic_pairing_error < 1.0e-12 * fixture.side_length
    assert evidence["minimum_j"] > 0.99
    assert evidence["maximum_peeq"] > 1.0e-4
    assert evidence["maximum_hill_mandel_relative_error"] < 1.0e-8
    assert evidence["energy_closure"] < 1.0e-12
    assert statistics.sample_count == evidence["global_cells"]
    assert statistics.total_weight == pytest.approx(
        evidence["frame"].solid_reference_fraction,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def _fixed_stack_mismatches(card) -> tuple[str, ...]:
    expected = card["regression_identity"]["fixed_stack"]
    current = provenance.runtime_manifest()["identity"]
    mismatches = []
    platform = expected["platform"]
    if current["operating_system"]["system"] != platform["system"]:
        mismatches.append("operating system")
    if current["machine"] != platform["machine"]:
        mismatches.append("machine architecture")
    if current["python"] != expected["python"]:
        mismatches.append("Python")
    for name, version in expected["packages"].items():
        if name == "agentfem":
            continue
        if current["packages"].get(name) != version:
            mismatches.append(name)
    if current["mpi"] != expected["mpi"]:
        mismatches.append("MPI")
    if current["numerics"] != expected["numerics"]:
        mismatches.append("scalar/numeric contract")

    try:
        import gmsh
    except ImportError:
        if os.environ.get("AGENTFEM_REQUIRE_RVE_GOLDEN") == "1":
            mismatches.append("Gmsh is unavailable")
            return tuple(mismatches)
        pytest.skip("fixed-stack Golden requires optional gmsh")
    current_gmsh = {
        "version": gmsh.__version__,
        "options": PERIODIC_VOID_GMSH_OPTIONS,
    }
    if current_gmsh != expected["gmsh"]:
        mismatches.append("Gmsh version/generation policy")
    return tuple(mismatches)


def _tensor_relative_change(left, right) -> float:
    first = np.asarray(left["frame"].first_piola_stress, dtype=float)
    second = np.asarray(right["frame"].first_piola_stress, dtype=float)
    return float(
        np.linalg.norm(first - second)
        / max(np.linalg.norm(second), np.finfo(float).tiny)
    )


def _scalar_relative_change(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(
        abs(float(right)),
        np.finfo(float).tiny,
    )
