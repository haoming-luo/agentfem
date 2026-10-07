"""Run one exact-geometry 2D Q2/DPC1 Zhang Table 5 diagnostic.

The driver exercises the public AgentFEM workflow on the published unit-cell
geometry and interpolation family. It recovers the homogenized current-state
algorithmic tangent from the converged Jacobian, but deliberately omits the
remaining convergence evidence, so its assessment cannot promote the benchmark
even when individual observable comparisons pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from mpi4py import MPI
import numpy as np

import agentfem
from agentfem import constraints, models, operators, results, solvers, steps, studies
from agentfem.provenance import content_fingerprint, runtime_manifest

from zhang_2021_periodic_composite_fixture import (
    TABLE5,
    assess_table5,
    zhang_2021_plane_strain_composite,
)


def _file_identity(path: Path) -> dict[str, str]:
    selected = path.resolve()
    return {
        "name": selected.name,
        "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
    }


def _benchmark_implementation_identity() -> dict[str, object]:
    return {
        "schema": "agentfem.benchmark-implementation-identity.v1",
        "files": (
            _file_identity(Path(__file__)),
            _file_identity(
                Path(zhang_2021_plane_strain_composite.__code__.co_filename)
            ),
        ),
    }


def _discretization_identity(fixture, periodicity) -> dict[str, object]:
    record = {
        "schema": "agentfem.external-benchmark-discretization.v1",
        "mesh": operators.mesh_executable_identity(fixture.domain),
        "cell_tags": operators.meshtags_executable_identity(
            fixture.domain,
            fixture.cell_tags,
        ),
        "facet_tags": operators.meshtags_executable_identity(
            fixture.domain,
            fixture.facet_tags,
        ),
        "periodic_constraint": periodicity.scientific_identity(),
    }
    return record | {"fingerprint": content_fingerprint(record)}


def _require_checkout_runtime() -> None:
    expected = (Path(__file__).resolve().parents[1] / "src" / "agentfem").resolve()
    imported = Path(agentfem.__file__).resolve().parent
    if imported != expected:
        raise RuntimeError(
            "Zhang benchmark driver imported AgentFEM from a different checkout: "
            f"expected {expected}, received {imported}. Run with "
            f"PYTHONPATH={expected.parent} so evidence cannot cross worktrees."
        )


def main() -> int:
    _require_checkout_runtime()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-size", type=float, default=0.20)
    parser.add_argument(
        "--cell-repetitions",
        type=int,
        nargs=2,
        default=(1, 1),
        metavar=("NX", "NY"),
        help="Replicate the published unit cell into an NX by NY supercell.",
    )
    parser.add_argument("--quadrature-degree", type=int, default=4)
    parser.add_argument(
        "--macro-gradient",
        type=float,
        nargs=4,
        metavar=("F11", "F21", "F12", "F22"),
        help="Override the final 2D macro gradient in published column-major order.",
    )
    parser.add_argument(
        "--penultimate-gradient",
        type=float,
        nargs=4,
        metavar=("F11", "F21", "F12", "F22"),
        help=(
            "Add one explicit pre-final path state in published column-major "
            "order. This is used by the fixed-old-state tangent oracle."
        ),
    )
    parser.add_argument("--penultimate-coordinate", type=float)
    parser.add_argument("--initial-increment", type=float, default=0.05)
    parser.add_argument("--minimum-increment", type=float, default=1.0e-4)
    parser.add_argument("--maximum-increment", type=float, default=0.10)
    parser.add_argument("--maximum-inelastic-increment", type=float, default=0.05)
    parser.add_argument("--max-increments", type=int, default=80)
    parser.add_argument("--max-cutbacks", type=int, default=10)
    parser.add_argument(
        "--increments",
        type=int,
        help=(
            "Use an exact uniform load path with this many increments. "
            "When supplied, automatic-incrementation options are ignored."
        ),
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show the human progress stream (quiet by default for diagnostics).",
    )
    parser.add_argument(
        "--skip-tangent",
        action="store_true",
        help=(
            "Skip the four condensed macro-tangent solves. This is intended "
            "for mesh, increment, and quadrature diagnostics; the resulting "
            "candidate remains incomplete for benchmark promotion."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.quadrature_degree < 2:
        parser.error("--quadrature-degree must be at least 2")
    if arguments.max_increments <= 0 or arguments.max_cutbacks < 0:
        parser.error("--max-increments must be positive and --max-cutbacks nonnegative")
    if arguments.increments is not None and arguments.increments <= 0:
        parser.error("--increments must be positive")
    if any(value <= 0 for value in arguments.cell_repetitions):
        parser.error("--cell-repetitions values must be positive")
    if (arguments.penultimate_gradient is None) != (
        arguments.penultimate_coordinate is None
    ):
        parser.error(
            "--penultimate-gradient and --penultimate-coordinate must be supplied together"
        )
    if arguments.penultimate_coordinate is not None and not (
        0.0 < arguments.penultimate_coordinate < 1.0
    ):
        parser.error("--penultimate-coordinate must lie strictly inside (0, 1)")

    final_gradient = (
        None
        if arguments.macro_gradient is None
        else np.asarray(arguments.macro_gradient, dtype=float).reshape(
            (2, 2), order="F"
        )
    )

    comm = MPI.COMM_WORLD
    runtime_at_start = runtime_manifest()
    benchmark_at_start = _benchmark_implementation_identity()
    total_started = time.perf_counter()
    fixture_started = total_started
    fixture = zhang_2021_plane_strain_composite(
        comm,
        mesh_size=arguments.mesh_size,
        macro_deformation_gradient=final_gradient,
        cell_repetitions=tuple(arguments.cell_repetitions),
    )
    fixture_seconds = time.perf_counter() - fixture_started
    build_started = time.perf_counter()
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=2,
            assumption="plane_strain",
        ),
        mesh=fixture.domain,
        name="zhang_2021_table5_plane_strain_diagnostic",
    )
    target = model.field(fixture.mixed_field())
    matrix_region, inclusion_region = fixture.regions()
    matrix, inclusion = fixture.materials()
    model.material(matrix, region=matrix_region)
    model.material(inclusion, region=inclusion_region)
    macro_path = None
    if arguments.penultimate_gradient is not None:
        penultimate = np.asarray(
            arguments.penultimate_gradient,
            dtype=float,
        ).reshape((2, 2), order="F")
        macro_path = constraints.deformation_gradient_path(
            (0.0, arguments.penultimate_coordinate, 1.0),
            (np.eye(2), penultimate, fixture.deformation_gradient),
            name="fixed_old_state_macro_tangent_path",
        )
    periodicity = model.constraint(
        fixture.constraint(target, deformation_gradient_path=macro_path)
    )
    discretization_at_start = _discretization_identity(fixture, periodicity)
    output = results.output_plan(
        arguments.output,
        field=results.field_output(
            "U",
            "S",
            "MISES",
            intervals=1,
            configuration="reference",
        ),
        requests=(results.periodic_cell_history(periodicity),),
        presentation=None,
        basename="zhang_2021_table5_plane_strain",
    )
    if arguments.increments is None:
        incrementation = steps.automatic(
            initial=arguments.initial_increment,
            minimum=arguments.minimum_increment,
            maximum=arguments.maximum_increment,
            max_increments=arguments.max_increments,
            max_cutbacks=arguments.max_cutbacks,
            maximum_inelastic_increment=(arguments.maximum_inelastic_increment),
        )
    else:
        incrementation = steps.fixed(arguments.increments)
    problem = model.step(
        target=target,
        constraints=periodicity,
        quadrature_degree=arguments.quadrature_degree,
        incrementation=incrementation,
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-10,
            maximum_iterations=35,
            line_search="backtracking",
        ),
        output=output,
        progress=arguments.progress,
    )
    build_seconds = time.perf_counter() - build_started
    solve_started = time.perf_counter()
    simulation = problem.solve_result()
    solve_seconds = time.perf_counter() - solve_started
    recorder = problem.accepted_history_recorders["homogenized_history"]
    frame = recorder.frames[-1]
    if frame.elastic_energy_density is None:
        raise RuntimeError(
            "The mixed provider did not expose accepted condensed ELENER."
        )
    postprocess_started = time.perf_counter()
    energy = results.mixed_j2_elastic_energy_diagnostics(
        deformation_gradient=problem.state_transaction.deformation_gradient,
        pressure=problem.state_transaction.mixed_pressure,
        inverse_bulk_modulus=problem.state_transaction.inverse_bulk_modulus,
        condensed_elastic_energy_density=(
            problem.response.stored_energy_density_components["ELENER"]
        ),
        reference_volume=periodicity.reference_cell_volume,
    )
    energy_seconds = time.perf_counter() - postprocess_started
    tangent_started = time.perf_counter()
    tangent = (
        None
        if arguments.skip_tangent
        else results.homogenized_algorithmic_tangent(problem, periodicity)
    )
    tangent_seconds = time.perf_counter() - tangent_started
    condensed_scale = max(abs(frame.elastic_energy_density), 1.0)
    if (
        abs(frame.elastic_energy_density - energy.condensed_elastic_energy_density)
        > 256.0 * math.ulp(1.0) * condensed_scale
    ):
        raise RuntimeError(
            "The homogenized-history and mixed-energy condensed ELENER "
            "channels disagree."
        )
    assessment = assess_table5(
        first_piola=frame.first_piola_stress,
        elastic_energy_density=energy.primal_elastic_energy_density,
        elastic_energy_semantics="primal_hencky_elastic_energy",
        effective_tangent=None if tangent is None else tangent.values,
        convergence_evidence={
            "load_increment_path_converged": False,
            "mesh_converged": False,
            "plane_strain_formulation_converged": False,
            "periodic_cell_size_invariant": False,
            "serial_mpi_equivalent": False,
            "restart_equivalent": False,
        },
    )
    last_checks = problem.last_solve_info.increments[-1].checks
    performance = results.performance_evidence(
        stages={
            "total": time.perf_counter() - total_started,
            "fixture": fixture_seconds,
            "model_build": build_seconds,
            "solve": solve_seconds,
            "energy_postprocess": energy_seconds,
            "macro_tangent": tangent_seconds,
        },
        solution=problem.solution,
        source=problem,
        scope="external_benchmark_candidate",
    )
    runtime_at_end = runtime_manifest()
    benchmark_at_end = _benchmark_implementation_identity()
    discretization_at_end = _discretization_identity(fixture, periodicity)
    if runtime_at_end["identity"] != runtime_at_start["identity"]:
        raise RuntimeError(
            "AgentFEM runtime identity changed while the benchmark was running; "
            "the candidate is not reproducible."
        )
    if benchmark_at_end != benchmark_at_start:
        raise RuntimeError(
            "Benchmark implementation changed while the solve was running; "
            "the candidate is not reproducible."
        )
    if discretization_at_end != discretization_at_start:
        raise RuntimeError(
            "Mesh, region tags, boundary tags, or periodic equations changed "
            "while the benchmark was running; the candidate is not reproducible."
        )
    assessment.update(
        {
            "candidate_schema": "agentfem.external-benchmark-candidate.v2",
            "result_status": simulation.status,
            "formulation": "2D_plane_strain_Q2_DPC1",
            "mesh_size": float(arguments.mesh_size),
            "cell_repetitions": list(fixture.cell_repetitions),
            "reference_cell_area": fixture.reference_cell_area,
            "global_cells": int(fixture.domain.topology.index_map(2).size_global),
            "published_q9_element_count": TABLE5.published_q9_element_count,
            "element_count_fraction_of_published": (
                fixture.element_count / TABLE5.published_q9_element_count
            ),
            "minimum_scaled_jacobian": fixture.minimum_scaled_jacobian,
            "quadrature_degree": int(arguments.quadrature_degree),
            "incrementation": incrementation.summary(),
            "accepted_increment_path": [
                item.as_dict() for item in problem.accepted_increments
            ],
            "accepted_increments": len(problem.accepted_increments),
            "attempted_increments": len(problem.attempted_increments),
            "periodic_pairing_error": fixture.periodic_pairing_error,
            "periodic_equation_mismatch": periodicity.mismatch(),
            "maximum_hill_mandel_relative_error": max(
                item.relative_error for item in recorder.hill_mandel
            ),
            "pressure_block_residual_norm": last_checks["pressure_block_residual_norm"],
            "maximum_quadrature_pressure_constraint_defect": last_checks[
                "maximum_quadrature_pressure_projection_defect"
            ],
            "mixed_elastic_energy_diagnostics": energy.as_dict(),
            "homogenized_algorithmic_tangent": (
                None if tangent is None else tangent.as_dict()
            ),
            "runtime": {
                "agentfem_version": agentfem.__version__,
                "agentfem_import_path": str(Path(agentfem.__file__).resolve()),
                "manifest": runtime_at_start,
            },
            "benchmark_implementation": benchmark_at_start,
            "discretization_identity": discretization_at_start,
            "identity_stable_during_run": True,
            "candidate": {
                "formulation": "2D_plane_strain_Q2_DPC1",
                "mesh_size": float(arguments.mesh_size),
                "cell_repetitions": list(fixture.cell_repetitions),
                "reference_cell_area": fixture.reference_cell_area,
                "global_cells": int(fixture.domain.topology.index_map(2).size_global),
                "discretization_fingerprint": discretization_at_start["fingerprint"],
                "quadrature_degree": int(arguments.quadrature_degree),
                "accepted_increments": len(problem.accepted_increments),
                "requested_fixed_increments": arguments.increments,
                "mpi_ranks": int(comm.size),
                "macro_tangent_requested": not arguments.skip_tangent,
                "macroscopic_deformation_gradient": (
                    fixture.deformation_gradient.tolist()
                ),
                "deformation_gradient_path": (
                    None if macro_path is None else macro_path.summary()
                ),
            },
            "performance": performance.as_dict(),
            "increment_performance": results.increment_performance(
                problem.accepted_increments
            ),
            "stored_energy_scope": (
                "primal Hencky elastic energy reconstructed from the "
                "provider-owned condensed ELENER channel; HARDENER and "
                "PDENER are excluded"
            ),
        }
    )
    if comm.rank == 0:
        arguments.output.mkdir(parents=True, exist_ok=True)
        assessment_path = (
            arguments.output / "zhang_2021_table5_plane_strain_assessment.json"
        )
        temporary_path = assessment_path.with_suffix(assessment_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(assessment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(assessment_path)
        tangent_error = assessment["effective_tangent_relative_frobenius_error"]
        tangent_summary = (
            "not-computed" if tangent_error is None else f"{100.0 * tangent_error:.3f}%"
        )
        print(
            "Zhang 2021 Table 5 diagnostic "
            f"| {assessment['status'].upper()} "
            f"| cells={assessment['global_cells']} "
            f"| increments={assessment['accepted_increments']} "
            f"| P_error={100.0 * assessment['first_piola_relative_l2_error']:.3f}% "
            f"| energy_error={100.0 * assessment['elastic_energy_relative_error']:.3f}% "
            f"| tangent_error={tangent_summary}"
        )
        print(f"Evidence: {assessment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
