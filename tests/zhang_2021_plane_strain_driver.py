"""Run one exact-geometry 2D Q2/DPC1 Zhang Table 5 diagnostic.

The driver exercises the public AgentFEM workflow on the published unit-cell
geometry and interpolation family. It recovers the homogenized current-state
algorithmic tangent from the converged Jacobian, but deliberately omits the
remaining convergence evidence, so its assessment cannot promote the benchmark
even when individual observable comparisons pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from mpi4py import MPI

from agentfem import models, results, solvers, steps, studies

from zhang_2021_periodic_composite_fixture import (
    assess_table5,
    zhang_2021_plane_strain_composite,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-size", type=float, default=0.20)
    parser.add_argument("--quadrature-degree", type=int, default=4)
    parser.add_argument("--initial-increment", type=float, default=0.05)
    parser.add_argument("--minimum-increment", type=float, default=1.0e-4)
    parser.add_argument("--maximum-increment", type=float, default=0.10)
    parser.add_argument("--maximum-inelastic-increment", type=float, default=0.05)
    parser.add_argument("--max-increments", type=int, default=80)
    parser.add_argument("--max-cutbacks", type=int, default=10)
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show the human progress stream (quiet by default for diagnostics).",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.quadrature_degree < 2:
        parser.error("--quadrature-degree must be at least 2")
    if arguments.max_increments <= 0 or arguments.max_cutbacks < 0:
        parser.error("--max-increments must be positive and --max-cutbacks nonnegative")

    comm = MPI.COMM_WORLD
    fixture = zhang_2021_plane_strain_composite(
        comm,
        mesh_size=arguments.mesh_size,
    )
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
    periodicity = model.constraint(fixture.constraint(target))
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
    problem = model.step(
        target=target,
        constraints=periodicity,
        quadrature_degree=arguments.quadrature_degree,
        incrementation=steps.automatic(
            initial=arguments.initial_increment,
            minimum=arguments.minimum_increment,
            maximum=arguments.maximum_increment,
            max_increments=arguments.max_increments,
            max_cutbacks=arguments.max_cutbacks,
            maximum_inelastic_increment=(
                arguments.maximum_inelastic_increment
            ),
        ),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-10,
            maximum_iterations=35,
            line_search="backtracking",
        ),
        output=output,
        progress=arguments.progress,
    )
    simulation = problem.solve_result()
    recorder = problem.accepted_history_recorders["homogenized_history"]
    frame = recorder.frames[-1]
    if frame.elastic_energy_density is None:
        raise RuntimeError(
            "The mixed provider did not expose accepted condensed ELENER."
        )
    energy = results.mixed_j2_elastic_energy_diagnostics(
        deformation_gradient=problem.state_transaction.deformation_gradient,
        pressure=problem.state_transaction.mixed_pressure,
        inverse_bulk_modulus=problem.state_transaction.inverse_bulk_modulus,
        condensed_elastic_energy_density=(
            problem.response.stored_energy_density_components["ELENER"]
        ),
        reference_volume=periodicity.reference_cell_volume,
    )
    tangent = results.homogenized_algorithmic_tangent(problem, periodicity)
    condensed_scale = max(abs(frame.elastic_energy_density), 1.0)
    if (
        abs(
            frame.elastic_energy_density
            - energy.condensed_elastic_energy_density
        )
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
        effective_tangent=tangent.values,
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
    assessment.update(
        {
            "result_status": simulation.status,
            "formulation": "2D_plane_strain_Q2_DPC1",
            "mesh_size": float(arguments.mesh_size),
            "global_cells": int(
                fixture.domain.topology.index_map(2).size_global
            ),
            "quadrature_degree": int(arguments.quadrature_degree),
            "accepted_increments": len(problem.accepted_increments),
            "attempted_increments": len(problem.attempted_increments),
            "periodic_pairing_error": fixture.periodic_pairing_error,
            "periodic_equation_mismatch": periodicity.mismatch(),
            "maximum_hill_mandel_relative_error": max(
                item.relative_error for item in recorder.hill_mandel
            ),
            "pressure_block_residual_norm": last_checks[
                "pressure_block_residual_norm"
            ],
            "maximum_quadrature_pressure_constraint_defect": last_checks[
                "maximum_quadrature_pressure_projection_defect"
            ],
            "mixed_elastic_energy_diagnostics": energy.as_dict(),
            "homogenized_algorithmic_tangent": tangent.as_dict(),
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
        assessment_path.write_text(
            json.dumps(assessment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
