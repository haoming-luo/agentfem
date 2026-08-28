"""Run one diagnostic level of the Zhang et al. (2021) Table 5 benchmark.

This driver can never report ``accepted`` because it deliberately omits the
effective tangent and promotion-level convergence evidence.  Available values
outside the fixed tolerance report ``failed``; otherwise the result remains
``incomplete``.  It is a manual evidence producer, not part of the fast
regression suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpi4py import MPI

from agentfem import fields, models, results, solvers, steps, studies

from zhang_2021_periodic_composite_fixture import (
    assess_table5,
    zhang_2021_periodic_composite,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-size", type=float, default=0.24)
    parser.add_argument("--thickness", type=float, default=0.10)
    parser.add_argument("--increments", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.increments <= 0:
        parser.error("--increments must be positive")

    comm = MPI.COMM_WORLD
    fixture = zhang_2021_periodic_composite(
        comm,
        mesh_size=args.mesh_size,
        thickness=args.thickness,
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="zhang_2021_table5_diagnostic",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    matrix_region, inclusion_region = fixture.regions()
    matrix, inclusion = fixture.materials()
    model.material(matrix, region=matrix_region)
    model.material(inclusion, region=inclusion_region)
    periodicity = model.constraint(fixture.constraint(displacement))
    output = results.output_plan(
        args.output,
        requests=(results.periodic_cell_history(periodicity),),
        presentation=None,
        basename="zhang_2021_table5",
    )
    step = model.step(
        target=displacement,
        constraints=periodicity,
        incrementation=steps.fixed(args.increments),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-7,
            absolute_tolerance=1.0e-9,
            maximum_iterations=30,
            line_search="backtracking",
        ),
        output=output,
        progress=True,
    )
    result = step.solve_result()
    recorder = step.accepted_history_recorders["homogenized_history"]
    frame = recorder.frames[-1]
    if frame.elastic_energy_density is None:
        raise RuntimeError(
            "The finite-strain provider did not expose accepted ELENER. "
            "Table 5 must not be compared against aggregate SENER."
        )
    elastic_energy = frame.elastic_energy_density
    assessment = assess_table5(
        first_piola=frame.first_piola_stress,
        elastic_energy_density=elastic_energy,
        effective_tangent=None,
        convergence_evidence={
            "mesh_converged": False,
            "plane_strain_formulation_converged": False,
            "periodic_cell_size_invariant": False,
            "serial_mpi_equivalent": False,
            "restart_equivalent": False,
        },
    )
    assessment.update(
        {
            "result_status": result.status,
            "mesh_size": float(args.mesh_size),
            "thickness": float(args.thickness),
            "increments": int(args.increments),
            "global_cells": int(fixture.domain.topology.index_map(3).size_global),
            "periodic_pairing_error": fixture.periodic_pairing_error,
            "periodic_equation_mismatch": periodicity.mismatch(),
            "maximum_hill_mandel_relative_error": max(
                item.relative_error for item in recorder.hill_mandel
            ),
            "elastic_energy_density": elastic_energy,
            "stored_energy_scope": (
                "provider-owned ELENER (recoverable Hencky elastic energy); "
                "HARDENER and plastic dissipation are excluded"
            ),
            "formulation_mapping": (
                "published 2D plane strain -> periodic thin 3D P1 tetrahedra, F33=1"
            ),
        }
    )
    if comm.rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "zhang_2021_table5_assessment.json"
        path.write_text(
            json.dumps(assessment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
