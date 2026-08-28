"""Two-rank smoke for a true-geometry finite-strain J2 periodic RVE."""

from __future__ import annotations

import json

import numpy as np
from mpi4py import MPI

from agentfem import (
    constitutive,
    fields,
    models,
    results,
    solvers,
    steps,
    studies,
)

from periodic_void_fixture import periodic_spherical_void_cell


def main() -> None:
    comm = MPI.COMM_WORLD
    fixture = periodic_spherical_void_cell(
        comm,
        mesh_size=0.25,
        stretch=1.004,
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="parallel_finite_strain_j2_real_void_smoke",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=200_000.0,
            poisson=0.3,
            yield_stress=200.0,
            hardening_modulus=2_000.0,
        )
    )
    periodicity = model.constraint(fixture.constraint(displacement))
    step = model.step(
        target=displacement,
        material=material,
        constraints=periodicity,
        incrementation=steps.fixed(2),
        solver_options=solvers.newton(
            relative_tolerance=2.0e-7,
            absolute_tolerance=1.0e-8,
            maximum_iterations=25,
            line_search="backtracking",
        ),
        quadrature_degree=1,
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
    local_minimum_j = min(
        float(np.min(np.linalg.det(snapshot.fields["F"].owned_values)))
        for snapshot in step.snapshots
    )
    minimum_j = float(comm.allreduce(local_minimum_j, op=MPI.MIN))
    maximum_peeq = (
        step.response.state.committed["equivalent_plastic_strain"].global_max()
    )
    evidence = {
        "rank_count": int(comm.size),
        "status": result.status,
        "converged": bool(step.last_solve_info.converged),
        "global_cells": int(
            fixture.domain.topology.index_map(3).size_global
        ),
        "periodic_equation_mismatch": float(periodicity.mismatch()),
        "minimum_j": minimum_j,
        "maximum_peeq": float(maximum_peeq),
        "solid_reference_fraction": float(
            frames[-1].solid_reference_fraction
        ),
        "macro_first_piola_11": float(frames[-1].first_piola_stress[0, 0]),
        "maximum_hill_mandel_relative_error": float(
            max(item.relative_error for item in hill_mandel)
        ),
    }
    if evidence["rank_count"] != 2:
        raise RuntimeError("This driver is a two-rank acceptance test.")
    if evidence["status"] != "completed" or not evidence["converged"]:
        raise RuntimeError("The distributed real-void J2 step did not complete.")
    if evidence["periodic_equation_mismatch"] >= 1.0e-10:
        raise RuntimeError("Distributed periodic equations are not satisfied.")
    if evidence["minimum_j"] <= 0.99:
        raise RuntimeError("The distributed RVE has an inadmissible local J.")
    if evidence["maximum_peeq"] <= 1.0e-4:
        raise RuntimeError("The distributed RVE did not enter plastic flow.")
    if evidence["maximum_hill_mandel_relative_error"] >= 1.0e-8:
        raise RuntimeError("The distributed RVE violates Hill--Mandel work.")
    if comm.rank == 0:
        print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
