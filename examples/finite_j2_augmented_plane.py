# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Experimental finite-J2 contact against one fixed frictionless plane.

Units: metre, pascal and newton. This deliberately soft block is a small
numerical example, not a calibrated industrial material or energy benchmark.
Use --until 0.5, then --restart <output>/checkpoint.npz in a new output folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import boundary_models, constitutive, fields, mesh, models, results, solvers, steps, studies


def make_step(*, cells=2, penalty=500., increments=4, minimum_step_length=1/128):
    """Build the declared example, with visible mesh and numerical controls."""

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This experimental normal-augmentation path is serial only.")
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, cells, cells, cells)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain,
        name="finite_j2_augmented_plane",
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.), name="right")
    model.clamp(displacement, on=left)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000., poisson=0., yield_stress=2., hardening_modulus=100.,
    ))
    model.traction((10., 0., 0.), on=right)
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        right,
        boundary_models.rigid_body(boundary_models.rigid_plane(
            point=(1.005, 0., 0.), normal=(-1., 0., 0.),
        )),
        penalty=penalty, initial_penetration_limit=0., name="stop",
    ))
    step = model.step(
        target=displacement, material=material, incrementation=steps.fixed(increments),
        normal_augmentation=steps.normal_augmentation(
            penetration_tolerance=1e-7, complementarity_tolerance=1e-6,
            length_unit="m", pressure_unit="Pa", maximum_augmentations=120,
        ),
        solver_options=solvers.NewtonSolverOptions(
            relative_tolerance=1e-8, absolute_tolerance=1e-9,
            maximum_iterations=25, line_search="backtracking",
            minimum_step_length=minimum_step_length,
        ),
        progress=False,
    )
    return displacement, step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/finite_j2_augmented_plane"))
    parser.add_argument("--until", type=float, default=1.0)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--cells", type=int, default=2)
    parser.add_argument("--penalty", type=float, default=500., help="Explicit AL penalty in Pa/m.")
    parser.add_argument("--minimum-step-length", type=float, default=1/128,
                        help="Explicit Newton line-search floor, not a load increment.")
    args = parser.parse_args()
    displacement, step = make_step(cells=args.cells, penalty=args.penalty,
                                   minimum_step_length=args.minimum_step_length)
    if args.restart is not None:
        step.load_checkpoint(args.restart)
    if not step.accepted_load_factor < args.until <= 1.:
        raise ValueError("--until must exceed the accepted checkpoint factor and be <= 1.")
    result = step.solve_result() if args.until == 1. else None
    if result is None:
        step.solve(until=args.until)
    args.output.mkdir(parents=True, exist_ok=True)
    with results.UnifiedXDMFTimeSeries(args.output / "fields.xdmf") as writer:
        writer.write_fields(args.until, displacement.value)
    step.save_checkpoint(args.output / "checkpoint")
    if result is not None:
        results.export_accepted_contact_points(
            step, args.output / "contact", result=result, vtk_view=True,
            units={"length": "m", "pressure": "Pa", "force": "N"},
        )
        result.write_manifest(args.output / "result.json")
    print(f"Right-face displacement: {results.probe(displacement, at=(1., .5, .5))[0]:.6g} m")
    print(f"Accepted increments: {len(step.accepted_increments)}")
    print(f"Normal augmentations: {sum(i.normal_augmentations for i in step.accepted_increments)}")
    print("Energy audit: unavailable (AL enforcement-work residual is not yet bounded).")
    status = result.status if result is not None else "accepted partial checkpoint"
    print(f"Status: {status}; experimental fixed-plane scope only.")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
