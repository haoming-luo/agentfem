# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Experimental compression/unloading/recontact by a translating rigid plane.

Coherent units: metre, pascal, newton. A deliberately soft homogeneous block
with restrained lateral strain supplies a small, independently checkable
numerical example, not an industrial material or general forming model.

Run --until 0.5, then --restart <output>/checkpoint.npz --output <new-folder>.
The tool follows the Step amplitude; the material-side pressure trace stays
fixed. No rotation, friction, MPC, changing trace or MPI augmentation is admitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import (
    amplitudes, boundary_models, constitutive, fields, mesh, models,
    results, solvers, steps, studies,
)


def make_step(*, cells=2, increments=32, acceleration="none"):
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This experimental normal-augmentation path is serial only.")
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, cells, cells, cells)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain,
        name="translating_augmented_plane",
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.), name="right")
    exterior = mesh.boundary(
        domain, lambda x: np.ones(x.shape[1], dtype=bool), name="exterior",
    )
    model.fix(displacement, on=left, component=0, value=0.)
    for component in (1, 2):
        model.fix(displacement, on=exterior, component=component, value=0.)
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000., poisson=0., yield_stress=2., hardening_modulus=100.,
    ))
    tool = boundary_models.rigid_body(
        boundary_models.rigid_plane(point=(1.005, 0., 0.), normal=(-1., 0., 0.)),
        reference_point=(1.005, 0., 0.),
        motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
            boundary_models.prescribed_rigid_motion(
                translation=(-.02, 0., 0.), reference_point=(1.005, 0., 0.),
            ),
            start_time=0., end_time=1.,
        ),
    )
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        right, tool, penalty=5000., initial_penetration_limit=0., name="press",
    ))
    problem = model.step(
        target=displacement, material=material,
        amplitude=amplitudes.tabular((0., .5, .75, 1.), (0., 1., 0., 1.)),
        incrementation=steps.fixed(increments),
        normal_augmentation=steps.normal_augmentation(
            penetration_tolerance=1e-9, complementarity_tolerance=1e-8,
            maximum_augmentations=120, length_unit="m", pressure_unit="Pa",
            acceleration=acceleration,
        ),
        solver_options=solvers.newton(
            relative_tolerance=1e-8, absolute_tolerance=1e-9,
            maximum_iterations=25, line_search="backtracking",
        ),
        progress=False,
    )
    return displacement, problem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/translating_augmented_plane"))
    parser.add_argument("--cells", type=int, default=2)
    parser.add_argument("--increments", type=int, default=32)
    parser.add_argument("--until", type=float, default=1.)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--acceleration", choices=("none", "aitken"), default="none")
    args = parser.parse_args()
    if args.cells < 1 or args.increments < 4 or args.increments % 4:
        parser.error("Use positive cells and increments divisible by four to preserve reversals.")
    displacement, problem = make_step(
        cells=args.cells, increments=args.increments, acceleration=args.acceleration,
    )
    if args.restart is not None:
        problem.load_checkpoint(args.restart)
    if not problem.accepted_load_factor < args.until <= 1.:
        parser.error("--until must exceed the accepted coordinate and be <= 1.")
    result = problem.solve_result() if args.until == 1. else None
    if result is None:
        problem.solve(until=args.until)
    args.output.mkdir(parents=True, exist_ok=True)
    with results.UnifiedXDMFTimeSeries(args.output / "fields.xdmf") as writer:
        writer.write_fields(args.until, displacement.value)
    problem.save_checkpoint(args.output / "checkpoint")
    results.export_accepted_contact_points(
        problem, args.output / "contact", result=result, vtk_view=True,
        units={"length": "m", "pressure": "Pa", "force": "N"},
    )
    if result is not None:
        result.write_manifest(args.output / "result.json")
    energy = problem.energy_history[-1]
    print(f"Accepted increments: {len(problem.accepted_increments)}")
    print(f"Outer multiplier updates: {sum(i.normal_augmentations for i in problem.accepted_increments)}")
    print(f"Right-face displacement: {results.probe(displacement, at=(1., .5, .5))[0]:.6g} m")
    print(f"Tool work: {energy.tool_work:.6g} J")
    print(f"Numerical relative-gap work: {energy.normal_contact_work:.6g} J")
    print(f"Raw energy residual: {energy.mechanical_energy_residual:.6g} J")
    print(f"Contact-aware residual: {energy.mechanical_energy_residual + energy.normal_contact_work:.6g} J")
    print("Experimental translation-only scope; available energy is not automatic verification.")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
