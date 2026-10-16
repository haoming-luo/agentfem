# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Small, serial finite-J2 block pressed by a moving analytical rigid plane.

This is an API and restart example, not validation of curved tooling or forming.
The penalty, mesh and load path are deliberately visible to the user.
Use ``--until 0.5`` to save an accepted partial checkpoint, then pass its
``checkpoint.npz`` to ``--restart`` in a fresh output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import (
    boundary_models,
    constitutive,
    fields,
    mesh,
    models,
    results,
    steps,
    studies,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/finite_j2_plane"))
    parser.add_argument("--restart", type=Path, default=None)
    parser.add_argument("--until", type=float, default=1.0)
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This combined finite-J2/contact example is serial only.")
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1)
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True),
        mesh=domain,
        name="finite_j2_plane_contact",
    )
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="left")
    right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.0), name="right")
    model.clamp(displacement, on=left)
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=1_000.0,
            poisson=0.0,
            yield_stress=2.0,
            hardening_modulus=100.0,
        )
    )
    model.rigid_obstacle_contact(
        on=right,
        penalty=500.0,
        surface=boundary_models.rigid_plane(
            point=(1.0, 0.0, 0.0), normal=(-1.0, 0.0, 0.0)
        ),
        motion=boundary_models.prescribed_rigid_motion(
            translation=(-0.01, 0.0, 0.0)
        ),
    )
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(6),
        progress=False,
    )
    if args.restart is not None:
        step.load_checkpoint(args.restart)
    if not step.accepted_load_factor < args.until <= 1.0:
        raise ValueError("--until must exceed the accepted checkpoint factor and be <= 1")

    args.output.mkdir(parents=True, exist_ok=True)
    result = step.solve_result() if args.until == 1.0 else None
    if result is None:
        step.solve(until=args.until)
    # Keep one ParaView grid per frame. Quadrature state is separate scientific
    # data, not an implicitly smoothed visualization field.
    with results.UnifiedXDMFTimeSeries(args.output / "fields.xdmf") as writer:
        writer.write_fields(args.until, displacement.value)
    step.save_checkpoint(args.output / "checkpoint")
    if result is not None:
        raw = results.write_integration_point_fields(
            result, args.output / "integration-points.h5"
        )
        result.add_artifact("integration_points_hdf5", raw.hdf5)
        result.write_manifest(args.output / "result.json")
    right_displacement = results.probe(displacement, at=(1.0, 0.5, 0.5))[0]
    energy = step.energy_history[-1]
    print(f"Right-face displacement: {right_displacement:.6g}")
    print(f"Tool work: {energy.tool_work:.6g}")
    print(f"Discrete work residual: {energy.mechanical_energy_residual:.3g}")
    print("Status: complete" if result is not None else "Status: accepted partial checkpoint")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
