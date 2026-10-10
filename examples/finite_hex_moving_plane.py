# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Serial finite-J2 compression by a prescribed plane through model.step.

This small reference-surface penalty demonstration is not industrial forming
validation. Its caller bulk spectral ceiling is specific to this test.
"""

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import boundary_models, constitutive, elements, fields, mesh, models, studies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/finite_hex_moving_plane"))
    parser.add_argument("--restart", type=Path)
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This bounded finite-material/contact combination is serial only.")
    domain = mesh.cuboid((0, 0, 0), (1, 1, 1), (1, 1, 1), cell_type="hexahedron")
    model = models.create(study=studies.dynamic_solid(dimension=3), mesh=domain)
    u = model.field(fields.displacement(domain))
    model.material(constitutive.finite_strain_j2_logarithmic(
        young=100, poisson=0.3, yield_stress=0.02, hardening_modulus=5, density=2,
    ))
    model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(1, 2))
    slave = mesh.boundary(domain, lambda x: np.isclose(x[0], 1), name="loaded_end")
    tool = boundary_models.rigid_body(
        boundary_models.rigid_plane(point=(1, 0, 0), normal=(-1, 0, 0)),
        motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
            boundary_models.prescribed_rigid_motion(translation=(-0.01, 0, 0)),
            end_time=0.1,
        ), name="plane_tool",
    )
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        slave, tool, penalty=200, name="plane_contact",
    ))
    step = model.step(
        target=u, element_policy=elements.uniform_strain_hex8(
            hourglass_modulus=40, hourglass_scale=0.1, kinematics="finite_strain",
        ), omega_squared_bound=1e7, maximum_negative_growth_per_increment=0.1,
        dt=1e-4, steps=1000, history_every=100, save_every=100, progress=False,
    )
    if args.restart is not None:
        step.load_checkpoint(args.restart)
    args.output.mkdir(parents=True, exist_ok=True)
    result = step.solve_result(output=args.output / "fields.xdmf",
                               field_variables=("S", "F", "PEEQ", "PDENER"))
    step.save_checkpoint(args.output / "checkpoint")
    result.write_manifest(args.output / "result.json")
    row = step.history_records[-1]
    print(f"Tool work: {row['contact_motion_work']:.6g}")
    print(f"Material dissipation: {row['material_dissipation']:.6g}")
    print(f"Relative energy residual: {row['relative_energy_balance_error']:.3g}")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
