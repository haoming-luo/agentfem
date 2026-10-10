# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded finite-strain plastic Hex8 extension through the ordinary workflow.

Run in an installed AgentFEM environment, serially or with agentfem mpi-run.
The supplied spectral ceiling belongs to this mesh/material/loading example;
it is not an automatic stable-time estimate for arbitrary forming problems.
"""

import argparse
from pathlib import Path

import numpy as np

from agentfem import amplitudes, constitutive, elements, fields, mesh, models, studies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/finite_hex_extension")
    )
    parser.add_argument("--restart", type=Path)
    args = parser.parse_args()
    domain = mesh.cuboid((0, 0, 0), (1, 0.2, 0.2), (8, 2, 2), cell_type="hexahedron")
    model = models.create(
        study=studies.dynamic_solid(dimension=3),
        mesh=domain,
        name="finite_hex_extension",
    )
    u = model.field(fields.displacement(domain))
    model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=100, poisson=0.3, yield_stress=1, hardening_modulus=5, density=2
        )
    )
    model.fix(u, on=lambda x: np.ones(x.shape[1], dtype=bool), components=(1, 2))
    model.fix(u, on=lambda x: np.isclose(x[0], 0), components=0)
    model.fix(
        u,
        on=lambda x: np.isclose(x[0], 1),
        components=0,
        value=amplitudes.Amplitude(
            "quadratic_extension",
            lambda t: 10 * t * t,
            metadata={"quadratic_coefficient": 10.0},
        ),
    )
    step = model.step(
        target=u,
        element_policy=elements.uniform_strain_hex8(
            hourglass_modulus=40, hourglass_scale=0.1, kinematics="finite_strain"
        ),
        omega_squared_bound=1e8,
        maximum_negative_growth_per_increment=0.1,
        dt=1e-4,
        steps=1000,
        save_every=100,
        history_every=100,
        progress=False,
    )
    if args.restart is not None:
        step.load_checkpoint(args.restart)
    args.output.mkdir(parents=True, exist_ok=True)
    result = step.solve_result(
        field_variables=("S", "F", "PEEQ", "SENER", "PDENER"),
        output=args.output / "fields.xdmf",
    )
    step.save_checkpoint(args.output / "checkpoint")
    result.write_manifest(args.output / "result.json", comm=domain.comm)
    row = step.history_records[-1]
    if domain.comm.rank == 0:
        print(
            f"Completed {step.completed_steps} increments; bounded finite-strain Hex8."
        )
        print(f"Material dissipation: {row['material_dissipation']:.6g}")
        print(f"Artificial hourglass energy: {row['hourglass_energy']:.6g}")
        print(
            f"Relative work/energy residual: {row['relative_energy_balance_error']:.3g}"
        )
        print(
            "Completion is not independent verification or industrial-forming validation."
        )
        print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
