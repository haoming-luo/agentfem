# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Experimental serial elastic Hex8 wave with visible artificial energy.

Run from an installed AgentFEM environment. No commercial C3D8R equivalence,
finite rotations, interface damage, or MPI support is implied.
"""

import argparse
from pathlib import Path

import numpy as np

from agentfem import constitutive, elements, fields, mesh, models, studies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/uniform_hex_wave"))
    args = parser.parse_args()
    domain = mesh.cuboid((0, 0, 0), (1, 0.2, 0.2), (16, 2, 2), cell_type="hexahedron")
    model = models.create(
        study=studies.dynamic_solid(dimension=3), mesh=domain, name="uniform_hex_wave"
    )
    u = model.field(fields.displacement(domain))
    model.material(constitutive.isotropic_elastic(young=100, poisson=0, density=2))
    u.value.interpolate(
        lambda x: np.vstack(
            (1e-4 * np.cos(np.pi * x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]))
        )
    )
    formulation = elements.uniform_strain_hex8(
        hourglass_modulus=50, hourglass_scale=0.1
    )
    step = model.step(
        target=u,
        element_policy=formulation,
        dt=1e-4,
        steps=100,
        save_every=10,
        progress=False,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result = step.solve_result(output=args.output / "fields.xdmf")
    result.write_manifest(args.output / "result.json")
    latest = step.history_records[-1]
    print(
        f"Completed {step.completed_steps} steps; experimental small-strain elastic Hex8."
    )
    print(f"Physical mechanical energy: {latest['total_mechanical_energy']:.6g}")
    print(f"Artificial hourglass energy: {latest['hourglass_energy']:.6g}")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
