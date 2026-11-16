# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Short-pulse 3D block: model, nodes, explicit process and independent history.

A reproducible integration example, not a replacement for the missing Abaqus
blade mesh/reference and not a C3D10M claim. Units: mm, N, tonne, s.
"""

import argparse
from pathlib import Path
import json
import numpy as np
from mpi4py import MPI
from agentfem import (
    mesh,
    models,
    studies,
    fields,
    constitutive,
    elements,
    amplitudes,
    results,
    project,
)


def solve_pulse(*, output, element="tet15_nodal"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    comm = MPI.COMM_WORLD
    geometry = mesh.cuboid(
        (0, 0, 0), (50, 5, 5), (8, 2, 2), comm=comm, cell_type="tetrahedron"
    )
    model = models.create(
        study=studies.dynamic_solid(dimension=3, method="explicit"), mesh=geometry
    )
    u = model.field(
        fields.displacement(geometry, degree=2)
        if element == "tet10_hrz"
        else fields.displacement(geometry, element=elements.tet15_mass_lumped())
    )
    model.material(
        constitutive.isotropic_elastic(young=210000, poisson=0.3, density=7.8e-9)
    )
    model.clamp(
        u, on=mesh.boundary(geometry, lambda x: np.isclose(x[0], 0), tag=1, name="root")
    )
    model.nodal_force(
        (0, -1230, 0),
        at=(50, 5, 5),
        amplitude=amplitudes.half_sine(25e-6),
        name="impact",
    )
    mass = model.lumped_mass(u, method=element)
    step = model.step(
        target=u,
        mass=mass,
        dt="auto",
        steps=1,
        save_every=25,
        history_every=1,
        progress=False,
    )
    # End precisely at the declared duration without exceeding the automatic limit.
    step.steps = int(np.ceil(50e-6 / step.dt))
    step.dt = 50e-6 / step.steps
    station = results.dynamic_history(
        "sensor", at=(40, 2.5, 2.5), length_unit="mm", time_unit="s"
    )
    result = step.solve_result(history=station, output=output / "fields.xdmf")
    if comm.rank == 0:
        results.write_dynamic_history(
            result,
            output / "sensor.csv",
            metadata={"units": "mm-N-tonne-s", "element": element},
        )
        (output / "run.json").write_text(
            json.dumps(
                {
                    "element": element,
                    "dt": step.dt,
                    "increments": step.steps,
                    "stability": step.stability.summary(),
                    "model": model.summary(),
                    "final_history": step.history_records[-1],
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        print(output.resolve(), flush=True)

    return result


def main():
    run = project.current_run(project_root=Path(__file__).parent)
    result = solve_pulse(output=run.output_directory)
    run.publish(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--element", choices=["tet10_hrz", "tet15_nodal"], default="tet15_nodal"
    )
    args = parser.parse_args()
    if args.output is None and args.element == "tet15_nodal":
        main()
    else:
        solve_pulse(
            output=args.output or Path("outputs/explicit_nodal_pulse"),
            element=args.element,
        )
