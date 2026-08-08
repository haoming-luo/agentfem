"""Two-rank/write and one-rank/read split-interface Explicit restart."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import (
    amplitudes,
    constitutive,
    constraints,
    fields,
    fracture,
    interfaces,
    models,
    studies,
)


def _split_strip():
    coordinates = np.asarray(
        [(x / 3.0, y / 2.0) for y in range(3) for x in range(4)],
        dtype=float,
    )
    cells = np.asarray(
        [
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [4, 5, 9, 8],
            [5, 6, 10, 9],
            [6, 7, 11, 10],
        ],
        dtype=int,
    )
    return interfaces.split_conforming_line_interface(
        coordinates,
        cells,
        [[4, 5], [5, 6], [6, 7]],
        positive_cells=[3, 4, 5],
    )


def _step():
    split = _split_strip()
    domain = interfaces.create_dolfinx_split_mesh(split, comm=MPI.COMM_WORLD)
    model = models.create(
        study=studies.dynamic_solid(
            dimension=2,
            assumption="plane_strain",
            method="explicit",
        ),
        mesh=domain,
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        constitutive.neo_hookean(young=1000.0, poisson=0.25, density=1.0)
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement,
            1,
            on=lambda x: np.isclose(x[1], 0.0),
            value=0.0,
        )
    )
    model.constraint(
        constraints.component_dirichlet(
            displacement,
            0,
            on=lambda x: np.isclose(x[0], 0.0),
            value=0.0,
        )
    )
    model.constraint(
        constraints.time_dependent_component_dirichlet(
            displacement,
            1,
            on=lambda x: np.isclose(x[1], 1.0),
            amplitude=amplitudes.ramp(
                0.0,
                2.0e-4,
                start_time=0.0,
                end_time=2.0e-5,
            ),
        )
    )
    law = interfaces.bilinear_cohesive(
        strength=10.0,
        fracture_energy=2.0,
        initial_stiffness=1000.0,
    )
    cohesive = fracture.mode_i_cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
    )
    step = model.finite_strain_explicit_dynamics_step(
        target=displacement,
        material=material,
        cohesive_force=cohesive,
        dt=1.0e-5,
        steps=2,
        progress=False,
    )
    return step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "read"))
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    if arguments.mode == "write":
        if MPI.COMM_WORLD.size != 2:
            raise RuntimeError("Cohesive portable write requires two ranks.")
        partial = _step()
        partial.run(until_step=1)
        partial.save_checkpoint(arguments.checkpoint, portable=True)
        return
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Cohesive portable read requires one rank.")
    reference = _step()
    reference.run()
    restarted = _step()
    restarted.load_checkpoint(arguments.checkpoint)
    restarted.run()
    np.testing.assert_allclose(
        restarted.state.u.value.x.array,
        reference.state.u.value.x.array,
        rtol=1.0e-11,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        restarted.residual.cohesive.assembler.state.committed_maximum,
        reference.residual.cohesive.assembler.state.committed_maximum,
        rtol=1.0e-11,
        atol=1.0e-13,
    )
    if restarted.completed_steps != 2 or reference.completed_steps != 2:
        raise RuntimeError("Cohesive portable restart did not complete.")


if __name__ == "__main__":
    main()
