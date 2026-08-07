"""Two-process/write and one-process/read checkpoint acceptance driver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import constitutive, fields, mesh, models, studies


def heat_step():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.25),
        (4, 2),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    model = models.create(
        study=studies.transient_heat_transfer(dimension=2),
        mesh=domain,
    )
    temperature = model.field(fields.temperature(domain, value=400.0))
    model.material(
        constitutive.thermoelastic(
            young=1.0e9,
            poisson=0.3,
            density=1000.0,
            thermal_expansion=1.0e-5,
            conductivity=10.0,
            specific_heat=500.0,
            reference_temperature=300.0,
        )
    )
    model.convection(
        on=mesh.face(domain, axis="x", value=1.0, name="right", tag=1),
        coefficient=25.0,
        ambient_temperature=300.0,
    )
    return model.step(
        target=temperature,
        dt=0.5,
        steps=3,
        progress=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "read"))
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    if arguments.mode == "write":
        step = heat_step()
        step.run(until_step=1)
        path = step.save_checkpoint(arguments.checkpoint, portable=True)
        if MPI.COMM_WORLD.rank == 0:
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["rank_count"] == MPI.COMM_WORLD.size
            assert payload["portable"] is True
        return

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Portable checkpoint read acceptance runs with one rank.")
    reference = heat_step()
    reference.run()
    restarted = heat_step()
    restarted.load_checkpoint(arguments.checkpoint)
    restarted.run()
    np.testing.assert_allclose(
        restarted.current.x.array,
        reference.current.x.array,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert restarted.completed_steps == reference.completed_steps == 3


if __name__ == "__main__":
    main()
