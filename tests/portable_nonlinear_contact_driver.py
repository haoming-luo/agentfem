"""Cross-rank-count acceptance for ordinary nonlinear contact restart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import constitutive, fields, mesh, models, results, steps, studies


def _left(x):
    return np.isclose(x[0], 0.0)


def _right(x):
    return np.isclose(x[0], 1.0)


def _step():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (8, 4),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )
    model = models.create(
        study=studies.static_solid(
            dimension=2,
            assumption="plane_stress",
            nonlinear=True,
        ),
        mesh=domain,
        name="portable_contact_patch",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        constitutive.elasticity.isotropic_elastic(
            young=1.0e3,
            poisson=0.0,
            density=1.0,
        )
    )
    left = mesh.boundary(domain, _left, name="left", tag=1)
    right = mesh.boundary(domain, _right, name="contact_and_load", tag=2)
    model.clamp(displacement, on=left)
    model.rigid_obstacle_contact(
        on=right,
        penalty=1.0e4,
        normal=(-1.0, 0.0),
    )
    model.traction((10.0, 0.0), on=right)
    return (
        model.step(
            target=displacement,
            incrementation=steps.fixed(4),
            progress=False,
        ),
        displacement,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "read"))
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    step, displacement = _step()
    comm = MPI.COMM_WORLD
    if arguments.mode == "write":
        step.solve(until=0.5)
        manifest = step.save_checkpoint(arguments.path)
        if comm.rank == 0:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload["writer_rank_count"] != comm.size:
                raise RuntimeError("Nonlinear checkpoint writer evidence is wrong.")
        return

    step.load_checkpoint(arguments.path)
    if step.checkpoints[-1].metadata["reader_rank_count"] != comm.size:
        raise RuntimeError("Nonlinear checkpoint reader evidence is wrong.")
    step.solve()
    expected = 10.0 / (1.0e3 + 1.0e4)
    value = results.probe(displacement, at=(1.0, 0.5))[0]
    if not np.isclose(value, expected, rtol=2.0e-6):
        raise RuntimeError("Restarted nonlinear contact solution is incorrect.")
    if len(step.constraint_dual_history.records) != 5:
        raise RuntimeError("Restarted nonlinear constraint history is incomplete.")
    energy_recorder = step.accepted_history_recorders["conservative_energy"]
    if len(energy_recorder.frames) != 5:
        raise RuntimeError("Restarted nonlinear energy history is incomplete.")
    energy = energy_recorder.evidence(accepted_factor=1.0)
    if energy["status"] != "complete":
        raise RuntimeError("Restarted nonlinear energy evidence is unavailable.")
    if float(energy["relative_energy_balance_error"]) > 1.0e-10:
        raise RuntimeError("Restarted nonlinear contact energy does not close.")


if __name__ == "__main__":
    main()
