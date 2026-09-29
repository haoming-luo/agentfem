"""Cross-rank-count acceptance for ordinary nonlinear contact restart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
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
    surface = boundary_models.rigid_plane(
        point=(1.0, 0.5),
        normal=(-1.0, 0.0),
        name="portable_tool_surface",
    )
    motion = boundary_models.prescribed_rigid_motion(
        translation=(-0.01, 0.0),
        reference_point=(1.0, 0.5),
        name="portable_tool_motion",
    )
    contact = model.rigid_obstacle_contact(
        on=right,
        penalty=1.0e4,
        surface=surface,
        motion=motion,
        name="portable_moving_tool_contact",
    )
    return (
        model.step(
            target=displacement,
            incrementation=steps.fixed(4),
            progress=False,
        ),
        displacement,
        contact,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "read"))
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    step, displacement, contact = _step()
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
    if not np.allclose(contact.surface_point.value, (0.995, 0.5), atol=1.0e-14):
        raise RuntimeError("Restarted rigid-tool geometry is incorrect.")
    if step.checkpoints[-1].metadata["reader_rank_count"] != comm.size:
        raise RuntimeError("Nonlinear checkpoint reader evidence is wrong.")
    result = step.solve_result()
    expected = -1.0e4 * 0.01 / (1.0e3 + 1.0e4)
    value = results.probe(displacement, at=(1.0, 0.5))[0]
    if not np.isclose(value, expected, rtol=2.0e-6):
        raise RuntimeError("Restarted nonlinear contact solution is incorrect.")
    if len(step.constraint_dual_history.records) != 5:
        raise RuntimeError("Restarted nonlinear constraint history is incomplete.")
    energy_recorder = step.accepted_history_recorders["conservative_energy"]
    if len(energy_recorder.frames) != 5:
        raise RuntimeError("Restarted nonlinear energy history is incomplete.")
    energy = result.metadata["static_work"]
    if energy["status"] != "complete":
        raise RuntimeError("Restarted nonlinear energy evidence is unavailable.")
    balance_error = float(energy["relative_energy_balance_error"])
    if balance_error > 1.0e-10:
        raise RuntimeError(
            "Restarted nonlinear contact energy does not close: "
            f"relative_error={balance_error:.16g}, evidence={energy!r}."
        )
    work = result.quantity("portable_moving_tool_contact_path_work")
    expected_work = 0.5 * (1.0e4 * 0.01 / 11.0) * 0.01
    if not np.isclose(work, expected_work, rtol=2.0e-6):
        raise RuntimeError("Restarted moving-tool work path is incorrect.")


if __name__ == "__main__":
    main()
