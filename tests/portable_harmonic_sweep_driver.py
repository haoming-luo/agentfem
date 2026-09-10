"""Cross-rank-count acceptance driver for scalar harmonic sweep checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import fields, mesh, models, operators, results, studies
from agentfem.constitutive import isotropic_elastic


def harmonic_sweep(*, execution_order: str = "forward"):
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (2.0, 1.0, 1.0),
        (12, 1, 1),
        comm=MPI.COMM_WORLD,
        cell_type="hexahedron",
    )
    model = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
        name="portable_harmonic_bar",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(isotropic_elastic(young=1000.0, poisson=0.0, density=1.0))
    model.fix(displacement, on=mesh.face(domain, axis="x", value=0.0), component=0)
    model.fix(displacement, on=mesh.face(domain, axis="y", value=0.0), component=1)
    model.fix(displacement, on=mesh.face(domain, axis="z", value=0.0), component=2)
    loaded_end = mesh.face(domain, axis="x", value=2.0)
    model.traction((3.0, 0.0, 0.0), on=loaded_end)
    stiffness = model.stiffness(displacement)
    mass = model.mass(displacement)
    damping = operators.rayleigh_damping(
        mass,
        stiffness,
        mass_coefficient=0.4,
        stiffness_coefficient=2.0e-3,
    )
    force = model.external_force(displacement)

    def tip_x(point):
        return point.solution_real[0], point.solution_imaginary[0]

    return model.step(
        target=displacement,
        K=stiffness,
        M=mass,
        C=damping,
        F=force,
        frequencies=(1.25, 0.5, 0.875),
        responses=(
            results.harmonic_average_response(
                "tip_x",
                tip_x,
                on=loaded_end,
                unit="m",
                description="Mean loaded-end axial displacement phasor.",
            ),
        ),
        execution_order=execution_order,
        progress=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "read"))
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()
    comm = MPI.COMM_WORLD

    if arguments.mode == "write":
        step = harmonic_sweep()
        step.solve(max_points=1)
        checkpoint = step.save_checkpoint(arguments.checkpoint)
        if comm.rank == 0:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            assert payload["rank_count_at_write"] == comm.size
            assert payload["portable"] is True
            assert payload["completed_indices"] == [0]
        return

    # Rank-count portability does not relax the frozen scientific request.
    # In particular, the execution order belongs to the restart contract:
    # a resumed sweep may use a different partition, but it must not silently
    # change which pending frequency is evaluated next.
    restarted = harmonic_sweep(execution_order="forward")
    payload = restarted.load_checkpoint(arguments.checkpoint)
    if int(payload["rank_count_at_write"]) == comm.size:
        raise AssertionError("Acceptance read must use a different MPI rank count.")
    restarted_result = restarted.solve_result()
    reference_result = harmonic_sweep().solve_result()

    for name in (
        "tip_x_REAL",
        "tip_x_IMAG",
        "tip_x_AMPLITUDE",
    ):
        np.testing.assert_allclose(
            restarted_result.histories[name].values,
            reference_result.histories[name].values,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    # Algebraically equivalent partitions need not reduce round-off in the
    # same order. Treat residual and energy balance as acceptance evidence,
    # not physical response values that must be bitwise rank-count invariant.
    for name in (
        "relative_residual_norm",
        "relative_cycle_energy_balance_error",
    ):
        assert max(restarted_result.histories[name].values) < 1.0e-8
        assert max(reference_result.histories[name].values) < 1.0e-8
    assert restarted.completed
    assert restarted.execution_events[-1].kind == "sweep_completed"
    assert any(
        item.metadata.get("role") == "restart_source"
        and item.metadata.get("rank_count_at_read") == comm.size
        for item in restarted_result.checkpoints.values()
    )


if __name__ == "__main__":
    main()
