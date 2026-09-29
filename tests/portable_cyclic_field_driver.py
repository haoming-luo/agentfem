"""Write and restore cyclic bulk fields across MPI rank-count changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import fatigue_fracture, fields, mesh


def _state():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.5),
        (6, 4),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    displacement = fields.displacement(domain).value
    return displacement, fatigue_fracture.field_state(displacement=displacement)


def _assign_reference(displacement) -> None:
    displacement.interpolate(
        lambda x: np.vstack(
            (
                0.25 + 2.0 * x[0] - 0.5 * x[1],
                -0.75 + 0.125 * x[0] + 3.0 * x[1],
            )
        )
    )
    displacement.x.scatter_forward()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "read"))
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()

    displacement, state = _state()
    if arguments.mode == "write":
        _assign_reference(displacement)
        manifest = state.save_checkpoint(arguments.checkpoint)
        if MPI.COMM_WORLD.rank == 0:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            assert payload["schema"] == "agentfem.cyclic-field-checkpoint.v2"
            assert payload["writer_rank_count"] == MPI.COMM_WORLD.size
        return

    _assign_reference(displacement)
    expected = displacement.x.array.copy()
    displacement.x.array[:] = -99.0
    displacement.x.scatter_forward()
    metadata = state.load_checkpoint(arguments.checkpoint)
    np.testing.assert_allclose(
        displacement.x.array,
        expected,
        rtol=0.0,
        atol=1.0e-13,
    )
    assert metadata["writer_rank_count"] != 0


if __name__ == "__main__":
    main()
