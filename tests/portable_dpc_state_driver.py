"""Cross-rank portability acceptance for cell-local DPC moment fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import checkpointing, fields, mesh


def _pressure_field():
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 1.0),
        (4, 3),
        comm=MPI.COMM_WORLD,
        cell_type="quadrilateral",
    )
    mixed = fields.displacement_pressure(
        domain,
        displacement_degree=2,
        pressure_family="DPC",
        pressure_degree=1,
    )
    return mixed.collapsed_pressure(name="MEAN_KIRCHHOFF_STRESS")


def _deterministic_values(field) -> np.ndarray:
    """Value every local mode from its original cell and modal position."""

    domain = field.function_space.mesh
    cell_map = domain.topology.index_map(domain.topology.dim)
    local_cells = int(cell_map.size_local + cell_map.num_ghosts)
    original_cells = np.asarray(domain.topology.original_cell_index, dtype=np.int64)
    expected = np.full(field.x.array.shape, np.nan, dtype=float)
    for cell in range(local_cells):
        for local_mode, dof in enumerate(field.function_space.dofmap.cell_dofs(cell)):
            expected[int(dof)] = 100.0 * float(original_cells[cell]) + local_mode + 0.25
    if np.any(~np.isfinite(expected)):
        raise RuntimeError("DPC acceptance driver could not identify every local mode.")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "read"))
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()

    pressure = _pressure_field()
    expected = _deterministic_values(pressure)
    manifest = arguments.checkpoint.with_suffix(".checkpoint.json")
    if arguments.mode == "write":
        pressure.x.array[:] = expected
        pressure.x.scatter_forward()
        bundle = checkpointing.save_portable_state_bundle(
            arguments.checkpoint,
            state={"MEAN_KIRCHHOFF_STRESS": pressure},
        )
        if MPI.COMM_WORLD.rank == 0:
            manifest.write_text(
                json.dumps(bundle, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    checkpointing.load_portable_state_bundle(
        arguments.checkpoint,
        state={"MEAN_KIRCHHOFF_STRESS": pressure},
        record=payload["record"],
        identities=payload["identities"],
    )
    np.testing.assert_array_equal(pressure.x.array, expected)


if __name__ == "__main__":
    main()
