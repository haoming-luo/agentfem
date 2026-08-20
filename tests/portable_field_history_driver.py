"""Cross-rank-count acceptance driver for nodal scientific histories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import fields, histories, mesh


def temperature_field(*, value: float = 0.0):
    domain = mesh.rectangle(
        (0.0, 0.0),
        (1.0, 0.25),
        (4, 2),
        comm=MPI.COMM_WORLD,
        cell_type="triangle",
    )
    return fields.temperature(domain, value=value)


def write(path: Path) -> None:
    temperature = temperature_field()
    history = histories.temperature(temperature)
    temperature.value.interpolate(lambda x: 300.0 + 50.0 * x[0])
    history.record(0.0)
    temperature.value.interpolate(lambda x: 500.0 + 50.0 * x[0])
    history.record(2.0)
    history.save(path)


def read(path: Path) -> None:
    temperature = temperature_field()
    history = histories.FieldHistory.load(path, source=temperature)
    history.apply(1.0)
    coordinates = temperature.space.tabulate_dof_coordinates()
    owned = temperature.space.dofmap.index_map.size_local
    expected = 400.0 + 50.0 * coordinates[:owned, 0]
    np.testing.assert_allclose(
        temperature.value.x.array[:owned],
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert history.portable_identity()["layout"] == "portable_nodal"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "read"))
    parser.add_argument("archive", type=Path)
    arguments = parser.parse_args()
    if arguments.mode == "write":
        write(arguments.archive)
    else:
        read(arguments.archive)


if __name__ == "__main__":
    main()
