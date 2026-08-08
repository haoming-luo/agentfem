"""Two-rank write and reordered one-rank read for cohesive state."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import interfaces


def _law():
    return interfaces.bilinear_cohesive(
        strength=10.0,
        fracture_energy=2.0,
        initial_stiffness=1000.0,
    )


def _topology(indices):
    coordinates = []
    negative = []
    positive = []
    for selected in indices:
        start = len(coordinates)
        x0 = float(selected)
        x1 = x0 + 1.0
        coordinates.extend(((x0, 0.0), (x1, 0.0), (x0, 0.0), (x1, 0.0)))
        negative.append((start, start + 1))
        positive.append((start + 2, start + 3))
    return interfaces.pair_coincident_line_facets(
        np.asarray(coordinates, dtype=float),
        np.asarray(negative, dtype=int),
        np.asarray(positive, dtype=int),
        normal_hint=(0.0, 1.0),
    )


def _values(indices):
    return np.asarray(
        [(0.01 * (index + 1), 0.02 * (index + 1)) for index in indices],
        dtype=float,
    ).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "read", "reject-inconsistent"))
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    comm = MPI.COMM_WORLD
    if arguments.mode in {"write", "reject-inconsistent"}:
        if comm.size != 2:
            raise RuntimeError("Portable cohesive write acceptance requires two ranks.")
        indices = (0, 1, 2) if comm.rank == 0 else (1, 2, 3)
        topology = _topology(indices)
        state = interfaces.CohesiveTransaction(_law(), topology.number_of_points)
        values = _values(indices)
        if arguments.mode == "reject-inconsistent" and comm.rank == 1:
            values[0] += 1.0e-3
        state.initialize(values)
        try:
            interfaces.save_portable_cohesive_state(
                arguments.path,
                topology,
                state,
                comm=comm,
            )
        except RuntimeError as exc:
            if arguments.mode != "reject-inconsistent":
                raise
            if "state differs between ranks" not in str(exc):
                raise RuntimeError("Unexpected cohesive rejection reason.") from exc
            return
        if arguments.mode == "reject-inconsistent":
            raise RuntimeError("Inconsistent owner/ghost cohesive state was accepted.")
        return
    if comm.size != 1:
        raise RuntimeError("Portable cohesive read acceptance requires one rank.")
    indices = (3, 1, 0, 2)
    topology = _topology(indices)
    state = interfaces.CohesiveTransaction(_law(), topology.number_of_points)
    metadata = interfaces.load_portable_cohesive_state(
        arguments.path,
        topology,
        state,
        comm=comm,
    )
    np.testing.assert_allclose(state.committed_maximum, _values(indices))
    if metadata["writer_rank_count"] != 2 or metadata["reader_rank_count"] != 1:
        raise RuntimeError("Cohesive checkpoint rank-count evidence is incomplete.")


if __name__ == "__main__":
    main()
