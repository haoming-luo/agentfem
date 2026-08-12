"""Two-rank write and reordered one-rank read for mixed cyclic state."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import fatigue_fracture, interfaces


def _law():
    monotonic = interfaces.mixed_mode_bilinear_cohesive(
        normal_strength=10.0,
        shear_strength=8.0,
        normal_fracture_energy=2.0,
        shear_fracture_energy=3.0,
        normal_stiffness=1000.0,
        tangential_stiffness=800.0,
        interaction="bk",
        interaction_exponent=1.6,
    )
    return fatigue_fracture.cyclic_cohesive(
        monotonic=monotonic,
        driver=fatigue_fracture.mixed_mode_energy_range_driver(),
        fatigue_coefficient=0.1,
        fatigue_exponent=1.0,
        residual_exponent=1.0,
        range_threshold=0.0,
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


def _cycle(indices):
    valley = []
    peak = []
    for selected in indices:
        value = float(selected + 1)
        local_valley = np.asarray((0.001 * value, 0.00075 * value))
        valley.extend((local_valley, local_valley))
        peak.extend((4.0 * local_valley, 4.0 * local_valley))
    return np.asarray(valley), np.asarray(peak)


def _state(indices):
    topology = _topology(indices)
    state = _law().transaction(topology.number_of_points)
    state.configure_dimension(2)
    valley, peak = _cycle(indices)
    state.begin_cycle(valley, peak, cycles=17)
    state.commit_cycle()
    return topology, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "read"))
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    comm = MPI.COMM_WORLD
    if arguments.mode == "write":
        if comm.size != 2:
            raise RuntimeError("Mixed cyclic write acceptance requires two ranks.")
        indices = (0, 1, 2) if comm.rank == 0 else (1, 2, 3)
        topology, state = _state(indices)
        interfaces.save_portable_cohesive_state(
            arguments.path, topology, state, comm=comm
        )
        return
    if comm.size != 1:
        raise RuntimeError("Mixed cyclic read acceptance requires one rank.")
    indices = (3, 1, 0, 2)
    topology = _topology(indices)
    restored = _law().transaction(topology.number_of_points)
    restored.configure_dimension(2)
    metadata = interfaces.load_portable_cohesive_state(
        arguments.path, topology, restored, comm=comm
    )
    _, expected = _state(indices)
    for name, values in expected.state_arrays().items():
        np.testing.assert_allclose(restored.state_arrays()[name], values)
    if metadata["writer_rank_count"] != 2 or metadata["reader_rank_count"] != 1:
        raise RuntimeError("Mixed cyclic checkpoint rank evidence is incomplete.")


if __name__ == "__main__":
    main()
