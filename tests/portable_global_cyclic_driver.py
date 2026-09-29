"""Resume one global cyclic bulk-plus-interface state across MPI sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import fatigue_fracture, fields, fracture, interfaces

from portable_cohesive_dynamics_driver import _split_strip


def _step():
    split = _split_strip()
    domain = interfaces.create_dolfinx_split_mesh(split, comm=MPI.COMM_WORLD)
    displacement = fields.displacement(domain).value
    monotonic = interfaces.bilinear_cohesive(
        strength=10.0,
        fracture_energy=2.0,
        initial_stiffness=1000.0,
    )
    law = fatigue_fracture.cyclic_cohesive(
        monotonic=monotonic,
        fatigue_coefficient=0.02,
        fatigue_exponent=1.0,
        range_threshold=0.0,
        residual_exponent=1.0,
    )
    cohesive = fracture.mode_i_cohesive_force(
        split,
        displacement,
        law,
        normal_hint=(0.0, 1.0),
    )
    named = fracture.named_cohesive_forces(crack=cohesive)
    state = fatigue_fracture.field_state(displacement=displacement)
    positive_nodes = np.unique(split.positive_facets).astype(int)

    def impose(load: float) -> float:
        displacement.x.array[:] = 0.0
        if hasattr(cohesive, "node_to_block_dof"):
            mapping = cohesive.node_to_block_dof
            owned = np.ones(mapping.shape, dtype=bool)
        else:
            mapping = cohesive.input_node_to_block_dof
            owned = cohesive.input_node_owned
        opening = 0.02 * float(load)
        for node in positive_nodes:
            if not owned[node]:
                continue
            dof = int(mapping[node])
            if dof >= 0:
                displacement.x.array[2 * dof + 1] = opening
        displacement.x.scatter_forward()
        return opening

    def solve_equilibrium(*, load, branch, cycle):
        opening = impose(load)
        return {
            "iterations": 1,
            "reaction": float(load),
            "control_displacement": opening,
            "energy_balance_error": 0.0,
            "metadata": {"branch": branch, "cycle": int(cycle)},
        }

    step = fatigue_fracture.global_cyclic_fatigue_step(
        cycle=fatigue_fracture.force_cycle(fmin=0.1, fmax=1.0),
        stop_cycle=4,
        interfaces=named,
        state=state,
        solve_equilibrium=solve_equilibrium,
        landing_cycles=(2,),
        maximum_opening_feedback=0.1,
        name="portable global cyclic",
    )
    return step, displacement, named


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "read"))
    parser.add_argument("checkpoint", type=Path)
    arguments = parser.parse_args()

    if arguments.mode == "write":
        partial, _displacement, _interfaces = _step()
        partial.run(until_cycle=2)
        manifest = partial.save_checkpoint(arguments.checkpoint)
        if MPI.COMM_WORLD.rank == 0:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            assert payload["bulk_state"]["portability"].startswith("portable")
        return

    reference, reference_displacement, reference_interfaces = _step()
    reference.run()
    restarted, restarted_displacement, restarted_interfaces = _step()
    restarted.load_checkpoint(arguments.checkpoint)
    if restarted.current_cycle != 2:
        raise RuntimeError("Global cyclic restart did not restore the accepted cycle.")
    restarted.run()
    np.testing.assert_allclose(
        restarted_displacement.x.array,
        reference_displacement.x.array,
        rtol=0.0,
        atol=1.0e-13,
    )
    if restarted_interfaces.snapshot() != reference_interfaces.snapshot():
        raise RuntimeError("Global cyclic restart changed cohesive interface state.")
    if restarted.ledger.snapshot() != reference.ledger.snapshot():
        raise RuntimeError("Global cyclic restart changed the cycle-jump ledger.")


if __name__ == "__main__":
    main()
