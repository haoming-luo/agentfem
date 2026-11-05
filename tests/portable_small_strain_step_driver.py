"""Write or restore the generic small-strain Step across MPI rank counts."""

import argparse
from pathlib import Path

import numpy as np

from test_small_strain_restore_transaction import _step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--phase", choices=("write", "continue"), required=True)
    args = parser.parse_args()
    step = _step()
    comm = step.solution.function_space.mesh.comm
    if args.phase == "write":
        step.solve(until=0.5)
        manifest = step.save_checkpoint(args.directory / "restart")
        if comm.rank == 0:
            print(f"Checkpoint: {manifest}", flush=True)
        return
    manifest = args.directory / "restart.checkpoint.json"
    step.load_checkpoint(manifest)
    assert step.accepted_load_factor == 0.5
    step.solve()
    reference = _step()
    reference.solve()
    np.testing.assert_allclose(
        step.solution.x.array, reference.solution.x.array, rtol=1.0e-11, atol=1.0e-13
    )
    np.testing.assert_allclose(
        step.state.committed_state_vectors(),
        reference.state.committed_state_vectors(),
        rtol=1.0e-11,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        step.response.cauchy_stress.values,
        reference.response.cauchy_stress.values,
        rtol=1.0e-11,
        atol=1.0e-13,
    )
    assert step.last_solve_info.completed_step
    if comm.rank == 0:
        print(f"Portable Step continuation passed with {comm.size} ranks.", flush=True)


if __name__ == "__main__":
    main()
