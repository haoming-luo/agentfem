"""Compare the public NAFEMS LE10 benchmark across MPI rank counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpi4py import MPI
import numpy as np

from agentfem import benchmarks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "compare"))
    parser.add_argument("reference", type=Path)
    arguments = parser.parse_args()

    assessment, _ = benchmarks.nafems_le10_3d_benchmark()
    if not assessment.acceptable:
        raise RuntimeError(f"NAFEMS LE10 failed: {assessment.as_dict()}")

    if arguments.action == "write":
        if MPI.COMM_WORLD.size != 1:
            raise RuntimeError("write the NAFEMS LE10 reference with one MPI rank")
        arguments.reference.write_text(
            json.dumps(assessment.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    expected = json.loads(arguments.reference.read_text(encoding="utf-8"))[
        "quantities"
    ]
    for name in (
        "sigma_yy_D_pa",
        "relative_sigma_yy_D_error",
        "u_x_D_m",
        "u_z_D_m",
        "relative_point_D_displacement_error",
    ):
        np.testing.assert_allclose(
            assessment.quantities[name],
            expected[name],
            rtol=2.0e-10,
            atol=2.0e-9,
        )
    for name in (
        "relative_force_balance_error",
        "relative_energy_balance_error",
    ):
        assert assessment.quantities[name] < 1.0e-9


if __name__ == "__main__":
    main()
