"""Compare the public J2 structural benchmark across MPI rank counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import benchmarks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "compare"))
    parser.add_argument("reference", type=Path)
    arguments = parser.parse_args()

    assessment = benchmarks.j2_thick_cylinder_benchmark()
    if not assessment.acceptable:
        raise RuntimeError(f"external J2 benchmark failed: {assessment.as_dict()}")
    quantities = assessment.quantities

    if arguments.action == "write":
        if MPI.COMM_WORLD.size != 1:
            raise RuntimeError("write the reference with one MPI rank")
        arguments.reference.write_text(
            json.dumps(assessment.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    expected = json.loads(arguments.reference.read_text(encoding="utf-8"))[
        "quantities"
    ]
    for name in (
        "yield_bracket_error",
        "yield_bracket_width",
        "maximum_equivalent_plastic_strain",
        "maximum_displacement",
    ):
        np.testing.assert_allclose(
            quantities[name],
            expected[name],
            rtol=2.0e-9,
            atol=2.0e-13,
        )


if __name__ == "__main__":
    main()
