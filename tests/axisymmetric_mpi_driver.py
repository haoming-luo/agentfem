"""Two-rank native-axisymmetric J2 structural smoke test."""

from __future__ import annotations

from mpi4py import MPI

from agentfem import benchmarks


def main() -> None:
    assessment = benchmarks.j2_thick_cylinder_benchmark(
        comm=MPI.COMM_WORLD,
        radial_cells=4,
        axial_cells=1,
        increments=24,
        formulation="axisymmetric",
    )
    if not assessment.acceptable:
        raise RuntimeError(assessment.as_dict())
    if assessment.quantities["yield_bracket_error"] >= 0.01:
        raise RuntimeError(assessment.as_dict())
    if MPI.COMM_WORLD.rank == 0:
        print(assessment.as_dict())


if __name__ == "__main__":
    main()
