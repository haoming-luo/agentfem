"""NAFEMS LE10 thick elliptical plate through the public 3D workflow."""

from __future__ import annotations

from pathlib import Path

from mpi4py import MPI

from agentfem import benchmarks


def main() -> None:
    output = (
        Path(__file__).resolve().parents[1]
        / "examples_output"
        / "nafems_le10_3d.xdmf"
    )
    benchmark, simulation = benchmarks.nafems_le10_3d_benchmark(
        comm=MPI.COMM_WORLD,
        output=output,
    )
    if MPI.COMM_WORLD.rank == 0:
        simulation.write_manifest(output.with_suffix(".result.json"))
        print(
            "NAFEMS LE10 sigma_yy(D): "
            f"{benchmark.quantities['sigma_yy_D_pa'] / 1.0e6:.6f} MPa"
        )
        print(
            "relative error: "
            f"{100.0 * benchmark.quantities['relative_sigma_yy_D_error']:.3f}%"
        )


if __name__ == "__main__":
    main()
