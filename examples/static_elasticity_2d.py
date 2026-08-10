"""Small static linear-elasticity example using the AgentFEM workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import benchmarks
from agentfem import fields
from agentfem import mesh as fem_mesh
from agentfem import models
from agentfem import studies
from agentfem.constitutive import elasticity
from agentfem.diagnostics import print_on_root
from agentfem.diagnostics import max_magnitude
from agentfem.solvers import LinearSolverOptions


def main() -> dict[str, float]:
    comm = MPI.COMM_WORLD

    # 1. Study: define the analysis type and mechanical assumption.
    study = studies.linear_static(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_strain",
        name="cantilever_plane_strain_static",
    )

    # 2. Mesh and model: create the geometry mesh and attach it to a model.
    mesh_cells = (40, 8)
    element_degree = 2
    domain = fem_mesh.rectangle(
        lower=(0.0, 0.0),
        upper=(1.0, 0.2),
        cells=mesh_cells,
        comm=comm,
        cell_type="quadrilateral",
    )
    model = models.create(study=study, mesh=domain, name="cantilever_model")

    # 3. Field: create the unknown displacement field.
    displacement = model.field(fields.displacement(domain, degree=element_degree))

    # 4. Material: assign an isotropic linear-elastic material.
    model.material(
        elasticity.isotropic_elastic(
            young=210.0e9,
            poisson=0.30,
            density=7800.0,
            name="steel-like plane strain",
        )
    )

    # 5. Constraints: clamp the left boundary.
    def left(x):
        return np.isclose(x[0], 0.0)

    left_boundary = fem_mesh.boundary(domain, left, name="left", tag=1)
    model.fix(displacement, on=left_boundary, value=0.0)

    # 6. Load: apply a traction on the right boundary.
    def right(x):
        return np.isclose(x[0], 1.0)

    right_boundary = fem_mesh.boundary(domain, right, name="right", tag=2)
    model.traction(value=(1.0e8, -1.0e6), on=right_boundary)
    model.check()

    # 7. Step and solve: assemble K u = F and solve the linear system.
    step = model.step(
        target=displacement,
        solver_options=LinearSolverOptions(ksp_type="preonly", pc_type="lu"),
        name="cantilever_Ku_eq_F",
    )
    ir_out = (
        Path(__file__).resolve().parents[1]
        / "examples_output"
        / "static_elasticity_2d.afir.json"
    )
    model.write_ir(
        ir_out,
        metadata={
            "example": "static_elasticity_2d",
            "purpose": "executable AF-IR record before solve",
        },
    )
    out = (
        Path(__file__).resolve().parents[1]
        / "examples_output"
        / "static_elasticity_2d.xdmf"
    )
    simulation = step.solve_result(output=out)
    observables = {
        "maximum_displacement": max_magnitude(displacement.value),
    }
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.linear_static_cantilever"
    )

    # 8. Output: engineering-default U/S/E/MISES share one XDMF/HDF5 result.
    simulation.add_quantity(
        "maximum_displacement",
        observables["maximum_displacement"],
        unit="m",
    )
    simulation.add_artifact("afir", ir_out)
    simulation.add_artifact("displacement", out)
    quality = simulation.verify(
        "release",
        claims=golden.claims(observables),
        required_quantities=("maximum_displacement",),
        required_artifacts=("afir", "displacement"),
    )
    quality.require()
    if comm.rank == 0:
        simulation.write_manifest(
            out.with_suffix(".result.json"),
            include_histories=True,
        )

    print_on_root(comm, model.tree())
    print_on_root(comm, f"AF-IR record: {ir_out}")
    print_on_root(comm, f"Static elasticity result: {out}")
    print_on_root(
        comm,
        f"Static golden observable: max|U|={observables['maximum_displacement']:.16e}",
    )
    print_on_root(comm, simulation.format())
    return observables


if __name__ == "__main__":
    main()
