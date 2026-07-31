"""C3D10 periodic cell imported from Abaqus and solved by AgentFEM.

The top level states the scientific problem. Mesh conversion, chained
``*EQUATION`` elimination, weak forms, homogenization, convergence histories,
and result packaging live in reusable AgentFEM modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import (
    constraints,
    constitutive,
    fields,
    mesh,
    models,
    results,
    solvers,
    steps,
    studies,
)
from agentfem.diagnostics import print_on_root


HERE = Path(__file__).resolve().parent


def run(
    *,
    stretch: float = 1.20,
    video_fps: int = 2,
    video_format: str = "gif",
    source=None,
    output=None,
):
    source = Path(source or HERE)
    output = Path(output or HERE / "output")
    comm = MPI.COMM_WORLD
    print_on_root(comm, "[JOB] periodic_neo_hookean_cell")
    if comm.size > 1:
        print_on_root(
            comm,
            f"[MPI] {comm.size} ranks | distributed equations via dolfinx_mpc",
        )

    # Import: preserve Abaqus labels, C3D10 geometry, and equation semantics.
    cell = mesh.read_abaqus_mesh(
        source / "R1f10n30vc.dat",
        output / "mesh" / "periodic_cell.xdmf",
        comm=comm,
        cell_type="tetra10",
    )
    equations = mesh.abaqus.read_equations(source / "R1f10n30vc.mpc")

    # Model: a three-dimensional finite-deformation solid.
    study = studies.nonlinear_static(
        physics="solid_mechanics",
        dimension=3,
        name="periodic_cell_large_deformation",
    )
    model = models.create(study=study, mesh=cell, name="periodic_neo_hookean_cell")
    displacement = model.field(fields.displacement(cell.domain, degree=2))
    material = model.material(
        constitutive.neo_hookean(
            young=1000.0,
            poisson=0.30,
            name="matrix_neo_hookean",
        )
    )

    # Loading and boundary conditions: isochoric uniaxial macro stretch.
    lateral_stretch = 1.0 / np.sqrt(stretch)
    target_F = np.diag([stretch, lateral_stretch, lateral_stretch])
    periodicity = model.constraint(
        constraints.abaqus_periodic_cell(
            displacement,
            nodes=cell.nodes,
            equations=equations,
            deformation_gradient=target_F,
            anchor_node=1,
            reference_nodes=(7, 9, 4),
        )
    )

    # Output: field frames, histories, diagnostics, and optional presentation.
    output_request = results.output_plan(
        output,
        field=results.field_output(
            "U", "S", "E", "EVOL", "F", "P", "MISES", "J", "SENER",
            every="increment",
            configuration="deformed",
            backend="xdmf",
        ),
        requests=(
            results.solver_history(),
            results.periodic_cell_history(periodicity),
            results.source_node_history(
                cell.nodes,
                RIGHT=7,
                TOP=9,
                FRONT=4,
            ),
            results.finite_strain_checks(constraint=periodicity),
        ),
        presentation=results.presentation(
            animation=video_format,
            fps=video_fps,
        ),
        basename="periodic_cell",
    )

    # Step: automatic increments and a backend-neutral Newton policy.
    incrementation = steps.automatic(
        initial=0.25,
        minimum=1.0e-4,
        maximum=0.50,
        max_increments=10,
        max_cutbacks=5,
    )
    step = model.step(
        target=displacement,
        material=material,
        constraints=periodicity,
        incrementation=incrementation,
        output=output_request,
        solver_options=solvers.newton(
            relative_tolerance=1.0e-7,
            absolute_tolerance=1.0e-8,
            maximum_iterations=25,
            linear_solver=solvers.direct_solver(package="mumps"),
        ),
        status_file=output / "periodic_cell.sta",
        name="periodic_neo_hookean",
    )
    result = step.solve_result()
    result = output_request.finalize(
        model=model,
        step=step,
        result=result,
        target=displacement,
        material=material,
        metadata={
            "migration_scope": {
                "equivalent": [
                    "Abaqus C3D10 topology and quadratic geometry",
                    "linear *EQUATION periodic constraints",
                    "3D geometrically nonlinear static equilibrium",
                ],
                "deliberate_substitution": (
                    "The unavailable USER MATERIAL and *MPC,USER laws are not "
                    "reproduced; this case uses a compressible Neo-Hookean law "
                    "and a prescribed macroscopic deformation gradient."
                ),
            }
        },
    )
    print_on_root(comm, "[JOB] COMPLETED")
    return model, result


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stretch", type=float, default=1.20)
    parser.add_argument("--video-fps", type=int, default=2)
    parser.add_argument("--video-format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    selected_model, selected_result = run(
        stretch=arguments.stretch,
        video_fps=arguments.video_fps,
        video_format=arguments.video_format,
        source=arguments.source,
        output=arguments.output,
    )
    print_on_root(MPI.COMM_WORLD, selected_result.format())
