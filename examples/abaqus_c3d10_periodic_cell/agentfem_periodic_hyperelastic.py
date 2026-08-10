"""Quasi-incompressible C3D10H periodic cell solved by AgentFEM.

The top level states the scientific problem. Mesh conversion, chained
``*EQUATION`` elimination, weak forms, homogenization, convergence histories,
and result packaging live in reusable AgentFEM modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    axial_displacement: float = 0.20,
    shear_modulus: float = 1.0,
    bulk_to_shear_ratio: float = 1.0e4,
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

    if comm.size > 1:
        raise NotImplementedError(
            "This mixed macro-control periodic route is currently serial. "
            "Fully prescribed displacement periodicity remains available "
            "through the distributed dolfinx_mpc backend."
        )
    if not 0.0 <= axial_displacement:
        raise ValueError("axial_displacement must be non-negative.")
    if shear_modulus <= 0.0 or bulk_to_shear_ratio <= 0.0:
        raise ValueError(
            "shear_modulus and bulk_to_shear_ratio must be positive."
        )
    bulk_modulus = shear_modulus * bulk_to_shear_ratio

    # Import: C3D10H changes formulation identity, not nodes/connectivity.
    original_mesh = source / "R1f10n30vc.dat"
    selected_mesh = output / "mesh" / "R1f10n30vc_C3D10H.dat"
    derivation = mesh.abaqus.derive_element_formulation(
        original_mesh,
        selected_mesh,
        source_type="C3D10",
        target_type="C3D10H",
    )
    cell = mesh.read_abaqus_mesh(
        selected_mesh,
        output / "mesh" / "periodic_cell_c3d10h.xdmf",
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
    unknown = model.field(fields.displacement_pressure(cell.domain))
    material = model.material(
        constitutive.mixed_neo_hookean(
            shear_modulus=shear_modulus,
            bulk_modulus=bulk_modulus,
            name="quasi_incompressible_matrix_neo_hookean",
        )
    )

    # Three-dimensional uniaxial stress: the axial displacement is prescribed,
    # both transverse normal components are solved, and all macro shear
    # components are zero.
    periodicity = model.constraint(
        constraints.abaqus_periodic_cell(
            unknown,
            nodes=cell.nodes,
            equations=equations,
            control_displacements=(
                (axial_displacement, 0.0, 0.0),  # RIGHT, node 7
                (0.0, None, 0.0),                # TOP, node 9
                (0.0, 0.0, None),                # FRONT, node 4
            ),
            anchor_node=1,
            reference_nodes=(7, 9, 4),
        )
    )

    # Output: field frames, histories, diagnostics, and optional presentation.
    field_variables = [
        "U", "S", "E", "EVOL", "F", "P", "PRESSURE", "MISES", "J", "SENER"
    ]
    output_request = results.output_plan(
        output,
        field=results.field_output(
            *field_variables,
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
        target=unknown,
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
    output_target = unknown.collapsed_displacement(name="U")
    result = output_request.finalize(
        model=model,
        step=step,
        result=result,
        target=output_target,
        material=material,
        metadata={
            "case_definition": {
                "mesh_source": "Abaqus C3D10 geometry with C3D10H formulation identity",
                "periodicity": "linear equation constraints",
                "macro_control": (
                    "uniaxial stress: RIGHT-U1 prescribed; TOP-U2 and "
                    "FRONT-U3 solved; macro shear suppressed"
                ),
                "axial_displacement": axial_displacement,
            },
            "element_formulation": {
                "source_type": "C3D10H",
                "solver_route": "P2 displacement / DG0 pressure mixed formulation",
                "shear_modulus": shear_modulus,
                "bulk_modulus": bulk_modulus,
                "bulk_to_shear_ratio": bulk_to_shear_ratio,
                "strain_energy": (
                    "mu/2*(J^(-2/3)*I1-3) + kappa/2*(J-1)^2"
                ),
                "derivation": derivation.summary(),
            },
        },
    )
    print_on_root(comm, "[JOB] COMPLETED")
    return model, result


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--displacement", type=float, default=0.20)
    parser.add_argument("--shear-modulus", type=float, default=1.0)
    parser.add_argument("--bulk-to-shear-ratio", type=float, default=1.0e4)
    parser.add_argument("--video-fps", type=int, default=2)
    parser.add_argument("--video-format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    selected_model, selected_result = run(
        axial_displacement=arguments.displacement,
        shear_modulus=arguments.shear_modulus,
        bulk_to_shear_ratio=arguments.bulk_to_shear_ratio,
        video_fps=arguments.video_fps,
        video_format=arguments.video_format,
        source=arguments.source,
        output=arguments.output,
    )
    print_on_root(MPI.COMM_WORLD, selected_result.format())
