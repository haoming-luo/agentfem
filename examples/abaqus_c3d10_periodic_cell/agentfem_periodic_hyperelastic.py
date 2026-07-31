"""C3D10 periodic cell imported from Abaqus and solved by AgentFEM.

The top level states the scientific problem.  Abaqus label preservation,
element conversion, chained *EQUATION elimination, and reduced Newton algebra
live in reusable AgentFEM modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI

from agentfem import (
    constraints,
    constitutive,
    fields,
    mesh,
    models,
    results,
    steps,
    studies,
)
from agentfem.diagnostics import print_on_root
from agentfem.solvers import AffineNewtonOptions
from postprocess_homogenized_response import plot_response


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
            f"[MPI] {comm.size} ranks | distributed Abaqus *EQUATION via dolfinx_mpc",
        )
    print_on_root(comm, "[PRE] Reading Abaqus C3D10 mesh and *EQUATION data...")

    # 1. Import the quadratic tetrahedral mesh and Abaqus periodic equations.
    cell = mesh.read_abaqus_mesh(
        source / "R1f10n30vc.dat",
        output / "mesh" / "periodic_cell.xdmf",
        comm=comm,
        cell_type="tetra10",  # Abaqus C3D10
    )
    equations = mesh.abaqus.read_equations(source / "R1f10n30vc.mpc")

    # 2. State the 3D finite-deformation mechanics model.
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

    # 3. Prescribe the macroscopic deformation gradient.
    lateral_stretch = 1.0 / np.sqrt(stretch)
    target_F = np.diag([stretch, lateral_stretch, lateral_stretch])
    periodicity = model.constraint(
        constraints.abaqus_periodic_cell(
            displacement,
            nodes=cell.nodes,
            equations=equations,
            deformation_gradient=target_F,
            anchor_node=1,
            reference_nodes=(7, 9, 4),  # x, y, z cell-control nodes
        )
    )

    # 4. Declare conventional output, then solve the Study-selected step.
    field_output = results.field_output(
        "U", "S", "E", "EVOL", "F", "P", "MISES", "J", "SENER",
        every="increment",
        configuration="deformed",
        deformation_scale=1.0,
        backend="xdmf",
    )
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
        output=field_output,
        solver_options=AffineNewtonOptions(
            rtol=1.0e-7,
            atol=1.0e-8,
            max_it=25,
            ksp_type="preonly",
            pc_type="lu",
            factor_solver_type="mumps",
        ),
        status_file=output / "periodic_cell.sta",
        name="periodic_neo_hookean",
    )
    result = step.solve_result()

    # 5. Build exact RVE homogenized histories from every saved increment.
    print_on_root(comm, "[POST] Homogenizing accepted increments...")
    cell_reference_volume = periodicity.reference_cell_volume
    homogenized = []
    for snapshot in step.snapshots:
        macro_F = np.eye(3) + snapshot.load_factor * (target_F - np.eye(3))
        homogenized.append(
            results.homogenize_periodic_cell(
                snapshot.solution,
                material,
                macro_deformation_gradient=macro_F,
                cell_reference_volume=cell_reference_volume,
                load_factor=snapshot.load_factor,
            )
        )

    # 6. Record physical checks and write inspection-ready artifacts.
    F = constitutive.hyperelasticity.deformation_gradient(displacement.value)
    J = ufl.det(F)
    dx = ufl.dx(domain=cell.domain)
    minimum_J, maximum_J = results.quadrature_extrema(J, cell.domain, degree=4)
    final_homogenized = homogenized[-1]
    local_maximum_displacement = float(
        np.max(
            np.linalg.norm(
                displacement.value.x.array.reshape(-1, 3),
                axis=1,
            )
        )
    )
    maximum_displacement = float(
        comm.allreduce(local_maximum_displacement, op=MPI.MAX)
    )
    result.add_quantities(
        {
            "target_deformation_gradient": target_F,
            "average_deformation_gradient": results.average(F, measure=dx),
            "average_J": results.average(J, measure=dx),
            "minimum_quadrature_J": minimum_J,
            "maximum_quadrature_J": maximum_J,
            "homogenized_first_piola_stress": final_homogenized.first_piola_stress,
            "homogenized_cauchy_stress": final_homogenized.cauchy_stress,
            "homogenized_stress_consistency_error": (
                final_homogenized.stress_consistency_error
            ),
            "solid_reference_fraction": final_homogenized.solid_reference_fraction,
            "periodic_equation_max_error": periodicity.mismatch(),
            "maximum_displacement": maximum_displacement,
        }
    )
    solve_steps = step.last_solve_info.increments
    result.add_histories(
        [item.load_factor for item in solve_steps],
        {
            "newton_residual": [item.residual_norm for item in solve_steps],
            "newton_iterations": [item.iterations for item in solve_steps],
        },
        abscissa_name="load_factor",
        abscissa_unit=None,
    )
    result.add_histories(
        [item.load_factor for item in homogenized],
        {
            "homogenized_first_piola_11": [
                item.first_piola_stress[0, 0] for item in homogenized
            ],
            "homogenized_cauchy_11": [
                item.cauchy_stress[0, 0] for item in homogenized
            ],
        },
        abscissa_name="load_factor",
        abscissa_unit=None,
    )
    saved_factors = [snapshot.load_factor for snapshot in step.snapshots]
    source_displacements = [
        mesh.abaqus.displacement_in_source_order(snapshot.solution, cell.nodes)
        for snapshot in step.snapshots
    ]
    for control_name, node_label in (
        ("RIGHT", 7),
        ("TOP", 9),
        ("FRONT", 4),
    ):
        source_index = cell.nodes.index(node_label)
        control_displacement = np.asarray(
            [values[source_index] for values in source_displacements]
        )
        reference_coordinate = cell.nodes.coordinate(node_label)
        result.add_history(
            f"{control_name.lower()}_displacement",
            saved_factors,
            control_displacement,
            abscissa_name="load_factor",
            abscissa_unit=None,
            description=f"Abaqus node {node_label} control-point U history.",
        )
        result.add_history(
            f"{control_name.lower()}_coordinate",
            saved_factors,
            reference_coordinate + control_displacement,
            abscissa_name="load_factor",
            abscissa_unit=None,
            description=f"Abaqus node {node_label} current COORD history.",
        )

    output.mkdir(parents=True, exist_ok=True)
    print_on_root(comm, "[OUTPUT] Writing field, history, and visualization data...")
    field_artifacts = field_output.write_finite_strain(
        output,
        domain=cell.domain,
        snapshots=step.snapshots,
        material=material,
        basename="periodic_cell",
    )
    final_cell_fields = field_artifacts.final_fields
    for field in final_cell_fields:
        result.add_field(
            field.name,
            field,
            location="cells",
            description="P0 centroid/whole-element visualization field.",
        )
    normalized_video_format = video_format.lower().lstrip(".")
    if normalized_video_format not in {"gif", "mp4"}:
        raise ValueError("video_format must be 'gif' or 'mp4'.")
    screenshot = animation = undeformed = deformed = None
    if comm.size == 1:
        screenshot = results.render_unified_xdmf_comparison(
            field_artifacts.unified_xdmf,
            output / "periodic_cell_comparison.png",
        )
        if normalized_video_format == "gif":
            animation = results.render_unified_xdmf_animation(
                field_artifacts.unified_xdmf,
                output / "periodic_cell_deformation.gif",
                scalar="UMAG",
                fps=video_fps,
            )
        else:
            undeformed, deformed = mesh.abaqus.write_deformation_vtu_pair(
                source / "R1f10n30vc.dat",
                cell.nodes,
                displacement,
                output,
                deformation_scale=1.0,
                basename="periodic_cell",
            )
            animation = results.render_deformation_animation(
                undeformed,
                step.snapshots,
                cell.nodes,
                output / "periodic_cell_deformation.mp4",
                fps=video_fps,
            )
    history_npz = output / "homogenized_history.npz"
    history_csv = output / "homogenized_history.csv"
    response_plot = output / "homogenized_response.png"
    if comm.rank == 0:
        results.write_homogenized_history(history_npz, homogenized)
        results.write_homogenized_csv(history_csv, homogenized)
        plot_response(history_npz, response_plot)
    comm.barrier()
    if field_artifacts.reference_xdmf is not None:
        result.add_artifact(
            "reference_field_history",
            field_artifacts.reference_xdmf,
        )
    if field_artifacts.unified_xdmf is not None:
        result.add_artifact(
            "field_history",
            field_artifacts.unified_xdmf,
        )
    if field_artifacts.deformed_pvd is not None:
        result.add_artifact(
            "deformed_field_history",
            field_artifacts.deformed_pvd,
        )
    if undeformed is not None:
        result.add_artifact("undeformed_vtu", undeformed)
        result.add_artifact("deformed_vtu", deformed)
    if screenshot is not None:
        result.add_artifact("deformation_comparison", screenshot)
    if animation is not None:
        result.add_artifact("deformation_animation", animation)
    result.add_artifact("homogenized_history_npz", history_npz)
    result.add_artifact("homogenized_history_csv", history_csv)
    result.add_artifact("homogenized_response_plot", response_plot)
    result.metadata["migration_scope"] = {
        "equivalent": [
            "Abaqus C3D10 mesh topology and quadratic geometry",
            "linear *EQUATION periodic constraints",
            "3D geometrically nonlinear static equilibrium",
        ],
        "deliberate_substitution": (
            "The Abaqus USER MATERIAL and unavailable *MPC,USER law are not "
            "reproduced. This validation uses an explicit compressible "
            "Neo-Hookean law and a prescribed macroscopic F."
        ),
    }
    result.metadata["field_output"] = {
        **field_output.summary(),
        "analysis_steps": 1,
        "accepted_load_increments": len(step.last_solve_info.increments),
        "increment_attempts": len(step.last_solve_info.attempts),
        "incrementation": incrementation.summary(),
        "frame_count_including_initial": len(step.snapshots),
        "saved_load_factors": [
            snapshot.load_factor for snapshot in step.snapshots
        ],
        "fields": [
            "Displacement",
            *(field.name for field in final_cell_fields),
        ],
        "abaqus_mapping": {
            "S": "Cauchy stress",
            "E": "LE (spatial logarithmic strain for finite strain)",
            "GREEN": "Green--Lagrange strain, available only when requested explicitly",
            "EVOL": "current element volume",
            "U": "displacement",
            "SDV": "not applicable: Neo-Hookean model has no state variables",
            "control-point U": "RIGHT/TOP/FRONT vector histories",
            "control-point COORD": "RIGHT/TOP/FRONT vector histories",
            "RF/TF": (
                "not yet reported: affine-MPC reactions require verified "
                "constraint-multiplier recovery"
            ),
        },
    }
    ir_path = output / "periodic_cell.afir.json"
    manifest_path = output / "periodic_cell.result.json"
    result.add_artifact("model_ir", ir_path)
    result.add_artifact("result_manifest", manifest_path)
    # Model.write_ir is collective for distributed meshes: every rank enters,
    # while the implementation limits the filesystem write to rank zero.
    model.write_ir(ir_path)
    if comm.rank == 0:
        result.write_manifest(
            manifest_path,
            include_histories=True,
        )
    comm.barrier()
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
