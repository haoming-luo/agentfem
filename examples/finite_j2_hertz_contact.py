# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded sphere-indentation comparison against the Hertz half-space limit.

This is a convergence *experiment*, not a validation claim.  The model is a
finite, bottom-supported block (full or symmetry-reduced) with a finite-strain J2 law kept in its
elastic regime and a finite penalty.  Hertz assumes a small-strain elastic
half-space with exact nonpenetration.  The report keeps these differences
visible instead of treating one coarse result as a pass/fail reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI
import numpy as np

from agentfem import (
    boundary_models,
    constitutive,
    fields,
    mesh,
    models,
    solvers,
    steps,
    studies,
)


def run_case(
    *, cells: int, penalty: float, indentation: float, quarter: bool = False,
    extent: float = 1.0, increments: int = 4, grading: float = 0.0,
    radius: float = 2.0, linear_policy: str = "direct",
    contact_quadrature_degree: int | None = None,
    displacement_degree: int = 1, bulk_quadrature_degree: int | None = None,
) -> dict[str, object]:
    """Return one serial structural response and its half-space reference."""

    started = time.perf_counter()
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("The bounded Hertz comparison is serial only.")
    if (
        cells < 2 or increments < 1
        or displacement_degree not in (1, 2)
        or (bulk_quadrature_degree is not None and bulk_quadrature_degree not in (2, 3, 4))
        or not all(np.isfinite(value) and value > 0.0 for value in (
            penalty, indentation, extent, radius,
        ))
        or not np.isfinite(grading) or grading < 0.0
    ):
        raise ValueError("Require cells >= 2, increments >= 1, displacement degree 1/2, bulk quadrature degree 2/3/4, positive penalty/indentation/extent/radius and nonnegative grading.")
    linear_choices = {
        "direct": solvers.direct_solver(),
        "gmres_gamg": solvers.LinearSolverOptions(
            ksp_type="gmres", pc_type="gamg", rtol=1.0e-8, max_it=500,
        ),
        "gmres_ilu": solvers.LinearSolverOptions(
            ksp_type="gmres", pc_type="ilu", rtol=1.0e-8, max_it=500,
        ),
    }
    if linear_policy not in linear_choices:
        raise ValueError(f"linear_policy must be one of {tuple(linear_choices)}.")

    young, poisson = 1000.0, 0.3
    domain = dolfinx_mesh.create_box(
        MPI.COMM_SELF,
        (
            np.array((0.0, 0.0, 0.0)) if quarter else np.array((-extent, -extent, 0.0)),
            np.array((extent, extent, extent)),
        ),
        (cells, cells, cells) if quarter else (2 * cells, 2 * cells, cells),
        cell_type=dolfinx_mesh.CellType.hexahedron,
    )
    if grading:
        # A separable, monotone coordinate map preserves hex connectivity and
        # exact quarter/full symmetry while resolving the small contact patch
        # without refining the remote boundaries at the same rate.  This is a
        # benchmark mesh, not an automatic mesh modification by the solver.
        coordinates = domain.geometry.x
        denominator = np.expm1(grading)
        for axis in (0, 1):
            fraction = np.abs(coordinates[:, axis]) / extent
            coordinates[:, axis] = (
                np.sign(coordinates[:, axis]) * extent
                * np.expm1(grading * fraction) / denominator
            )
        depth = (extent - coordinates[:, 2]) / extent
        coordinates[:, 2] = (
            extent - extent * np.expm1(grading * depth) / denominator
        )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True),
        mesh=domain,
        name="quarter_sphere_indentation" if quarter else "finite_block_sphere_indentation",
    )
    displacement = model.field(fields.displacement(domain, degree=displacement_degree))
    bottom = mesh.boundary(domain, lambda x: np.isclose(x[2], 0.0), name="bottom")
    top = mesh.boundary(domain, lambda x: np.isclose(x[2], extent), name="top")
    model.clamp(displacement, on=bottom)
    if quarter:
        x_symmetry = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.0), name="x_symmetry")
        y_symmetry = mesh.boundary(domain, lambda x: np.isclose(x[1], 0.0), name="y_symmetry")
        model.fix(displacement, on=x_symmetry, component=0)
        model.fix(displacement, on=y_symmetry, component=1)
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=young,
            poisson=poisson,
            yield_stress=1.0e9,
            hardening_modulus=100.0,
        )
    )
    center = (0.0, 0.0, extent + radius)
    motion = boundary_models.prescribed_rigid_motion(
        translation=(0.0, 0.0, -indentation),
        rotation=(0.0, 0.0, 0.0),
        reference_point=center,
    )
    model.add_boundary_model(
        boundary_models.rigid_contact_pair(
            top,
            boundary_models.rigid_body(
                boundary_models.rigid_sphere(center=center, radius=radius),
                reference_point=center,
                motion_schedule=boundary_models.prescribed_rigid_motion_schedule(
                    motion, start_time=0.0, end_time=1.0,
                ),
            ),
            penalty=penalty,
        )
    )
    step = model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(increments),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=30,
            line_search="backtracking",
            linear_solver=linear_choices[linear_policy],
        ),
        contact_quadrature_degree=contact_quadrature_degree,
        quadrature_degree=bulk_quadrature_degree,
        progress=False,
    )
    result = step.solve_result()
    provider = step.contact_provider
    contribution = provider.evaluate()
    trace_evaluation = provider.adapter.evaluate(step.solution)
    factor = float(np.asarray(provider.motion_factor.value).reshape(-1)[0])
    projection = provider.surface.project(
        trace_evaluation.query_points,
        motion=provider.pair.rigid_body.motion_schedule.motion,
        factor=factor,
    )
    if not projection.all_valid:
        raise RuntimeError("Accepted indentation has invalid surface projections.")
    response = provider.pair.law.evaluate(projection)
    active = np.asarray(response.active, dtype=bool)
    active_radii = np.linalg.norm(trace_evaluation.query_points[active, :2], axis=1)
    footprint_radius = float(np.max(active_radii)) if active_radii.size else 0.0
    reference_area = float(np.sum(provider.adapter.trace.weights[active]))
    force = -(4.0 if quarter else 1.0) * float(
        contribution.assembly.contact_force_on_structure[2]
    )
    hertz_force = (
        4.0 / 3.0 * young / (1.0 - poisson**2)
        * np.sqrt(radius) * indentation**1.5
    )
    # The half-space force scales as delta^(3/2), so its displacement-
    # controlled work limit is the integral 2*F(delta)*delta/5. This is an
    # external reference, not the finite-block energy-balance definition.
    hertz_work = float(2.0 / 5.0 * hertz_force * indentation)
    hertz_radius = float(np.sqrt(radius * indentation))
    hertz_area = float(np.pi * hertz_radius**2)
    hertz_peak_pressure = float(3.0 * hertz_force / (2.0 * hertz_area))
    # Sample the half-space reference on exactly this finite-element trace.
    # This separates face-quadrature aliasing from the remaining pressure-law,
    # penalty, finite-domain, and finite-strain differences. The comparison
    # uses nominal reference-area weights; it is not a Cauchy-stress norm.
    all_radii = np.linalg.norm(trace_evaluation.query_points[:, :2], axis=1)
    sampled_hertz_pressure = hertz_peak_pressure * np.sqrt(
        np.maximum(0.0, 1.0 - (all_radii / hertz_radius) ** 2)
    )
    trace_weights = provider.adapter.trace.weights
    symmetry_factor = 4.0 if quarter else 1.0
    sampled_hertz_force = float(
        symmetry_factor * np.dot(trace_weights, sampled_hertz_pressure)
    )
    pressure_l1_difference = float(
        symmetry_factor * np.dot(
            trace_weights,
            np.abs(np.asarray(response.pressures) - sampled_hertz_pressure),
        ) / hertz_force
    )
    phase_names = (
        "material_update_seconds", "residual_assembly_seconds",
        "tangent_assembly_seconds", "linear_solve_seconds",
        "line_search_seconds",
    )
    solver_phases = {
        name: float(sum(getattr(item, name) for item in step.accepted_increments))
        for name in phase_names
    }
    full_tool_work = float(symmetry_factor * step.energy_history[-1].tool_work)
    full_recoverable_energy = float(symmetry_factor * (
        step.energy_history[-1].stored_energy
        + step.energy_history[-1].contact_potential
    ))
    return {
        "status": str(result.status),
        "cells_per_side": int(cells),
        "increments": int(increments),
        "cell_count": int((1 if quarter else 4) * cells**3),
        "mesh_grading": float(grading),
        "displacement_degree": int(displacement_degree),
        "bulk_quadrature_degree": int(step.quadrature_degree),
        "linear_policy": linear_policy,
        "reference_contact_edge_spacing": (
            extent / cells if not grading
            else extent * np.expm1(grading / cells) / np.expm1(grading)
        ),
        "domain_fraction": 0.25 if quarter else 1.0,
        "penalty": float(penalty),
        "contact_quadrature_degree": (
            (2 * max(displacement_degree, int(domain.geometry.cmaps[0].degree)))
            if contact_quadrature_degree is None and displacement_degree == 2
            else 3 if contact_quadrature_degree is None
            else int(contact_quadrature_degree)
        ),
        "contact_quadrature_rule": provider.adapter.quadrature_rule,
        "indentation": float(indentation),
        "sphere_radius": radius,
        "block_half_width_and_depth": extent,
        "young": young,
        "poisson": poisson,
        "active_quadrature_points": int(contribution.active_points),
        "active_reference_area": reference_area * (4.0 if quarter else 1.0),
        "hertz_half_space_contact_area": hertz_area,
        "maximum_active_quadrature_radius": footprint_radius,
        "hertz_half_space_contact_radius": hertz_radius,
        "hertz_force_sampled_on_reference_trace": sampled_hertz_force,
        "hertz_trace_sampling_force_ratio": sampled_hertz_force / hertz_force,
        "nominal_pressure_l1_difference_over_hertz_force": pressure_l1_difference,
        "peak_quadrature_pressure": float(np.max(response.pressures)),
        "hertz_half_space_peak_pressure": hertz_peak_pressure,
        "maximum_penetration": float(max(0.0, -np.min(projection.signed_gaps))),
        "full_contact_force": force,
        "hertz_half_space_force": float(hertz_force),
        "hertz_half_space_displacement_work": hertz_work,
        "full_body_tool_work": full_tool_work,
        "tool_work_ratio_to_hertz_half_space": full_tool_work / hertz_work,
        "full_body_stored_plus_penalty_energy": full_recoverable_energy,
        "recoverable_energy_ratio_to_hertz_half_space": (
            full_recoverable_energy / hertz_work
        ),
        "force_ratio_to_hertz": force / hertz_force,
        "contact_potential": float(contribution.assembly.potential_energy),
        "stored_energy": float(step.energy_history[-1].stored_energy),
        "plastic_dissipation": float(step.energy_history[-1].plastic_dissipation),
        "tool_work": float(step.energy_history[-1].tool_work),
        "discrete_work_residual": float(step.energy_history[-1].mechanical_energy_residual),
        "work_semantics": "accepted_trapezoidal_path_not_general_first_law",
        "accepted_solver_phase_seconds": solver_phases,
        "accepted_increment_seconds": float(sum(
            item.total_seconds for item in step.accepted_increments
        )),
        "linear_solve_calls": int(sum(
            item.linear_solve_calls for item in step.accepted_increments
        )),
        "linear_iterations": int(sum(
            item.linear_iterations for item in step.accepted_increments
        )),
        "linear_converged_reasons": tuple(
            reason for item in step.accepted_increments
            for reason in item.linear_converged_reasons
        ),
        "total_case_wall_seconds": float(time.perf_counter() - started),
        "reference_scope": "small_strain_elastic_half_space_exact_contact",
        "model_scope": (
            "finite_bottom_clamped_quarter_block_finite_penalty_elastic_J2"
            if quarter else "finite_bottom_clamped_block_finite_penalty_elastic_J2"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--penalty", type=float, default=1.0e5)
    parser.add_argument("--indentation", type=float, default=0.02)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--extent", type=float, default=1.0)
    parser.add_argument("--grading", type=float, default=0.0)
    parser.add_argument("--contact-quadrature-degree", type=int)
    parser.add_argument("--displacement-degree", type=int, choices=(1, 2), default=1)
    parser.add_argument("--bulk-quadrature-degree", type=int, choices=(2, 3, 4))
    parser.add_argument("--quarter", action="store_true")
    parser.add_argument("--increments", type=int, default=4)
    parser.add_argument(
        "--linear-policy", choices=("direct", "gmres_gamg", "gmres_ilu"),
        default="direct",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_case(
        cells=args.cells, penalty=args.penalty, indentation=args.indentation,
        radius=args.radius,
        quarter=args.quarter, extent=args.extent, increments=args.increments,
        grading=args.grading,
        contact_quadrature_degree=args.contact_quadrature_degree,
        displacement_degree=args.displacement_degree,
        bulk_quadrature_degree=args.bulk_quadrature_degree,
        linear_policy=args.linear_policy,
    )
    formatted = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted + "\n", encoding="utf-8")
    print(formatted)


if __name__ == "__main__":
    main()
