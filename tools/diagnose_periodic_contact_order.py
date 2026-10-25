# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Common physical fixture, independent P1/P2 Gaussian penalty-contact study.

This diagnostic uses ordinary AgentFEM Steps, not another solver. It changes
the approximation AND contact enforcement relative to the nodal AL benchmark.
Never compare their reactions as if the discrete models were identical. All
boundary field DOFs, including quadratic midsides, enter the periodic equations.
No field archives or automatic tuning are produced.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import runpy
import subprocess
from time import perf_counter

import numpy as np
from dolfinx import mesh as dm
from mpi4py import MPI

import agentfem
from agentfem import boundary_models, checkpointing, constitutive, constraints, fields, mesh, models, solvers, steps, studies
from agentfem.checkpointing import atomic_write_text
from agentfem.mesh import abaqus


def periodic_field_equations(coordinates):
    """Unit-cube diagnostic fixture; fail closed on unmatched/duplicate DOFs."""
    coordinates = np.asarray(coordinates, dtype=float)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.isfinite(coordinates).all()
            or np.any(coordinates < -1e-12) or np.any(coordinates > 1. + 1e-12)):
        raise ValueError("Expected finite unit-cube field coordinates.")
    keys = [tuple(np.rint(x * 1e12).astype(np.int64)) for x in coordinates]
    labels = {key: i + 1 for i, key in enumerate(keys)}
    if len(labels) != len(keys):
        raise ValueError("Duplicate field coordinates are not admitted by this fixture.")
    scale = 10**12
    anchor = labels[(0, 0, 0)]
    references = tuple(labels[tuple(scale if i == axis else 0 for i in range(3))] for axis in range(3))
    equations = []
    for key, label in labels.items():
        axes = [i for i, value in enumerate(key) if value == scale]
        if axes and label not in (anchor, *references):
            wrapped = tuple(0 if axis in axes else value for axis, value in enumerate(key))
            if wrapped not in labels:
                raise ValueError("Periodic field trace lacks an opposing DOF partner.")
            for component in (1, 2, 3):
                equations.append(abaqus.LinearEquation(tuple([
                    abaqus.EquationTerm(label, component, 1.),
                    abaqus.EquationTerm(labels[wrapped], component, -1.),
                    *(abaqus.EquationTerm(references[axis], component, -1.) for axis in axes),
                    abaqus.EquationTerm(anchor, component, float(len(axes))),
                ])))
        elif not axes and key[2] == 0 and label != anchor:
            # The same finite-area physical x-motion guides as the AL fixture.
            x, y = coordinates[label - 1, :2]
            equations.append(abaqus.LinearEquation(tuple([
                abaqus.EquationTerm(label, 1, 1.),
                *(abaqus.EquationTerm(node, 1, -weight) for node, weight in
                  ((references[0], x), (references[1], y), (anchor, 1. - x - y)) if weight != 0.),
            ])))
    return abaqus.AbaqusNodeTable(labels=np.arange(1, len(keys) + 1), coordinates=coordinates), \
        abaqus.AbaqusEquationSet(tuple(equations)), anchor, references


def diagnostic_coordinate_lines(cells, family="uniform"):
    """Prescribed mesh study, not an adaptive mesher or a solver default.

    The sine map fixes 0, 1/4, 1/2, 3/4, 1 and has derivative
    1 + pi/4*cos(4*pi*s) > 0. It clusters both sides of the two patch
    edges without changing their physical positions or the cell count.
    Normal cosine grading resolves both periodic faces. Geometry is fixed
    before constructing any spaces, constraints, contact traces or state.
    """
    if cells < 4 or cells % 4:
        raise ValueError("Use a positive cell count divisible by four.")
    s = np.linspace(0., 1., cells + 1)
    if family == "uniform":
        return np.tile(s, (3, 1))
    if family != "edge_sine":
        raise ValueError("Unknown diagnostic mesh family.")
    if cells < 8:
        raise ValueError("edge_sine requires at least eight cells to resolve patch-edge grading.")
    tangential = s + np.sin(4. * np.pi * s) / 16.
    # Exact anchors also keep facet selection independent of roundoff.
    tangential[::cells // 4] = np.linspace(0., 1., 5)
    normal = .5 * (1. - np.cos(np.pi * s))
    return np.vstack((normal, tangential, tangential))


def make_step(*, cells=4, degree=1, increments=16, penalty=40000., contact_degree=7,
              yield_stress=2., bulk_degree=4, incrementation=None, checkpoint=None, status_file=None,
              mesh_family="uniform"):
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("This diagnostic is serial only.")
    if cells < 4 or cells % 4 or increments < 4 or increments % 4 or degree not in (1, 2):
        raise ValueError("Use cells/increments divisible by four and displacement degree 1 or 2.")
    lines = diagnostic_coordinate_lines(cells, mesh_family)
    domain = dm.create_unit_cube(MPI.COMM_SELF, cells, cells, cells)
    if mesh_family != "uniform":
        for axis in range(3):
            domain.geometry.x[:, axis] = np.interp(
                domain.geometry.x[:, axis], np.linspace(0., 1., cells + 1), lines[axis])
        mesh.audit_quality(domain, strict=True)
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
    u = model.field(fields.displacement(domain, degree=degree))
    nodes, equations, anchor, references = periodic_field_equations(u.value.function_space.tabulate_dof_coordinates())
    periodicity = model.constraint(constraints.abaqus_periodic_cell(
        u, nodes=nodes, equations=equations, anchor_node=anchor, reference_nodes=references,
        deformation_gradient_path=constraints.deformation_gradient_path(
            (0., .25, .75, 1.), [np.diag((s, s**-.5, s**-.5)) for s in (1., 1.02, .99, 1.02)]),
    ))
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000., poisson=.3, yield_stress=yield_stress, hardening_modulus=100.))
    patch = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.) & (x[1] >= .25-1e-12)
                          & (x[1] <= .75+1e-12) & (x[2] >= .25-1e-12) & (x[2] <= .75+1e-12),
                          name="interior_contact_patch")
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        patch, boundary_models.rigid_body(boundary_models.rigid_plane(
            point=(1.005, 0., 0.), normal=(-1., 0., 0.))), penalty=penalty,
        initial_penetration_limit=0., name="interior_plane"))
    return model.step(
        target=u, material=material, constraints=periodicity,
        quadrature_degree=bulk_degree, contact_quadrature_degree=contact_degree,
        incrementation=steps.fixed(increments) if incrementation is None else incrementation,
        progress=status_file is not None, output_every=None,
        checkpoint=checkpoint, status_file=status_file,
        solver_options=solvers.newton(relative_tolerance=1e-9, absolute_tolerance=1e-10,
                                      maximum_iterations=60, line_search="backtracking",
                                      linear_solver=solvers.direct_solver(package="mumps")),
    )


def observe_penalty(step, factor):
    observe = runpy.run_path(str(Path(__file__).with_name("verify_affine_augmented_path.py")))["observe"]
    observation = observe(step, factor, volume_weighted=True, include_normal_work=False)
    # This route does not own the AL relative-work recorder. Missing work is
    # unavailable, never zero; this diagnostic cannot establish energy closure.
    observation["normal_relative_work_status"] = "unavailable_on_this_diagnostic_route"
    return observation


def patch_pressure_partition(samples, *, edge_width=.0625):
    """Fixed physical edge belt, not one mesh-dependent row of points.

    Only this unit-cube diagnostic owns the .25--.75 square. Samples and their
    reference positions/measure come from the existing contact provider.
    The partition diagnoses force distribution; it is not an error estimator.
    """
    points = np.asarray(samples["reference_positions"])
    weights = np.asarray(samples["reference_weights"])
    pressure = np.asarray(samples["normal_pressures"])
    if (samples["measure_configuration"] != "reference"
            or not 0. < edge_width < .25
            or points.shape != (len(weights), 3) or pressure.shape != weights.shape
            or not all(np.isfinite(x).all() for x in (points, weights, pressure))
            or np.any(weights <= 0.) or np.any(pressure < 0.)):
        raise ValueError("Expected finite reference-area contact samples and positive edge width.")
    yz = points[:, 1:]
    if (not np.allclose(points[:, 0], 1., rtol=0., atol=1e-12)
            or np.any(yz < .25-1e-12) or np.any(yz > .75+1e-12)
            or not np.isclose(weights.sum(), .25, rtol=0., atol=1e-12)):
        raise ValueError("Contact samples do not cover the declared square patch.")
    distance = np.minimum(yz - .25, .75 - yz).min(axis=1)
    edge = distance < edge_width
    rows = {}
    for name, mask in (("edge", edge), ("interior", ~edge)):
        rows[name] = {
            "reference_area": float(weights[mask].sum()),
            "active_reference_area": float(weights[mask & (pressure > 0.)].sum()),
            "normal_force_magnitude": float(weights[mask] @ pressure[mask]),
            "maximum_pressure": float(pressure[mask].max(initial=0.)),
        }
    # A continuous partition avoids mistaking a discontinuous belt indicator's
    # quadrature error for pressure redistribution. Its center weight is one
    # at the center and zero on all four edges; integral over the square=1/9.
    center = 256. * np.prod((yz - .25) * (.75 - yz), axis=1)
    smooth = {name: {
        "weighted_reference_area": float(weights @ fraction),
        "weighted_normal_force": float((weights * fraction) @ pressure),
    } for name, fraction in (("edge", 1. - center), ("center", center))}
    return {"edge_width": edge_width, "classification": "reference_square_distance",
            "partition_quadrature": "sampled_indicator_not_exact_subcell_integration",
            "total_reference_area": float(weights.sum()),
            "normal_force_magnitude": float(weights @ pressure), "regions": rows,
            "smooth_partition": smooth}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--mesh-family", choices=("uniform", "edge_sine"), default="uniform",
                        help="Explicit fixed-geometry discretization study; never automatic tuning.")
    path_policy = parser.add_mutually_exclusive_group()
    path_policy.add_argument("--increments", type=int, default=None,
                             help="Fixed path increments (default: 32).")
    path_policy.add_argument("--plastic-increment-limit", type=float, default=None,
                             help="Use the existing accepted-plastic predictor at this limit; refine it separately from mesh size.")
    parser.add_argument("--penalty", type=float, default=40000.)
    parser.add_argument("--contact-degree", type=int, default=7)
    parser.add_argument("--bulk-degree", type=int, default=4)
    parser.add_argument("--yield-stress", type=float, default=2.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--until", type=float, choices=(.25, .5, .75, 1.), default=1.,
                        help="Stop at a declared common station, without claiming a complete cycle.")
    parser.add_argument("--patch-partition", action="store_true",
                        help="Record fixed reference edge/interior pressure integrals.")
    parser.add_argument("--checkpoint-directory", default=None,
                        help="Optional existing Step checkpoint policy: every 50 accepted increments, retain two.")
    parser.add_argument("--restart", type=Path, default=None,
                        help="Resume one mesh/degree from an ordinary compatible Step checkpoint; no identity override.")
    args = parser.parse_args()
    for cells in args.cells:
        try:
            diagnostic_coordinate_lines(cells, args.mesh_family)
        except ValueError as exc:
            parser.error(str(exc))
    if args.output.exists():
        parser.error("Choose a fresh evidence path.")
    if args.checkpoint_directory is not None and Path(args.checkpoint_directory).exists():
        parser.error("Choose a fresh checkpoint directory; existing diagnostics are not overwritten.")
    if args.restart is not None:
        if len(args.cells) != 1 or len(args.degrees) != 1:
            parser.error("Restart requires exactly one mesh and displacement degree.")
        if not args.restart.is_file():
            parser.error("Restart requires an existing checkpoint file.")
    if args.increments is not None and (args.increments < 4 or args.increments % 4):
        parser.error("Fixed increments must be positive multiples of four for the common stations.")
    control = (steps.fixed(32 if args.increments is None else args.increments)
               if args.plastic_increment_limit is None else steps.automatic(
                   initial=1/32, minimum=1/4096, maximum=1/16,
                   max_increments=4096, max_cutbacks=12, cutback_factor=.5,
                   maximum_inelastic_increment=args.plastic_increment_limit,
                   inelastic_prediction_safety=.8))
    report = {"scope": "P1/P2 Gaussian penalty diagnostic, not nodal AL or independent continuum reference",
              "agentfem_version": agentfem.__version__,
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                  cwd=Path(__file__).resolve().parents[1], text=True).strip(),
              "tool_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_sha256": {name: sha256((Path(agentfem.__file__).parent / name).read_bytes()).hexdigest()
                                for name in ("solvers.py", "_affine_equilibrium.py",
                                             "mechanics/finite_strain_plasticity.py",
                                             "constitutive/finite_strain_plasticity.py",
                                             "constitutive/quadrature.py",
                                             "steps.py", "_nonlinear_problems.py",
                                             "boundary_models/implicit_contact.py",
                                             "boundary_models/_petsc_contact_rows.py",
                                             "boundary_models/dolfinx_implicit_facet_contact.py",
                                             "boundary_models/dolfinx_contact_trace.py")},
              "units": {"length": "m", "stress": "Pa", "force": "N", "work": "J"},
              "options": {k: str(v) if isinstance(v, Path) else v
                          for k, v in vars(args).items() if k != "output"}, "records": []}
    if args.restart is not None:
        with args.restart.open("rb") as stream:
            digest = sha256()
            while block := stream.read(1024 * 1024):
                digest.update(block)
        report["restart_source_sha256"] = digest.hexdigest()
        report["restart_scope"] = "Checkpoint identity is enforced by the Step; elapsed time covers only this segment."
    report["incrementation"] = control.summary()
    report["accuracy_claim"] = "none: limit refinement and spatial refinement are independent diagnostics"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    for cells in args.cells:
        for degree in args.degrees:
            row = {"cells": cells, "degree": degree, "status": "running", "stations": []}
            row["mesh_family"] = args.mesh_family
            row["coordinate_lines"] = diagnostic_coordinate_lines(cells, args.mesh_family).tolist()
            report["records"].append(row)
            save()
            started = perf_counter()
            try:
                checkpoint_directory = (None if args.checkpoint_directory is None else
                    Path(args.checkpoint_directory) / f"cells-{cells}-degree-{degree}")
                checkpoint = (None if checkpoint_directory is None else
                    checkpointing.every(50, directory=checkpoint_directory, keep_last=2))
                step = make_step(cells=cells, degree=degree, penalty=args.penalty,
                                 contact_degree=args.contact_degree, bulk_degree=args.bulk_degree,
                                 yield_stress=args.yield_stress, incrementation=control,
                                 checkpoint=checkpoint,
                                 mesh_family=args.mesh_family,
                                 status_file=(None if checkpoint_directory is None else
                                     checkpoint_directory / "progress.sta"))
                row["full_dofs"] = len(step.solution.x.array)
                start_factor = 0.
                if args.restart is not None:
                    step.load_checkpoint(args.restart)
                    start_factor = step.accepted_load_factor
                    row["resumed_from"] = start_factor
                    if start_factor >= args.until:
                        raise ValueError("Restart station must precede the requested final station.")
                for factor in (.25, .5, .75, 1.):
                    if factor > args.until:
                        break
                    if factor <= start_factor:
                        continue
                    step.solve(until=factor)
                    if checkpoint_directory is not None:
                        row["accepted_station_checkpoint"] = str(step.save_checkpoint(
                            checkpoint_directory / f"station-{factor:.2f}.npz"))
                    observation = observe_penalty(step, factor)
                    if args.patch_partition:
                        partition = patch_pressure_partition(step.contact_provider.accepted_point_samples())
                        np.testing.assert_allclose(partition["normal_force_magnitude"],
                            -observation["force_on_structure"][0], rtol=2e-12, atol=1e-12)
                        observation["patch_partition"] = partition
                    json.dumps(observation, allow_nan=False)
                    row["stations"].append(observation)
                    save()
                    print(json.dumps({"cells": cells, "degree": degree, "factor": factor}), flush=True)
                row["status"] = "completed"
                row["complete_cycle"] = args.until == 1.
                row["accepted_through"] = args.until
                row["accepted_increments"] = len(step.accepted_increments)
                row["rejected_attempts"] = sum(not item.converged for item in step.attempted_increments)
                row["newton_iterations"] = sum(item.iterations for item in step.attempted_increments)
                row["maximum_accepted_plastic_increment"] = max(
                    item.checks["maximum_plastic_increment"] for item in step.accepted_increments)
                if args.plastic_increment_limit is not None and (
                    row["maximum_accepted_plastic_increment"] > args.plastic_increment_limit
                ):
                    raise ValueError("Accepted plastic increment exceeded the declared limit.")
            except KeyboardInterrupt:
                row.update(status="interrupted", complete_cycle=False,
                           setup_solve_observe_seconds=perf_counter() - started)
                save()
                raise
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            row["setup_solve_observe_seconds"] = perf_counter() - started
            save()
    if any(row["status"] != "completed" for row in report["records"]):
        raise RuntimeError("A diagnostic candidate failed; evidence preserved.")


if __name__ == "__main__":
    main()
