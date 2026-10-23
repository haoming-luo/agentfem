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
from time import perf_counter

import numpy as np
from dolfinx import mesh as dm
from mpi4py import MPI

import agentfem
from agentfem import boundary_models, constitutive, constraints, fields, mesh, models, solvers, steps, studies
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


def make_step(*, cells=4, degree=1, increments=16, penalty=40000., contact_degree=7,
              yield_stress=2., bulk_degree=4, incrementation=None):
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("This diagnostic is serial only.")
    if cells < 4 or cells % 4 or increments < 4 or increments % 4 or degree not in (1, 2):
        raise ValueError("Use cells/increments divisible by four and displacement degree 1 or 2.")
    domain = dm.create_unit_cube(MPI.COMM_SELF, cells, cells, cells)
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
        progress=False, output_every=None,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--increments", type=int, default=32)
    parser.add_argument("--penalty", type=float, default=40000.)
    parser.add_argument("--contact-degree", type=int, default=7)
    parser.add_argument("--bulk-degree", type=int, default=4)
    parser.add_argument("--yield-stress", type=float, default=2.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh evidence path.")
    report = {"scope": "P1/P2 Gaussian penalty diagnostic, not nodal AL or independent continuum reference",
              "agentfem_version": agentfem.__version__,
              "tool_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_sha256": {name: sha256((Path(agentfem.__file__).parent / name).read_bytes()).hexdigest()
                                for name in ("solvers.py", "_affine_equilibrium.py",
                                             "mechanics/finite_strain_plasticity.py",
                                             "constitutive/quadrature.py",
                                             "boundary_models/dolfinx_contact_trace.py")},
              "units": {"length": "m", "stress": "Pa", "force": "N", "work": "J"},
              "options": {k: v for k, v in vars(args).items() if k != "output"}, "records": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    for cells in args.cells:
        for degree in args.degrees:
            row = {"cells": cells, "degree": degree, "status": "running", "stations": []}
            report["records"].append(row)
            save()
            started = perf_counter()
            try:
                step = make_step(cells=cells, degree=degree, increments=args.increments, penalty=args.penalty,
                                 contact_degree=args.contact_degree, bulk_degree=args.bulk_degree,
                                 yield_stress=args.yield_stress)
                row["full_dofs"] = len(step.solution.x.array)
                for factor in (.25, .5, .75, 1.):
                    step.solve(until=factor)
                    observation = observe_penalty(step, factor)
                    json.dumps(observation, allow_nan=False)
                    row["stations"].append(observation)
                    save()
                    print(json.dumps({"cells": cells, "degree": degree, "factor": factor}), flush=True)
                row["status"] = "completed"
                row["accepted_increments"] = len(step.accepted_increments)
            except Exception as exc:
                row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            row["setup_solve_observe_seconds"] = perf_counter() - started
            save()
    if any(row["status"] != "completed" for row in report["records"]):
        raise RuntimeError("A diagnostic candidate failed; evidence preserved.")


if __name__ == "__main__":
    main()
