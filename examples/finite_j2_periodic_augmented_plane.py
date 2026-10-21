# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Experimental periodic plastic cell with an interior flat-obstacle patch.

The macroscopic gradient loads, unloads and reloads a serial P1 tetrahedral
cell. Contact excludes constrained corners and has disjoint reduced normal
supports. This is a bounded nodal contact example, not general mortar contact.
Units are metre, pascal and newton; the soft material is illustrative.

Run --until 0.25, then --restart <output>/checkpoint.checkpoint.json in a
different output folder. Only accepted displacement frames are visualized.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import (
    boundary_models, constitutive, constraints, fields, mesh, models,
    results, solvers, steps, studies,
)
from agentfem.mesh import abaqus


def make_step(*, cells=4, increments=16, acceleration="aitken", penalty=10000.,
              affine_end_face_support=False, linear_solver=None):
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This bounded augmented-contact path is serial only.")
    if cells < 4 or cells % 4 or increments < 4 or increments % 4:
        raise ValueError("Use cells and increments divisible by four to preserve patch edges and path knots.")
    if not np.isfinite(penalty) or penalty <= 0.:
        raise ValueError("The numerical augmentation penalty must be finite and positive.")
    if type(affine_end_face_support) is not bool:
        raise TypeError("affine_end_face_support must be an explicit boolean.")
    domain = dolfinx_mesh.create_unit_cube(MPI.COMM_SELF, cells, cells, cells)
    model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
    displacement = model.field(fields.displacement(domain))
    material = model.material(constitutive.finite_strain_j2_logarithmic(
        young=1000., poisson=.3, yield_stress=2., hardening_modulus=100.,
    ))

    # Engineering definition of periodic equations: positive-face nodes wrap
    # to negative faces plus the corresponding macroscopic reference motion.
    side = cells + 1
    indices = [(i, j, k) for k in range(side) for j in range(side) for i in range(side)]
    labels = {ijk: index + 1 for index, ijk in enumerate(indices)}
    anchor = labels[(0, 0, 0)]
    references = (labels[(cells, 0, 0)], labels[(0, cells, 0)], labels[(0, 0, cells)])
    equations = []
    for ijk, slave in labels.items():
        axes = [axis for axis, value in enumerate(ijk) if value == cells]
        if not axes or slave in (anchor, *references):
            continue
        wrapped = tuple(0 if axis in axes else value for axis, value in enumerate(ijk))
        for dof in (1, 2, 3):
            equations.append(abaqus.LinearEquation(tuple([
                abaqus.EquationTerm(slave, dof, 1.),
                abaqus.EquationTerm(labels[wrapped], dof, -1.),
                *(abaqus.EquationTerm(references[axis], dof, -1.) for axis in axes),
                abaqus.EquationTerm(anchor, dof, float(len(axes))),
            ])))
    if affine_end_face_support:
        # A separately declared physical fixture, not solver stabilization:
        # prescribe x-motion on z=0 to its macroscopic affine value. Periodicity
        # supplies the corresponding z=1 face. The interior contact patch stays
        # unconstrained in its normal direction. Without this fixture, the net
        # one-sided tool force is carried by the point-pinned corner class.
        for j in range(cells):
            for i in range(cells):
                label = labels[(i, j, 0)]
                if label == anchor:
                    continue
                weights = ((references[0], i / cells), (references[1], j / cells),
                           (anchor, 1. - (i + j) / cells))
                equations.append(abaqus.LinearEquation((
                    abaqus.EquationTerm(label, 1, 1.),
                    *(abaqus.EquationTerm(node, 1, -weight)
                      for node, weight in weights if weight != 0.),
                )))
    gradients = [np.diag((s, s**-.5, s**-.5)) for s in (1., 1.02, .99, 1.02)]
    periodicity = model.constraint(constraints.abaqus_periodic_cell(
        displacement,
        nodes=abaqus.AbaqusNodeTable(
            labels=np.arange(1, len(indices) + 1), coordinates=np.array(indices, dtype=float) / cells,
        ),
        equations=abaqus.AbaqusEquationSet(tuple(equations)),
        anchor_node=anchor, reference_nodes=references,
        deformation_gradient_path=constraints.deformation_gradient_path((0., .25, .75, 1.), gradients),
    ))
    patch = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.) & (x[1] >= .25-1e-12)
        & (x[1] <= .75+1e-12) & (x[2] >= .25-1e-12) & (x[2] <= .75+1e-12),
        name="interior_contact_patch",
    )
    model.add_boundary_model(boundary_models.rigid_contact_pair(
        patch, boundary_models.rigid_body(boundary_models.rigid_plane(
            point=(1.005, 0., 0.), normal=(-1., 0., 0.),
    )), penalty=penalty, initial_penetration_limit=0., name="interior_plane",
    ))
    return model.step(
        target=displacement, material=material, constraints=periodicity,
        incrementation=steps.fixed(increments), output_every=1, progress=False,
        normal_augmentation=steps.normal_augmentation(
            penetration_tolerance=1e-8, complementarity_tolerance=1e-7,
            maximum_augmentations=60, length_unit="m", pressure_unit="Pa",
            acceleration=acceleration,
        ),
        solver_options=solvers.newton(
            relative_tolerance=1e-9, absolute_tolerance=1e-10,
            maximum_iterations=60, line_search="backtracking",
            linear_solver=linear_solver or solvers.direct_solver(),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/periodic_augmented_plane"))
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--increments", type=int, default=16)
    parser.add_argument("--penalty", type=float, default=10000., help="Explicit numerical penalty in Pa/m.")
    parser.add_argument("--acceleration", choices=("none", "aitken"), default="aitken")
    parser.add_argument("--affine-end-face-support", action="store_true",
                        help="Declare finite-area x-motion guides on z=0/1; changes the physical fixture.")
    parser.add_argument("--until", type=float, default=1.)
    parser.add_argument("--restart", type=Path)
    args = parser.parse_args()
    step = make_step(cells=args.cells, increments=args.increments, acceleration=args.acceleration,
                     penalty=args.penalty, affine_end_face_support=args.affine_end_face_support)
    if args.restart:
        step.load_checkpoint(args.restart)
    if not step.accepted_load_factor < args.until <= 1.:
        parser.error("--until must exceed the accepted coordinate and be at most one.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output folder; existing results are not overwritten.")
    result = step.solve_result() if args.until == 1. else None
    if result is None:
        step.solve(until=args.until)
    args.output.mkdir(parents=True, exist_ok=True)
    with results.UnifiedXDMFTimeSeries(args.output / "fields.xdmf") as writer:
        for snapshot in step.snapshots:
            writer.write_fields(snapshot.load_factor, snapshot.solution)
    checkpoint = step.save_checkpoint(args.output / "checkpoint")
    results.export_accepted_contact_points(
        step, args.output / "contact", result=result, vtk_view=True,
        units={"length": "m", "pressure": "Pa", "force": "N"},
    )
    if result is not None:
        result.write_manifest(args.output / "result.json")
    work = step.accepted_history_recorders["normal_contact_work:interior_plane"]
    print(f"Accepted increments: {len(step.accepted_increments)}")
    print(f"Outer updates: {sum(i.checks['normal_augmentations'] for i in step.accepted_increments)}")
    print(f"Maximum penetration: {step.contact_provider.current_point_diagnostics()['maximum_penetration']:.6g} m")
    print(f"Numerical relative-gap work: {work.records[-1]['work']:.6g} J")
    print("Experimental disjoint-normal MPC scope; no automatic energy-verification claim.")
    print(f"Checkpoint: {checkpoint}")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
