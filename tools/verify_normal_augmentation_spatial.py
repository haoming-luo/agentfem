# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Finite-Hencky elastic-limit manufactured solution with a rigid stop.

u=(A*x*(1-x)-d*x,0,0), F=diag(s,1,1), s=1+A*(1-2*x)-d.
At nu=0: P11=E*log(s)/s, all other P entries zero.
Reference body force is -Div(P)=(2*A*E*(1-log(s))/s**2,0,0).
The right rigid plane supplies P11(1)<0; other free faces have zero traction.
This is nonuniform spatial verification, not plastic-contact validation.
The manufactured field is the final equilibrium reference; the load ramp
is not a manufactured transient path or an energy-integration reference.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import ufl
from dolfinx import fem, mesh as dm
from mpi4py import MPI
from agentfem import boundary_models, constitutive, fields, mesh, models, solvers, steps, studies
from agentfem.boundary_models.dolfinx_contact_trace import __file__ as adapter_source

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tool-travel", type=float, default=0.,
                        help="Prescribed compression d in metres; default retains the fixed-stop reference.")
    args = parser.parse_args()
    if len(args.cells) < 2 or sorted(set(args.cells)) != args.cells or min(args.cells) < 1:
        parser.error("Use at least two strictly increasing positive mesh counts.")
    if args.output.exists():
        parser.error("Use a fresh output file.")
    E, A = 1000., .02
    d = args.tool_travel
    if not np.isfinite(d) or not 0. <= d < 1-A:
        parser.error("Tool travel must be finite, nonnegative and retain a positive final deformation gradient.")
    records = []
    for cells in args.cells:
        domain = dm.create_unit_cube(MPI.COMM_SELF, cells, cells, cells)
        model = models.create(study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain)
        u = model.field(fields.displacement(domain))
        left = mesh.boundary(domain, lambda x: np.isclose(x[0], 0.), name="left")
        right = mesh.boundary(domain, lambda x: np.isclose(x[0], 1.), name="right")
        model.clamp(u, on=left)
        material = model.material(constitutive.finite_strain_j2_logarithmic(
            young=E, poisson=0., yield_stress=1e9, hardening_modulus=0.,
        ))
        x = ufl.SpatialCoordinate(domain)
        s = 1 + A*(1-2*x[0]) - d
        model.body_force(ufl.as_vector((2*A*E*(1-ufl.ln(s))/s**2, 0., 0.)))
        model.add_boundary_model(boundary_models.rigid_contact_pair(
            right, boundary_models.rigid_body(
                boundary_models.rigid_plane(point=(1., 0., 0.), normal=(-1., 0., 0.)),
                motion_schedule=(None if d == 0. else boundary_models.prescribed_rigid_motion_schedule(
                    boundary_models.prescribed_rigid_motion(translation=(-d, 0., 0.)),
                    start_time=0., end_time=1.,
                )),
            ), penalty=5000., initial_penetration_limit=0., name="stop",
        ))
        problem = model.step(
            target=u, material=material, incrementation=steps.fixed(4), progress=False,
            normal_augmentation=steps.normal_augmentation(
                penetration_tolerance=1e-9, complementarity_tolerance=1e-8,
                maximum_augmentations=120, length_unit="m", pressure_unit="Pa",
            ),
            solver_options=solvers.newton(relative_tolerance=1e-10, absolute_tolerance=1e-11,
                                          maximum_iterations=30, line_search="backtracking"),
        )
        started = perf_counter()
        result = problem.solve_result()
        exact = ufl.as_vector((A*x[0]*(1-x[0])-d*x[0], 0., 0.))
        delta = u.value - exact
        measure = ufl.dx(metadata={"quadrature_degree": 6})
        l2 = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(delta, delta)*measure)))
        h1 = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(delta), ufl.grad(delta))*measure)))
        force = problem.contact_provider.evaluate().contact_force_on_structure
        expected_force = E*np.log(1-A-d)/(1-A-d)
        point = problem.contact_provider.accepted_point_samples()
        row = dict(cells=cells, l2_error=float(l2), h1_seminorm_error=float(h1),
                   force_x=float(force[0]), exact_force_x=float(expected_force),
                   force_error=float(abs(force[0]-expected_force)),
                   maximum_pressure_error=float(np.max(abs(point['normal_pressures']+expected_force))),
                   peeq=float(np.max(problem.response.state.committed_state_vectors()[:,9])),
                   maximum_penetration=float(np.max(point['penetrations'])),
                   solve_and_measure_seconds=perf_counter()-started,
                   augmentations=sum(i.normal_augmentations for i in problem.accepted_increments),
                   status=result.status)
        records.append(row)
        print(json.dumps(row), flush=True)
    report = dict(scope="nonuniform_finite_Hencky_elastic_limit_constant_normal_plane_MMS_not_plastic_or_general_mortar",
                  amplitude=A, tool_travel=d, young=E, poisson=0., penalty=5000., records=records,
                  l2_orders=[float(np.log(a['l2_error']/b['l2_error'])/np.log(b['cells']/a['cells'])) for a,b in zip(records,records[1:])],
                  h1_orders=[float(np.log(a['h1_seminorm_error']/b['h1_seminorm_error'])/np.log(b['cells']/a['cells'])) for a,b in zip(records,records[1:])])
    report['script_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    report['contact_adapter_sha256'] = sha256(Path(adapter_source).read_bytes()).hexdigest()
    report['verification_passed'] = (
        all(row['status'] == 'completed' and row['peeq'] == 0.
            and row['maximum_penetration'] <= 1e-9 for row in records)
        and min(report['l2_orders']) > 1.8 and min(report['h1_orders']) > .9
        and all(b['force_error'] < a['force_error']
                and b['maximum_pressure_error'] < a['maximum_pressure_error']
                for a, b in zip(records, records[1:]))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if not report['verification_passed']:
        raise RuntimeError('The declared manufactured-solution refinement screen failed.')


if __name__ == "__main__":
    main()
