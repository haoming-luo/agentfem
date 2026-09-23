"""Complete 2D/3D transient heat workflow with a known analytical solution.

For unit density, heat capacity and conductivity on the unit box:
T(x,t) = 300 + exp(-dimension*pi**2*t) * product(sin(pi*x_i)).
All faces have T=300 K. The platform owns the backward-Euler time loop and output.
Run from an installed AgentFEM environment; no source-path modification needed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from agentfem import constitutive, expressions, fields, mesh, models, results, studies
from agentfem.solvers import LinearSolverOptions


def run(*, dimension=2, cells=12, steps=40, final_time=0.02, output=None):
    if dimension not in (2, 3) or cells < 1 or steps < 1 or final_time <= 0:
        raise ValueError("Use dimension 2 or 3 and positive cells, steps and final_time.")
    comm = MPI.COMM_WORLD
    factory = mesh.rectangle if dimension == 2 else mesh.cuboid
    domain = factory((0.,)*dimension, (1.,)*dimension, (cells,)*dimension,
                     comm=comm, cell_type="quadrilateral" if dimension == 2 else "hexahedron")
    model = models.create(study=studies.transient_heat_transfer(dimension=dimension),
                          mesh=domain, name="heat_decay")
    temperature = model.field(fields.temperature(domain, degree=1, value=300.0))
    # Unit thermal coefficients; mechanical coefficients are unused by this Study.
    model.material(constitutive.thermoelastic(young=1., poisson=0.3, density=1.,
        thermal_expansion=0., conductivity=1., specific_heat=1., reference_temperature=300.))
    initial = " * ".join(f"sin(pi*{axis})" for axis in ('x','y','z')[:dimension])
    expressions.interpolate(temperature, "300 + " + initial)
    exterior = mesh.boundary(domain, lambda x: np.any(
        np.isclose(x[:dimension], 0.) | np.isclose(x[:dimension], 1.), axis=0),
        name="exterior", tag=1)
    model.fix(temperature, on=exterior, value=300.0)
    model.check()
    output_path = Path(output) if output is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    step = model.step(target=temperature, dt=final_time/steps, steps=steps,
        output=output_path, save_every=max(1, steps//4), progress=False,
        solver_options=LinearSolverOptions(ksp_type="preonly", pc_type="lu"))
    simulation = step.solve_result(history=(results.probe_history(
        "center_temperature", at=(0.5,)*dimension, unit="K"),))
    x = ufl.SpatialCoordinate(domain)
    exact = np.exp(-dimension*np.pi**2*final_time)
    for j in range(dimension):
        exact *= ufl.sin(np.pi*x[j])
    measure = ufl.Measure('dx', domain=domain, metadata={'quadrature_degree': 8})
    error = comm.allreduce(fem.assemble_scalar(fem.form(
        (temperature.value-300.0-exact)**2*measure)), op=MPI.SUM)
    norm = comm.allreduce(fem.assemble_scalar(fem.form(exact**2*measure)), op=MPI.SUM)
    report = dict(dimension=dimension, cells_per_axis=cells, steps=steps,
        final_time=final_time, relative_l2_error=float(np.sqrt(error/norm)),
        center_temperature=float(results.probe(temperature, at=(0.5,)*dimension)),
        exact_center=float(300.0 + np.exp(-dimension*np.pi**2*final_time)), status=simulation.status)
    simulation.add_quantity("relative_l2_error", report['relative_l2_error'], unit="1")
    if output_path is not None and comm.rank == 0:
        simulation.write_manifest(output_path.with_suffix('.result.json'), include_histories=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension', type=int, default=2, choices=[2,3])
    parser.add_argument('--cells', type=int, default=12)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--final-time', type=float, default=0.02)
    parser.add_argument('--output', type=Path, default=Path('heat_decay.xdmf'))
    args = parser.parse_args()
    report = run(**vars(args))
    if MPI.COMM_WORLD.rank == 0:
        print(json.dumps(report, indent=2))
    if report['status'] != 'completed' or not np.isfinite(report['relative_l2_error']):
        raise SystemExit(1)
