"""Write and resume a finite-strain J2 Step across MPI partition counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dolfinx import mesh as dolfinx_mesh
from mpi4py import MPI

from agentfem import constitutive, fields, mesh, models, solvers, steps, studies


def _step(comm):
    domain = dolfinx_mesh.create_unit_cube(comm, 2, 1, 1)
    study = studies.nonlinear_static(physics="solid_mechanics", dimension=3)
    model = models.create(study=study, mesh=domain, name="finite_strain_j2_portable")
    displacement = model.field(fields.displacement(domain))
    left = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 0.0), name="left", tag=1
    )
    right = mesh.boundary(
        domain, lambda x: np.isclose(x[0], 1.0), name="right", tag=2
    )
    y_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[1], 0.0), name="y_symmetry", tag=3
    )
    z_symmetry = mesh.boundary(
        domain, lambda x: np.isclose(x[2], 0.0), name="z_symmetry", tag=4
    )
    model.fix(displacement, on=left, component=0, value=0.0)
    model.fix(displacement, on=y_symmetry, component=1, value=0.0)
    model.fix(displacement, on=z_symmetry, component=2, value=0.0)
    model.fix(displacement, on=right, component=0, value=0.02)
    material = constitutive.finite_strain_j2_logarithmic(
        young=200_000.0,
        poisson=0.3,
        yield_stress=200.0,
        hardening_modulus=2_000.0,
    )
    model.material(material)
    return model.step(
        target=displacement,
        material=material,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-9,
            maximum_iterations=20,
            line_search="backtracking",
        ),
        progress=False,
        name="finite_strain_j2_portable",
    )


def _verify_completed(step) -> None:
    comm = step.solution.function_space.mesh.comm
    state = step.response.state.committed_state_vectors()
    points_per_cell = len(step.response.state.reference_field.points)
    cell_map = step.response.domain.topology.index_map(step.response.domain.topology.dim)
    owned_points = int(cell_map.size_local) * points_per_cell
    peeq = state[:owned_points, -1]
    global_min = comm.allreduce(float(np.min(peeq, initial=np.inf)), op=MPI.MIN)
    global_max = comm.allreduce(float(np.max(peeq, initial=-np.inf)), op=MPI.MAX)
    maximum_displacement = comm.allreduce(
        float(np.max(step.solution.x.array, initial=-np.inf)), op=MPI.MAX
    )
    if step.accepted_load_factor != 1.0:
        raise RuntimeError("Restarted finite-strain J2 path did not complete.")
    if global_min <= 0.0 or global_max - global_min > 2.0e-10:
        raise RuntimeError("Restarted finite-strain J2 state is inconsistent.")
    if abs(maximum_displacement - 0.02) > 2.0e-10:
        raise RuntimeError("Restarted prescribed displacement is incorrect.")
    if [item.load_factor for item in step.accepted_increments] != [
        0.25,
        0.5,
        0.75,
        1.0,
    ]:
        raise RuntimeError("Restarted accepted-increment history is incomplete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "read"))
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    step = _step(MPI.COMM_WORLD)
    if arguments.action == "write":
        step.solve(until=0.5)
        step.save_checkpoint(arguments.root, portable=True)
        return

    manifest = arguments.root.with_name(arguments.root.name + ".checkpoint.json")
    step.load_checkpoint(manifest)
    if step.accepted_load_factor != 0.5:
        raise RuntimeError("Portable finite-strain J2 coordinate was not restored.")
    step.solve()
    _verify_completed(step)


if __name__ == "__main__":
    main()
