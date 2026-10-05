"""Write and resume mixed finite-strain J2 across MPI partition counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import constitutive, constraints, fields, mesh, models, solvers, steps, studies
from agentfem.mesh import abaqus


def _distributed_periodic_constraint(target, deformation_gradient):
    displacement_space, _maps = target.space.sub(0).collapse()
    local_coordinates = np.asarray(
        displacement_space.tabulate_dof_coordinates(),
        dtype=float,
    )
    coordinate_keys = {
        tuple(np.rint(point / 1.0e-12).astype(np.int64))
        for shard in MPI.COMM_WORLD.allgather(local_coordinates)
        for point in shard
    }
    ordered_keys = tuple(sorted(coordinate_keys))
    coordinates = np.asarray(ordered_keys, dtype=float) * 1.0e-12
    labels = np.arange(1, len(coordinates) + 1, dtype=np.int64)
    label_by_key = {
        key: int(label) for key, label in zip(ordered_keys, labels, strict=True)
    }

    def label_at(point):
        key = tuple(np.rint(np.asarray(point) / 1.0e-12).astype(np.int64))
        return label_by_key[key]

    anchor = label_at((0.0, 0.0, 0.0))
    references = (
        label_at((1.0, 0.0, 0.0)),
        label_at((0.0, 1.0, 0.0)),
        label_at((0.0, 0.0, 1.0)),
    )
    controls = {anchor, *references}
    equations = []
    for label, coordinate in zip(labels, coordinates, strict=True):
        active_axes = tuple(
            axis
            for axis in range(3)
            if abs(float(coordinate[axis]) - 1.0) <= 1.0e-12
        )
        if not active_axes or int(label) in controls:
            continue
        wrapped = coordinate.copy()
        wrapped[list(active_axes)] = 0.0
        base = label_at(wrapped)
        for component in (1, 2, 3):
            terms = [
                abaqus.EquationTerm(int(label), component, 1.0),
                abaqus.EquationTerm(base, component, -1.0),
            ]
            terms.extend(
                abaqus.EquationTerm(references[axis], component, -1.0)
                for axis in active_axes
            )
            terms.append(
                abaqus.EquationTerm(
                    anchor,
                    component,
                    float(len(active_axes)),
                )
            )
            equations.append(abaqus.LinearEquation(tuple(terms)))
    return constraints.abaqus_periodic_cell(
        target,
        nodes=abaqus.AbaqusNodeTable(labels=labels, coordinates=coordinates),
        equations=abaqus.AbaqusEquationSet(tuple(equations)),
        deformation_gradient=np.asarray(deformation_gradient, dtype=float),
        anchor_node=anchor,
        reference_nodes=references,
        name="portable_mixed_p2_dg0_cube",
    )


def _step(comm):
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1, 1, 1),
        comm=comm,
        cell_type="tetrahedron",
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="portable_mixed_finite_strain_j2",
    )
    unknown = model.field(fields.displacement_pressure(domain))
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=20_000.0,
            poisson=0.45,
            yield_stress=80.0,
            hardening_modulus=200.0,
        )
    )
    final_gradient = np.diag((1.04, 0.99, 0.99))
    periodicity = model.constraint(
        _distributed_periodic_constraint(unknown, final_gradient)
    )
    step = model.step(
        target=unknown,
        material=material,
        constraints=periodicity,
        incrementation=steps.fixed(4),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-10,
            maximum_iterations=20,
            line_search="backtracking",
            linear_solver=solvers.direct_solver(package="mumps"),
        ),
        progress=False,
        name="portable_mixed_finite_strain_j2",
    )
    return step, unknown, periodicity


def _verify_completed(step, unknown, periodicity) -> None:
    comm = step.solution.function_space.mesh.comm
    if step.accepted_load_factor != 1.0:
        raise RuntimeError("Restarted mixed J2 path did not complete.")
    if periodicity.mismatch() >= 2.0e-10:
        raise RuntimeError("Restarted mixed J2 periodic equations are inconsistent.")
    if [item.load_factor for item in step.accepted_increments] != [
        0.25,
        0.5,
        0.75,
        1.0,
    ]:
        raise RuntimeError("Restarted mixed J2 increment history is incomplete.")
    pressure = np.asarray(unknown.collapsed_pressure().x.array, dtype=float)
    pressure_span = comm.allreduce(
        float(np.max(pressure, initial=-np.inf)),
        op=MPI.MAX,
    ) - comm.allreduce(
        float(np.min(pressure, initial=np.inf)),
        op=MPI.MIN,
    )
    if pressure_span > 2.0e-8:
        raise RuntimeError("Restarted mixed pressure field is not homogeneous.")
    peeq = step.response.state.committed["equivalent_plastic_strain"]
    if peeq.global_max() <= 0.0:
        raise RuntimeError("Restarted mixed J2 material state did not remain plastic.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("write", "read"))
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    step, unknown, periodicity = _step(MPI.COMM_WORLD)
    if arguments.action == "write":
        step.solve(until=0.5)
        step.save_checkpoint(arguments.root, portable=True)
        return

    manifest = arguments.root.with_name(arguments.root.name + ".checkpoint.json")
    step.load_checkpoint(manifest)
    if step.accepted_load_factor != 0.5:
        raise RuntimeError("Portable mixed J2 coordinate was not restored.")
    step.solve()
    _verify_completed(step, unknown, periodicity)


if __name__ == "__main__":
    main()
