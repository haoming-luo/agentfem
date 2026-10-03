from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from agentfem import (
    constitutive,
    constraints,
    fields,
    mesh,
    models,
    solvers,
    steps,
    studies,
)
from agentfem.mesh import abaqus


def _distributed_p2_periodic_constraint(target, deformation_gradient):
    """Build one rank-canonical exact periodic graph for a P2 cube."""

    displacement_space, _maps = target.space.sub(0).collapse()
    local_coordinates = np.asarray(
        displacement_space.tabulate_dof_coordinates(), dtype=float
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
        name="distributed_p2_dg0_periodic_cube",
    )


def test_mixed_hyperelastic_acceptance_is_mpi_safe():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed mixed solve requires at least two MPI ranks")
    domain = mesh.cuboid(
        (0, 0, 0), (1, 1, 1), (1, 1, 1),
        comm=MPI.COMM_WORLD, cell_type="tetrahedron",
    )
    model = models.create(
        study=studies.static_solid(dimension=3, nonlinear=True), mesh=domain,
    )
    unknown = model.field(fields.displacement_pressure(domain))
    material = model.material(
        constitutive.mixed_neo_hookean(young=1.0e6, poisson=0.499),
    )
    exterior = mesh.boundary(domain, lambda x: np.full(x.shape[1], True), name="all")
    model.fix(unknown.displacement, on=exterior)

    problem = model.step(target=unknown, material=material, increments=1, progress=False)
    problem.solve()

    local = float(np.max(np.abs(unknown.value.x.array)))
    assert MPI.COMM_WORLD.allreduce(local, op=MPI.MAX) == pytest.approx(0.0)
    assert problem.last_solve_info.converged


def test_mixed_finite_strain_j2_solves_distributed_periodic_block_system():
    if MPI.COMM_WORLD.size < 2:
        pytest.skip("distributed mixed solve requires at least two MPI ranks")
    domain = mesh.cuboid(
        (0, 0, 0),
        (1, 1, 1),
        (1, 1, 1),
        comm=MPI.COMM_WORLD,
        cell_type="tetrahedron",
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=domain,
        name="distributed_mixed_j2_periodic_cube",
    )
    unknown = model.field(fields.displacement_pressure(domain))
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=20_000.0,
            poisson=0.45,
            yield_stress=1.0e8,
        )
    )
    final_gradient = np.diag((1.01, 1.0 / np.sqrt(1.01), 1.0 / np.sqrt(1.01)))
    periodicity = model.constraint(
        _distributed_p2_periodic_constraint(unknown, final_gradient)
    )
    constraint_summary = periodicity.summary()
    assert constraint_summary["supports_parallel"] is True
    assert constraint_summary["parallel_backend"] == "petsc_affine_transformation"
    reduction = periodicity.distributed_reduction()
    assert reduction.full_size > reduction.reduced_size
    assert reduction.summary()["backend"] == "petsc_affine_transformation"

    step = model.step(
        target=unknown,
        material=material,
        constraints=periodicity,
        incrementation=steps.fixed(2),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-9,
            absolute_tolerance=1.0e-11,
            maximum_iterations=20,
            line_search="backtracking",
            linear_solver=solvers.direct_solver(package="mumps"),
        ),
        progress=False,
    )
    result = step.solve_result()

    assert result.status == "completed"
    assert step.last_solve_info.completed_step
    assert periodicity.mismatch() < 2.0e-10
    assert step.last_solve_info.increments[-1].checks[
        "pressure_block_residual_norm"
    ] < 1.0e-8
    assert {"U", "MEAN_KIRCHHOFF_STRESS", "P", "PEEQ"} <= set(result.fields)
    local_pressure = np.asarray(unknown.collapsed_pressure().x.array, dtype=float)
    pressure_maximum = MPI.COMM_WORLD.allreduce(
        float(np.max(np.abs(local_pressure), initial=0.0)),
        op=MPI.MAX,
    )
    assert pressure_maximum > 0.0
