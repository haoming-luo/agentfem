"""Elastic wave-packet propagation through a stiff circular inclusion.

This example mirrors the compact plate wave-packet workflow, but replaces the
hole/cavity idea with a material inclusion. The circular inclusion has the same
density and Poisson ratio as the matrix, and twice the Young's modulus.

Workflow highlights:

- matrix and inclusion are registered as cell/material regions,
- left boundary drives a Gaussian-modulated displacement source,
- top/bottom are made periodic by explicit nodal projection,
- right boundary uses a Lysmer-Kuhlemeyer absorbing boundary,
- time integration is explicit central difference.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import NamedTuple

import numpy as np
from dolfinx import fem
from mpi4py import MPI

SOURCE_PARENT = Path(__file__).resolve().parents[2]
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from agentfem import constraints
from agentfem import amplitudes
from agentfem import fields
from agentfem import io as fem_io
from agentfem import mesh as fem_mesh
from agentfem import models
from agentfem import operators
from agentfem import problems
from agentfem import spaces
from agentfem import studies
from agentfem import time as fem_time
from agentfem.boundary_models import absorbing
from agentfem.constitutive import elasticity


class PeriodicProjection(NamedTuple):
    """Parent dof pairs used for explicit top-bottom periodic projection."""

    pairs: tuple[tuple[np.ndarray, np.ndarray], ...]
    pair_count: int


def build_top_bottom_periodic_projection(V, height: float, tolerance: float):
    """Create component-wise matching top-bottom dof pairs."""

    domain = V.mesh
    if domain.comm.size > 1:
        raise RuntimeError("This explicit periodic projection example is serial-only.")

    bottom_marker = lambda x: np.isclose(x[1], 0.0, rtol=0.0, atol=tolerance)
    top_marker = lambda x: np.isclose(x[1], height, rtol=0.0, atol=tolerance)
    pairs = []
    pair_count = 0
    for component in range(V.num_sub_spaces):
        Vc, _ = V.sub(component).collapse()
        coords = Vc.tabulate_dof_coordinates()
        bottom_parent, bottom_child = fem.locate_dofs_geometrical(
            (V.sub(component), Vc), bottom_marker
        )
        top_parent, top_child = fem.locate_dofs_geometrical((V.sub(component), Vc), top_marker)

        if len(bottom_child) != len(top_child):
            raise RuntimeError(
                "Periodic projection requires matching top/bottom dofs "
                f"for component {component}."
            )

        bottom_order = np.argsort(coords[bottom_child, 0])
        top_order = np.argsort(coords[top_child, 0])
        bottom_parent = np.asarray(bottom_parent[bottom_order], dtype=np.int32)
        top_parent = np.asarray(top_parent[top_order], dtype=np.int32)
        bottom_x = coords[bottom_child[bottom_order], 0]
        top_x = coords[top_child[top_order], 0]

        if not np.allclose(bottom_x, top_x, atol=tolerance, rtol=0.0):
            mismatch = float(np.max(np.abs(bottom_x - top_x)))
            raise RuntimeError(
                "Top/bottom periodic dofs are not aligned in x "
                f"for component {component}: max mismatch={mismatch:.3e}."
            )

        pairs.append((bottom_parent, top_parent))
        pair_count = len(bottom_parent)
    return PeriodicProjection(tuple(pairs), pair_count)


def apply_periodic_projection(function, projection: PeriodicProjection | None) -> None:
    """Enforce top-bottom periodic equality by averaging paired dofs."""

    if projection is None:
        return
    values = function.x.array
    for bottom_dofs, top_dofs in projection.pairs:
        averaged = 0.5 * (values[bottom_dofs] + values[top_dofs])
        values[bottom_dofs] = averaged
        values[top_dofs] = averaged
    function.x.scatter_forward()


def periodic_projection_mismatch(function, projection: PeriodicProjection | None) -> float:
    """Return max absolute top-bottom mismatch after projection."""

    if projection is None:
        return 0.0
    values = function.x.array
    mismatch = 0.0
    for bottom_dofs, top_dofs in projection.pairs:
        if len(bottom_dofs) > 0:
            mismatch = max(mismatch, float(np.max(np.abs(values[bottom_dofs] - values[top_dofs]))))
    return function.function_space.mesh.comm.allreduce(mismatch, op=MPI.MAX)


def cell_tag_function(domain, cell_tags, *, name: str = "MaterialId"):
    """Create a DG0 visualization field from cell tags."""

    Q = fem.functionspace(domain, ("DG", 0))
    field = fem.Function(Q, name=name)
    values = field.x.array
    dofmap = Q.dofmap
    for cell, tag in zip(cell_tags.indices, cell_tags.values):
        values[dofmap.cell_dofs(int(cell))] = float(tag)
    field.x.scatter_forward()
    return field


def vector_dof_magnitude_stats(function, marker):
    """Return max/mean vector dof magnitude for dofs selected by marker."""

    V = function.function_space
    values = function.x.array
    component_values = []
    for component in range(V.num_sub_spaces):
        Vc, _ = V.sub(component).collapse()
        parent, _ = fem.locate_dofs_geometrical((V.sub(component), Vc), marker)
        if len(parent) == 0:
            component_values.append(np.zeros(0, dtype=float))
        else:
            component_values.append(values[np.asarray(parent, dtype=np.int32)])
    if not component_values or any(len(component) == 0 for component in component_values):
        return 0.0, 0.0
    magnitudes = np.sqrt(sum(component**2 for component in component_values))
    local_max = float(np.max(magnitudes))
    local_sum = float(np.sum(magnitudes))
    local_count = int(len(magnitudes))
    comm = function.function_space.mesh.comm
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    mean = 0.0 if global_count == 0 else global_sum / global_count
    return global_max, mean


def main() -> None:
    comm = MPI.COMM_WORLD
    study = studies.second_order_dynamics(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_strain",
        name="wave_packet_stiff_inclusion",
    )

    length = 1.2e-6
    height = 0.24e-6
    cells = (240, 48)
    domain = fem_mesh.rectangle(
        lower=(0.0, 0.0),
        upper=(length, height),
        cells=cells,
        comm=comm,
        cell_type="quadrilateral",
    )
    model = models.create(study=study, mesh=domain, name="wave_packet_inclusion_model")

    displacement = model.field(fields.displacement(domain, degree=1))
    V = displacement.space
    state = problems.second_order_state(displacement)

    center = np.array((0.25 * length, 0.50 * height))
    radius = 0.18 * height

    def inclusion_marker(x):
        return (x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2 <= radius**2

    def inclusion_core_marker(x):
        return (x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2 <= (0.90 * radius) ** 2

    def matrix_marker(x):
        return (x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2 > radius**2

    cell_tags = fem_mesh.mark_cell_regions(
        domain,
        {
            1: matrix_marker,
            2: inclusion_marker,
        },
    )
    material_id = cell_tag_function(domain, cell_tags)
    matrix_region = fem_mesh.cell_region(domain, cell_tags, tag=1, name="matrix")
    inclusion_region = fem_mesh.cell_region(domain, cell_tags, tag=2, name="stiff_inclusion")

    matrix_material = model.material(
        elasticity.isotropic_elastic(
            young=227.5e9,
            density=2900.0,
            poisson=0.27,
            name="matrix isotropic elastic",
        ),
        region=matrix_region,
    )
    inclusion_material = model.material(
        elasticity.isotropic_elastic(
            young=2.0 * matrix_material.young,
            density=matrix_material.density,
            poisson=matrix_material.poisson,
            name="2x Young inclusion isotropic elastic",
        ),
        region=inclusion_region,
    )
    model.check()

    matrix_cp, matrix_cs = elasticity.estimate_elastic_wave_speeds(matrix_material)
    inclusion_cp, _ = elasticity.estimate_elastic_wave_speeds(inclusion_material)
    dx = fem_mesh.cell_measure(domain)
    lumped = problems.LumpedMassOperator.assemble(V, matrix_material.density, measure=dx)
    integrator = fem_time.explicit.central_difference(state=state, mass=lumped)

    tolerance = min(length / cells[0], height / cells[1]) * 1.0e-6
    periodic_projection = build_top_bottom_periodic_projection(V, height, tolerance)

    left = lambda x: np.isclose(x[0], 0.0, rtol=0.0, atol=tolerance)
    frequency = 40.78e9
    width = 3.0 * np.pi / (2.0 * np.pi * frequency)
    source_amplitude = 4.9e-13
    source_pulse = model.amplitude(
        "source_pulse",
        amplitudes.gaussian_modulated_sine(
            amplitude=source_amplitude,
            frequency=frequency,
            width=width,
        ),
    )
    source_bc = model.fix(
        displacement,
        on=left,
        components=0,
        value=source_pulse,
        name="source_displacement",
    )

    right = lambda x: np.isclose(x[0], length, rtol=0.0, atol=tolerance)
    right_boundary = fem_mesh.boundary(domain, right, name="right_absorbing", tag=10)
    abc = model.add_boundary_model(
        absorbing.lysmer_kuhlemeyer_boundary(
            right_boundary.measure,
            density=matrix_material.density,
            pressure_wave_speed=matrix_cp,
            shear_wave_speed=matrix_cs,
            normal=fem_mesh.facet_normal(domain),
        )
    )

    dt = 0.35 * min(length / cells[0], height / cells[1]) / max(matrix_cp, inclusion_cp)
    steps = 2000

    internal_force = model.internal_force(state.u)
    absorbing_force = model.boundary_force(abc, state.v_mid)
    residual_operator = model.force_balance(
        internal=internal_force,
        absorbing=absorbing_force,
        name="R_internal_plus_absorbing",
    )

    out = Path(__file__).resolve().parents[1] / "examples_output" / "wave_packet_inclusion_2d.xdmf"
    stepper = fem_time.TimeStepper(
        total_steps=steps,
        dt=dt,
        save_every=10,
        print_every=100,
    )

    if comm.rank == 0:
        print(
            "Wave packet through stiff inclusion: "
            f"E_inclusion/E_matrix={inclusion_material.young / matrix_material.young:.1f}, "
            f"dt={dt:.3e} s, steps={steps}",
            flush=True,
        )

    with fem_io.XDMFTimeSeries(out, domain) as xdmf:
        xdmf.write_fields(0.0, state.u, state.v, material_id)
        for info in stepper:
            t = info.time
            source_bc.update(t)

            integrator.predict_displacement(dt)
            constraints.apply_dirichlet_bcs(state.u_next, [source_bc.bc])
            apply_periodic_projection(state.u_next, periodic_projection)
            integrator.update_displacement()

            integrator.update_midstep_velocity(dt)
            apply_periodic_projection(state.v_mid, periodic_projection)

            residual = operators.assemble_vector(residual_operator)
            integrator.solve_acceleration(residual)
            residual.destroy()
            apply_periodic_projection(state.a_next, periodic_projection)

            integrator.update_velocity(dt)
            apply_periodic_projection(state.v_next, periodic_projection)
            integrator.advance_velocity_acceleration()

            if info.should_save:
                xdmf.write_fields(t, state.u, state.v, material_id)
            if info.should_print and comm.rank == 0:
                periodic_err = periodic_projection_mismatch(state.u, periodic_projection)
                inclusion_max, inclusion_mean = vector_dof_magnitude_stats(
                    state.u, inclusion_core_marker
                )
                print(
                    f"step {info.index:4d}/{steps} "
                    f"t={t:.3e} max|u|={np.max(np.abs(state.u.x.array)):.3e} "
                    f"inclusion_max|u|={inclusion_max:.3e} "
                    f"inclusion_mean|u|={inclusion_mean:.3e} "
                    f"periodic_err={periodic_err:.3e}",
                    flush=True,
                )

    if comm.rank == 0:
        print(f"Wave-packet inclusion result: {out}")


if __name__ == "__main__":
    main()
