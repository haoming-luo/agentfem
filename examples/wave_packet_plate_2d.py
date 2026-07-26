"""Simplified elastic plate wave-packet example.

This is a compact AgentFEM-oriented version of the wave-packet workflow:

- left boundary: prescribed Gaussian-modulated displacement source,
- top/bottom: explicit nodal periodic projection,
- right boundary: Lysmer-Kuhlemeyer absorbing boundary model,
- time integration: explicit central difference.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import NamedTuple

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

SOURCE_PARENT = Path(__file__).resolve().parents[2]
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from agentfem import assembly
from agentfem import amplitudes
from agentfem import constraints
from agentfem import forms
from agentfem import io as fem_io
from agentfem import mesh as fem_mesh
from agentfem import problems
from agentfem import time as fem_time
from agentfem import spaces
from agentfem.boundary_models import absorbing
from agentfem.constitutive import elasticity
from agentfem.diagnostics import print_on_root


class PeriodicProjection(NamedTuple):
    """Parent dof pairs used for explicit top-bottom periodic projection."""

    pairs: tuple[tuple[np.ndarray, np.ndarray], ...]
    pair_count: int


def build_top_bottom_periodic_projection(V, height: float, tolerance: float):
    """Create component-wise matching top-bottom dof pairs.

    This mirrors the projection used in the wave_packet demo: after each
    explicit update, paired top/bottom dofs are averaged component by component.
    """

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


def main() -> None:
    comm = MPI.COMM_WORLD
    length = 1.2e-6
    height = 0.24e-6
    denx = 240
    deny = 48
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([length, height])],
        [denx,deny],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = spaces.vector_lagrange_space(domain, degree=1)
    state = problems.second_order_state(V)
    v_test = spaces.test_function(V)

    material = elasticity.isotropic_elastic(
        young=227.5e9,
        density=2900.0,
        poisson=0.27,
        name="silicon-nitride-like isotropic elastic",
    )
    cp, cs = elasticity.estimate_elastic_wave_speeds(material)
    dx = ufl.dx(domain=domain)
    lumped = problems.LumpedMassOperator.assemble(V, material.density, measure=dx)
    integrator = fem_time.explicit.central_difference(state=state, mass=lumped)

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], length)

    tolerance = min(length / denx, height / deny) * 1.0e-6
    periodic_projection = build_top_bottom_periodic_projection(V, height, tolerance)

    ds_right, _ = fem_mesh.tagged_boundary_measure(domain, right, tag=2)
    normal = ufl.FacetNormal(domain)
    abc = absorbing.lysmer_kuhlemeyer_boundary(
        ds_right(2),
        density=material.density,
        pressure_wave_speed=cp,
        shear_wave_speed=cs,
        normal=normal,
    )

    frequency = 40.78e9
    width = 3.0 * np.pi / (2.0 * np.pi * frequency)
    dt = 0.35 * min(length / denx, height / deny) / cp
    steps = 1000
    source_amplitude = 4.9e-13
    source_pulse = amplitudes.gaussian_modulated_sine(
        amplitude=source_amplitude,
        frequency=frequency,
        width=width,
        name="source_pulse",
    )
    left_y_constraint = constraints.time_dependent_component_dirichlet(
        V,
        component=0,
        on=left,
        value=source_pulse,
    )

    out = Path(__file__).resolve().parents[1] / "examples_output" / "wave_packet_plate_2d.xdmf"
    stepper = fem_time.TimeStepper(
        total_steps=steps,
        dt=dt,
        save_every=10,
        print_every=100,
    )
    residual_form = fem.form(
        forms.stiffness_form(elasticity.stress(state.u, material), elasticity.strain(v_test), measure=dx)
        + abc.form(state.v_mid, v_test)
    )

    with fem_io.XDMFTimeSeries(out, domain) as xdmf:
        xdmf.write_fields(0.0, state.u, state.v)
        for info in stepper:
            t = info.time
            left_y_constraint.update(t)
            integrator.predict_displacement(dt)
            constraints.apply_dirichlet_bcs(state.u_next, [left_y_constraint.bc])
            apply_periodic_projection(state.u_next, periodic_projection)
            integrator.update_displacement()

            integrator.update_midstep_velocity(dt)
            apply_periodic_projection(state.v_mid, periodic_projection)

            residual = assembly.assemble_vector(residual_form)
            integrator.solve_acceleration(residual)
            residual.destroy()
            apply_periodic_projection(state.a_next, periodic_projection)
            integrator.update_velocity(dt)
            apply_periodic_projection(state.v_next, periodic_projection)
            integrator.advance_velocity_acceleration()

            if info.should_save:
                xdmf.write_fields(t, state.u, state.v)
            if info.should_print:
                periodic_err = periodic_projection_mismatch(state.u, periodic_projection)
                print_on_root(
                    comm,
                    f"step {info.index:4d}/{steps} "
                    f"t={t:.3e} max|u|={np.max(np.abs(state.u.x.array)):.3e} "
                    f"periodic_err={periodic_err:.3e}",
                )

    print_on_root(comm, f"Wave-packet plate result: {out}")


if __name__ == "__main__":
    main()
