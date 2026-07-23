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

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

SOURCE_PARENT = Path(__file__).resolve().parents[2]
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from agentfem import assembly
from agentfem import boundary
from agentfem import forms
from agentfem import io as fem_io
from agentfem import mesh as fem_mesh
from agentfem import problems
from agentfem import runtime
from agentfem import spaces
from agentfem.boundary_models import absorbing
from agentfem.constitutive import elasticity


def wave_packet(t: float, amplitude: float, frequency: float, width: float) -> float:
    omega = 2.0 * np.pi * frequency
    return amplitude * np.sin(omega * t) * np.exp(-((t - 3.0 * width) / width) ** 2)


def periodic_pairs_by_y(domain, V, bottom_marker, top_marker, *, tol=1.0e-12):
    bottom_dofs = fem.locate_dofs_geometrical(V, bottom_marker)
    top_dofs = fem.locate_dofs_geometrical(V, top_marker)
    coords = V.tabulate_dof_coordinates()
    bottom = sorted(bottom_dofs, key=lambda dof: coords[dof, 0])
    top = sorted(top_dofs, key=lambda dof: coords[dof, 0])
    if len(bottom) != len(top):
        raise RuntimeError("Periodic projection requires matching top/bottom dofs.")
    pairs = []
    for b, t in zip(bottom, top):
        if not np.isclose(coords[b, 0], coords[t, 0], atol=tol, rtol=0.0):
            raise RuntimeError("Top/bottom periodic dofs are not aligned in x.")
        pairs.append((b, t))
    return pairs


def project_periodic(function, pairs) -> None:
    values = function.x.array
    for bottom, top in pairs:
        avg = 0.5 * (values[bottom] + values[top])
        values[bottom] = avg
        values[top] = avg
    function.x.scatter_forward()


def main() -> None:
    comm = MPI.COMM_WORLD
    length = 1.2e-6
    height = 0.24e-6
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([length, height])],
        [120, 24],
        cell_type=mesh.CellType.quadrilateral,
    )

    V = spaces.vector_lagrange_space(domain, degree=1)
    state = problems.SecondOrderDynamicsState.create(V)
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

    def left(x):
        return np.isclose(x[0], 0.0)

    def right(x):
        return np.isclose(x[0], length)

    def bottom(x):
        return np.isclose(x[1], 0.0)

    def top(x):
        return np.isclose(x[1], height)

    left_y, left_y_bc = boundary.component_dirichlet_bc(V, 1, left, value=0.0)
    periodic_pairs = periodic_pairs_by_y(domain, V, bottom, top, tol=1.0e-13)

    ds_right, _ = fem_mesh.tagged_boundary_measure(domain, right, tag=2)
    normal = ufl.FacetNormal(domain)
    abc = absorbing.lysmer_kuhlemeyer_boundary(
        ds_right(2),
        density=material.density,
        pressure_wave_speed=cp,
        shear_wave_speed=cs,
        normal=normal,
    )

    frequency = 90.78e9
    width = 3.0 * np.pi / (2.0 * np.pi * frequency)
    dt = 0.35 * min(length / 120, height / 24) / cp
    steps = 200
    source_amplitude = 4.9e-13

    out = Path(__file__).resolve().parents[1] / "examples_output" / "wave_packet_plate_2d.xdmf"
    stepper = runtime.TimeStepper(
        total_steps=steps,
        dt=dt,
        save_every=20,
        print_every=25,
    )

    with fem_io.XDMFTimeSeries(out, domain) as xdmf:
        xdmf.write_fields(0.0, state.u, state.v)
        for info in stepper:
            t = info.time
            left_y.value = PETSc.ScalarType(
                wave_packet(t, source_amplitude, frequency, width)
            )
            state.predict_displacement(dt)
            boundary.apply_dirichlet_bcs(state.u_next, [left_y_bc])
            project_periodic(state.u_next, periodic_pairs)

            stress = material.sigma(state.u_next)
            residual_form = fem.form(
                forms.stiffness_form(stress, elasticity.strain(v_test), measure=dx)
                + abc.form(state.v, v_test)
            )
            residual = assembly.assemble_vector(residual_form)
            state.set_acceleration_from_residual(residual, lumped.inv_mass)
            residual.destroy()
            boundary.apply_dirichlet_bcs(state.a_next, [left_y_bc])
            project_periodic(state.a_next, periodic_pairs)
            state.correct_velocity(dt)
            boundary.apply_dirichlet_bcs(state.v_next, [left_y_bc])
            project_periodic(state.v_next, periodic_pairs)
            state.accept_step()

            if info.should_save:
                xdmf.write_fields(t, state.u, state.v)
            if info.should_print and comm.rank == 0:
                print(
                    f"step {info.index:4d}/{steps} "
                    f"t={t:.3e} max|u|={np.max(np.abs(state.u.x.array)):.3e}",
                    flush=True,
                )

    if comm.rank == 0:
        print(f"Wave-packet plate result: {out}")


if __name__ == "__main__":
    main()
