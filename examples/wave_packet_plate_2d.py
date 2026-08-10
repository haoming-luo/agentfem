"""Elastic wave-packet propagation through a homogeneous plate.

This example is the compact single-material companion to
``wave_packet_inclusion_2d.py``. It uses the AgentFEM model language rather
than hand-written FEniCSx forms:

- left boundary drives a Gaussian-modulated displacement source,
- top/bottom use explicit nodal periodic projection,
- right boundary uses a Lysmer-Kuhlemeyer absorbing boundary,
- time integration is explicit central difference.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

from mpi4py import MPI

SOURCE_PARENT = Path(__file__).resolve().parents[2]
if str(SOURCE_PARENT) not in sys.path:
    sys.path.insert(0, str(SOURCE_PARENT))

from agentfem import amplitudes
from agentfem import fields
from agentfem import mesh as fem_mesh
from agentfem import models
from agentfem import problems
from agentfem import studies
from agentfem.constitutive import elasticity
from agentfem.diagnostics import max_magnitude, print_on_root


def main() -> None:
    comm = MPI.COMM_WORLD

    # 1. Study: choose the analysis family and mechanical assumption.
    study = studies.dynamic_solid(
        dimension=2,
        assumption="plane_strain",
        method="explicit",
        name="wave_packet_plate",
    )

    # 2. Mesh and model: build the plate domain.
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
    model = models.create(study=study, mesh=domain, name="wave_packet_plate_model")

    # 3. Unknown field and material.
    displacement = model.field(fields.displacement(domain, degree=1))
    material = model.material(
        elasticity.isotropic_elastic(
            young=227.5e9,
            density=2900.0,
            poisson=0.27,
            name="silicon-nitride-like isotropic elastic",
        )
    )
    model.check()
    cp, cs = elasticity.estimate_elastic_wave_speeds(material)

    tolerance = min(length / cells[0], height / cells[1]) * 1.0e-6

    # 4. Boundaries and constraints: periodic top/bottom and driven left edge.
    bottom = fem_mesh.face(domain, axis="y", value=0.0, name="bottom_periodic", tag=20)
    top = fem_mesh.face(
        domain,
        axis="y",
        value=height,
        name="top_periodic",
        tag=21,
        tolerance=tolerance,
    )
    periodic = model.periodic(
        displacement,
        master=bottom,
        slave=top,
        match_axis="x",
        method="projection",
        tolerance=tolerance,
        name="top_bottom_periodic_projection",
    )

    left = fem_mesh.face(domain, axis="x", value=0.0, name="source_left", tag=1)
    frequency = 40.78e9
    width = 3.0 * math.pi / (2.0 * math.pi * frequency)
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

    # 5. Boundary model: right edge uses a viscous absorbing boundary.
    right_boundary = fem_mesh.face(
        domain,
        axis="x",
        value=length,
        name="right_absorbing",
        tag=10,
        tolerance=tolerance,
    )
    abc = model.absorbing_boundary(
        on=right_boundary,
        density=material.density,
        pressure_wave_speed=cp,
        shear_wave_speed=cs,
    )

    # 6. Explicit dynamics state, force balance, and step.
    #
    # Semi-discrete elastodynamics:
    #     M a + F_int(u) + F_abs(v_mid) = F_ext
    #
    # Here the left source is an essential displacement boundary condition, so
    # there is no external load vector. AgentFEM forms the residual
    #     R = F_int(u) + F_abs(v_mid)
    # and the explicit update uses
    #     a = -M_lumped^{-1} R
    #
    # The absorbing boundary is a weak boundary contribution, approximately
    #     F_abs = int_Gamma c_abs(v_mid, w) ds
    # with w inferred from the registered displacement unknown.
    state = problems.second_order_state(displacement)
    internal_force = model.internal_force(state.u)
    absorbing_force = model.boundary_force(abc, state.v_mid)
    residual_operator = model.force_balance(
        internal=internal_force,
        absorbing=absorbing_force,
        name="R_internal_plus_absorbing",
    )

    dt = 0.35 * min(length / cells[0], height / cells[1]) / cp
    steps = 1000

    # central_difference is Newmark with beta=0 and gamma=1/2:
    #     u_next = u + dt v + 0.5 dt^2 a
    #     v_mid  = v + 0.5 dt a
    #     v_next = v_mid + 0.5 dt a_next
    step = model.step(
        target=displacement,
        state=state,
        residual=residual_operator,
        prescribed=[source_bc],
        constraints=[periodic],
        dt=dt,
        steps=steps,
        save_every=20,
        print_every=100,
        name="wave_packet_plate_explicit_step",
    )

    # 7. Time stepping and output.
    out = Path(__file__).resolve().parents[1] / "examples_output" / "wave_packet_plate_2d.xdmf"
    print_on_root(
        comm,
        f"Wave packet through homogeneous plate: dt={dt:.3e} s, steps={steps}",
    )

    def progress_message(info, step_state):
        periodic_err = periodic.mismatch(step_state.u)
        return (
            f"step {info.index:4d}/{steps} "
            f"t={info.time:.3e} max|u|={max_magnitude(step_state.u):.3e} "
            f"periodic_err={periodic_err:.3e}"
        )

    simulation = step.solve_result(
        output=out,
        fields=(state.u, state.v),
        progress=progress_message,
        comm=comm,
    )

    print_on_root(comm, f"Wave-packet plate result: {out}")
    print_on_root(comm, simulation.format())


if __name__ == "__main__":
    main()
