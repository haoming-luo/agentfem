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

import os
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import amplitudes
from agentfem import diagnostics
from agentfem import fields
from agentfem import mesh as fem_mesh
from agentfem import models
from agentfem import problems
from agentfem import studies
from agentfem.constitutive import elasticity
from agentfem.diagnostics import max_magnitude, print_on_root


def main() -> None:
    comm = MPI.COMM_WORLD
    smoke = os.environ.get("AGENTFEM_RELEASE_SMOKE") == "1"

    # 1. Study: choose the analysis family and mechanical assumption.
    study = studies.dynamic_solid(
        dimension=2,
        assumption="plane_strain",
        method="explicit",
        name="wave_packet_stiff_inclusion",
    )

    # 2. Mesh and model: build the plate domain.
    length = 1.2e-6
    height = 0.24e-6
    cells = (48, 12) if smoke else (240, 48)
    domain = fem_mesh.rectangle(
        lower=(0.0, 0.0),
        upper=(length, height),
        cells=cells,
        comm=comm,
        cell_type="quadrilateral",
    )
    model = models.create(study=study, mesh=domain, name="wave_packet_inclusion_model")

    # 3. Unknown field: displacement is the primary dynamics unknown.
    displacement = model.field(fields.displacement(domain, degree=1))

    # 4. Material regions: split cells into matrix and circular inclusion.
    center = np.array((0.25 * length, 0.50 * height))
    radius = 0.15 * height
    inclusion_selector = fem_mesh.disk(center=center, radius=radius)
    inclusion_core = fem_mesh.disk(center=center, radius=0.90 * radius)
    regions = fem_mesh.partition_cells(
        domain,
        matrix=~inclusion_selector,
        stiff_inclusion=inclusion_selector,
    )
    material_id = regions.field("MaterialId")

    # 5. Materials: the inclusion is twice as stiff as the matrix.
    matrix_material = model.material(
        elasticity.isotropic_elastic(
            young=227.5e9,
            density=2900.0,
            poisson=0.27,
            name="matrix isotropic elastic",
        ),
        region=regions.matrix,
    )
    inclusion_material = model.material(
        elasticity.isotropic_elastic(
            young=2.0 * matrix_material.young,
            density=matrix_material.density,
            poisson=matrix_material.poisson,
            name="2x Young inclusion isotropic elastic",
        ),
        region=regions.stiff_inclusion,
    )
    model.check()

    matrix_cp, matrix_cs = elasticity.estimate_elastic_wave_speeds(matrix_material)
    inclusion_cp, _ = elasticity.estimate_elastic_wave_speeds(inclusion_material)

    tolerance = min(length / cells[0], height / cells[1]) * 1.0e-6

    # 6. Boundaries and constraints: periodic top/bottom and driven left edge.
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

    # 7. Boundary model: right edge uses a viscous absorbing boundary.
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
        density=matrix_material.density,
        pressure_wave_speed=matrix_cp,
        shear_wave_speed=matrix_cs,
    )

    # 8. Explicit dynamics state, force balance, and step.
    state = problems.second_order_state(displacement)

    dt = 0.35 * min(length / cells[0], height / cells[1]) / max(matrix_cp, inclusion_cp)
    steps = 80 if smoke else 2000

    internal_force = model.internal_force(state.u)
    absorbing_force = model.boundary_force(abc, state.v_mid)
    residual_operator = model.force_balance(
        internal=internal_force,
        absorbing=absorbing_force,
        name="R_internal_plus_absorbing",
    )
    step = model.step(
        target=displacement,
        state=state,
        residual=residual_operator,
        prescribed=[source_bc],
        constraints=[periodic],
        dt=dt,
        steps=steps,
        save_every=10,
        name="wave_packet_explicit_step",
    )

    # 9. Time stepping and output.
    out = Path(__file__).resolve().parents[1] / "examples_output" / "wave_packet_inclusion_2d.xdmf"

    print_on_root(
        comm,
        "Wave packet through stiff inclusion: "
        f"E_inclusion/E_matrix={inclusion_material.young / matrix_material.young:.1f}, "
        f"dt={dt:.3e} s, steps={steps}",
    )

    def progress_message(info, step_state):
        periodic_err = periodic.mismatch(step_state.u)
        inclusion_stats = diagnostics.magnitude_stats(step_state.u, on=inclusion_core)
        return (
            f"step {info.index:4d}/{steps} "
            f"t={info.time:.3e} max|u|={max_magnitude(step_state.u):.3e} "
            f"inclusion_max|u|={inclusion_stats.max:.3e} "
            f"inclusion_mean|u|={inclusion_stats.mean:.3e} "
            f"periodic_err={periodic_err:.3e}"
        )

    simulation = step.solve_result(
        output=out,
        fields=(state.u, state.v, material_id),
        progress=progress_message,
        comm=comm,
    )

    print_on_root(comm, f"Wave-packet inclusion result: {out}")
    print_on_root(comm, simulation.format())


if __name__ == "__main__":
    main()
