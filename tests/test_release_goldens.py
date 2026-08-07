from __future__ import annotations

import numpy as np
from mpi4py import MPI

from agentfem import (
    amplitudes,
    benchmarks,
    diagnostics,
    fields,
    mesh,
    models,
    problems,
    results,
    solvers,
    studies,
    verification,
)
from agentfem.constitutive import elasticity


def _wave_release_patch() -> dict[str, float]:
    """Run a small counterpart of the public inclusion-wave demo."""

    length = 0.4
    height = 0.1
    cells = (32, 8)
    domain = mesh.rectangle(
        (0.0, 0.0),
        (length, height),
        cells,
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    study = studies.second_order_dynamics(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_strain",
    )
    model = models.create(study=study, mesh=domain, name="wave_release_patch")
    displacement = model.field(fields.displacement(domain, degree=1))
    center = np.array((0.12, 0.05))
    inclusion = mesh.disk(center, 0.025)
    regions = mesh.partition_cells(
        domain,
        matrix=~inclusion,
        stiff_inclusion=inclusion,
    )
    matrix = model.material(
        elasticity.isotropic_elastic(
            young=100.0,
            poisson=0.25,
            density=1.0,
            name="matrix",
        ),
        region=regions.matrix,
    )
    stiff = model.material(
        elasticity.isotropic_elastic(
            young=200.0,
            poisson=0.25,
            density=1.0,
            name="stiff_inclusion",
        ),
        region=regions.stiff_inclusion,
    )
    cp, cs = elasticity.estimate_elastic_wave_speeds(matrix)
    inclusion_cp, _ = elasticity.estimate_elastic_wave_speeds(stiff)
    tolerance = min(length / cells[0], height / cells[1]) * 1.0e-6
    bottom = mesh.face(domain, axis="y", value=0.0, name="bottom", tag=1)
    top = mesh.face(
        domain,
        axis="y",
        value=height,
        name="top",
        tag=2,
        tolerance=tolerance,
    )
    periodic = model.periodic(
        displacement,
        master=bottom,
        slave=top,
        match_axis="x",
        method="projection",
        tolerance=tolerance,
    )
    left = mesh.face(domain, axis="x", value=0.0, name="source", tag=3)
    source = model.fix(
        displacement,
        on=left,
        components=0,
        value=amplitudes.gaussian_modulated_sine(
            amplitude=1.0e-3,
            frequency=25.0,
            width=0.005,
            center=0.02,
        ),
    )
    right = mesh.face(
        domain,
        axis="x",
        value=length,
        name="absorbing",
        tag=4,
        tolerance=tolerance,
    )
    absorbing = model.absorbing_boundary(
        on=right,
        density=matrix.density,
        pressure_wave_speed=cp,
        shear_wave_speed=cs,
    )
    state = problems.second_order_state(displacement)
    residual = model.force_balance(
        internal=model.internal_force(state.u),
        absorbing=model.boundary_force(absorbing, state.v_mid),
    )
    dt = 0.35 * min(length / cells[0], height / cells[1]) / inclusion_cp
    total_steps = 320
    step = model.explicit_dynamics_step(
        target=displacement,
        state=state,
        residual=residual,
        prescribed=[source],
        constraints=[periodic],
        dt=dt,
        steps=total_steps,
        print_every=1,
        progress=False,
    )
    receiver = mesh.box((0.30, 0.0), (0.34, height))
    times = []
    maxima = []
    receiver_maxima = []
    periodic_errors = []

    def collect(info, current):
        times.append(info.time)
        maxima.append(diagnostics.max_magnitude(current.u))
        receiver_maxima.append(
            diagnostics.magnitude_stats(current.u, on=receiver).max
        )
        periodic_errors.append(periodic.mismatch(current.u))

    step.run(progress=collect)
    receiver_values = np.asarray(receiver_maxima)
    threshold = 1.0e-6
    arrivals = np.flatnonzero(receiver_values >= threshold)
    return {
        "matrix_pressure_wave_speed": cp,
        "matrix_shear_wave_speed": cs,
        "inclusion_pressure_wave_speed": inclusion_cp,
        "courant_number": dt * inclusion_cp / min(
            length / cells[0], height / cells[1]
        ),
        "maximum_displacement": float(np.max(maxima)),
        "receiver_peak_displacement": float(np.max(receiver_values)),
        "receiver_arrival_time": (
            float(times[arrivals[0]]) if len(arrivals) else float("inf")
        ),
        "maximum_periodic_mismatch": float(np.max(periodic_errors)),
    }


def test_wave_release_patch_matches_versioned_golden():
    actual = _wave_release_patch()
    golden = benchmarks.golden_benchmark("agentfem.benchmark.wave_release")

    assert all(golden.verify(actual).values())
    assert actual["maximum_periodic_mismatch"] < 1.0e-12


def test_golden_contracts_become_explicit_release_claims():
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.creep_hot_wall_release"
    )
    actual = {
        item.name: item.expected
        for item in golden.quantities
    }

    claims = golden.claims(actual)

    assert len(claims) == 3
    assert all(item.status == "passed" for item in claims)
    assert all(item.kind == "verification" for item in claims)
    assert all(
        item.evidence["reference_version"] == "q1-6x10-heat4-kr40-1"
        for item in claims
    )


def test_c3d10h_periodic_cell_golden_is_machine_readable():
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.c3d10h_periodic_cell"
    )

    assert golden.reference_version == "c3d10h-periodic-nu0499-stretch1001-1"
    assert {item.name for item in golden.quantities} == {
        "homogenized_first_piola_stress",
        "solid_reference_fraction",
        "minimum_quadrature_J",
        "periodic_equation_max_error",
    }


def _oriented_cantilever(*, rotated: bool) -> tuple[object, float]:
    if rotated:
        lower, upper, cells = (-0.2, 0.0), (0.0, 1.0), (4, 20)
        fixed_axis, fixed_value = "y", 0.0
        loaded_axis, loaded_value = "y", 1.0
        traction = (1.0, 0.0)
    else:
        lower, upper, cells = (0.0, 0.0), (1.0, 0.2), (20, 4)
        fixed_axis, fixed_value = "x", 0.0
        loaded_axis, loaded_value = "x", 1.0
        traction = (0.0, -1.0)
    domain = mesh.rectangle(
        lower,
        upper,
        cells,
        comm=MPI.COMM_SELF,
        cell_type="quadrilateral",
    )
    study = studies.linear_static(
        physics="solid_mechanics",
        dimension=2,
        assumption="plane_stress",
    )
    model = models.create(study=study, mesh=domain, name="orientation_cliff")
    displacement = model.field(fields.displacement(domain, degree=2))
    model.material(
        elasticity.isotropic_elastic(
            young=1000.0,
            poisson=0.3,
            density=1.0,
        )
    )
    fixed = mesh.face(
        domain,
        axis=fixed_axis,
        value=fixed_value,
        name="fixed",
        tag=1,
    )
    loaded = mesh.face(
        domain,
        axis=loaded_axis,
        value=loaded_value,
        name="loaded",
        tag=2,
    )
    model.fix(displacement, on=fixed, value=0.0)
    model.traction(traction, on=loaded)
    step = model.step(
        target=displacement,
        solver_options=solvers.direct_solver(),
    )
    simulation = step.solve_result()
    maximum = diagnostics.max_magnitude(displacement.value)
    simulation.add_quantity("maximum_displacement", maximum)
    return simulation, maximum


def test_cantilever_rotation_is_a_verified_metamorphic_contract():
    reference, original = _oriented_cantilever(rotated=False)
    rotated, transformed = _oriented_cantilever(rotated=True)
    claim = verification.VerificationClaim.compare(
        name="cantilever_rotation_covariance",
        observable="maximum_displacement",
        actual=transformed,
        expected=original,
        reference="90-degree rigid rotation of geometry, mesh, load, and support",
        relative_tolerance=2.0e-10,
        absolute_tolerance=1.0e-12,
        validity_domain="isotropic material and identically rotated discretization",
        evidence={
            "original_mesh": [20, 4],
            "rotated_mesh": [4, 20],
            "purpose": "detect section/load-axis interpretation errors",
        },
    )
    rotated.add_verification(verification.report(claim))

    assert reference.trust_level == "computed"
    assert rotated.trust_level == "verified"
    assert claim.status == "passed"
