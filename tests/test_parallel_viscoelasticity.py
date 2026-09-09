from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from mpi4py import MPI
import pytest

from agentfem import (
    benchmarks,
    fields,
    mesh,
    models,
    operators,
    results,
    solvers,
    studies,
)
from agentfem.constitutive import IsotropicGeneralizedMaxwell, isotropic_elastic


def test_harmonic_generalized_maxwell_solve_and_output_are_distributed(tmp_path):
    """Keep the harmonic solve, evidence and presentation output MPI-safe."""

    if MPI.COMM_WORLD.size != 2:
        pytest.skip("harmonic viscoelastic MPI regression requires two ranks")

    comm = MPI.COMM_WORLD
    output_root = str(tmp_path) if comm.rank == 0 else None
    output_root = Path(comm.bcast(output_root, root=0))
    length = 2.0
    traction = 3.0
    instantaneous_young = 1000.0
    frequency = 1.0 / (4.0 * np.pi)
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (length, 1.0, 1.0),
        (8, 2, 2),
        comm=comm,
        cell_type="hexahedron",
    )
    model = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
        name="parallel_harmonic_viscoelastic_bar",
    )
    displacement = model.field(fields.displacement(domain))
    material = model.material(
        IsotropicGeneralizedMaxwell.from_prony(
            instantaneous_young_modulus=instantaneous_young,
            instantaneous_poisson_ratio=0.0,
            shear_relaxation_ratios=[0.4],
            bulk_relaxation_ratios=[0.4],
            relaxation_times=[2.0],
        )
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=0.0),
        component=0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="y", value=0.0),
        component=1,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="z", value=0.0),
        component=2,
    )
    loaded_end = mesh.face(domain, axis="x", value=length)
    model.traction((traction, 0.0, 0.0), on=loaded_end)

    step = model.step(target=displacement, frequency=frequency)
    output = output_root / "harmonic_fields.xdmf"
    simulation = step.solve_result(output=output, strict_output=True)
    tip = results.average(
        step.solution_real[0], measure=loaded_end.measure
    ) + 1j * results.average(
        step.solution_imaginary[0], measure=loaded_end.measure
    )
    expected = traction * length / material.harmonic_moduli(0.5).young
    normalized = tip * instantaneous_young / (traction * length)
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.global_viscoelastic_harmonic_bar"
    )

    assert tip == pytest.approx(expected, rel=2.0e-10)
    golden.quantity("normalized_complex_end_displacement").assert_accepts(
        [normalized.real, normalized.imag]
    )
    assert simulation.quantity("dissipated_energy_per_cycle") > 0.0
    assert simulation.quantity("input_energy_per_cycle") > 0.0
    assert simulation.quantity("relative_cycle_energy_balance_error") < 1.0e-10
    assert simulation.quantity("relative_residual_norm") < 1.0e-10
    assert simulation.quantity("relative_real_block_residual_norm") < 1.0e-10
    assert simulation.quantity("relative_imaginary_block_residual_norm") < 1.0e-10
    assert set(simulation.fields) == {
        "U_REAL",
        "U_IMAG",
        "U_AMPLITUDE",
        "U_PHASE",
    }
    assert simulation.metadata["field_output"]["layout"] == (
        "scientific_xdmf_plus_single_paraview_dataset"
    )
    assert simulation.metadata["field_output"][
        "visualization_requires_extract_block"
    ] is False

    rank_evidence = comm.allgather(
        {
            "tip": (tip.real, tip.imag),
            "energy": {
                name: simulation.quantity(name)
                for name in (
                    "mean_stored_energy",
                    "mean_kinetic_energy",
                    "dissipated_energy_per_cycle",
                    "input_energy_per_cycle",
                    "cycle_energy_balance_error",
                    "relative_cycle_energy_balance_error",
                    "relative_residual_norm",
                    "relative_real_block_residual_norm",
                    "relative_imaginary_block_residual_norm",
                )
            },
            "fields": tuple(simulation.fields),
            "output": simulation.metadata["field_output"],
        }
    )
    assert all(item == rank_evidence[0] for item in rank_evidence)

    comm.barrier()
    if comm.rank == 0:
        assert output.is_file()
        assert output.with_suffix(".h5").is_file()
        paraview = simulation.artifacts["fields_paraview"]
        assert paraview.is_file()
        datasets = ET.parse(paraview).findall(".//DataSet")
        assert len(datasets) == 1
        parallel_grid = paraview.parent / datasets[0].attrib["file"]
        point_arrays = {
            item.attrib["Name"]
            for item in ET.parse(parallel_grid).findall(
                ".//PPointData/PDataArray"
            )
        }
        assert {"U", "U_IMAG", "U_AMPLITUDE", "U_PHASE"} <= point_arrays


def test_generic_direct_harmonic_sweep_is_distributed_and_monolithic():
    """Keep the generic K/M/C/F direct-solve route rank consistent."""

    if MPI.COMM_WORLD.size != 2:
        pytest.skip("generic direct harmonic MPI regression requires two ranks")

    comm = MPI.COMM_WORLD
    length = 2.0
    traction = 3.0
    young = 1000.0
    density = 1.0
    alpha = 0.4
    beta = 2.0e-3
    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (length, 1.0, 1.0),
        (12, 1, 1),
        comm=comm,
        cell_type="hexahedron",
    )
    model = models.create(
        study=studies.harmonic_solid(dimension=3),
        mesh=domain,
        name="parallel_generic_harmonic_bar",
    )
    displacement = model.field(fields.displacement(domain))
    model.material(
        isotropic_elastic(
            young=young,
            poisson=0.0,
            density=density,
        )
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="x", value=0.0),
        component=0,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="y", value=0.0),
        component=1,
    )
    model.fix(
        displacement,
        on=mesh.face(domain, axis="z", value=0.0),
        component=2,
    )
    loaded_end = mesh.face(domain, axis="x", value=length)
    model.traction((traction, 0.0, 0.0), on=loaded_end)
    stiffness = model.stiffness(displacement)
    mass = model.mass(displacement)
    damping = operators.rayleigh_damping(
        mass,
        stiffness,
        mass_coefficient=alpha,
        stiffness_coefficient=beta,
    )
    response = results.harmonic_response(
        "tip_x",
        lambda point: results.average(
            point.solution_real[0], measure=loaded_end.measure
        )
        + 1j
        * results.average(
            point.solution_imaginary[0], measure=loaded_end.measure
        ),
        unit="m",
    )
    frequencies = (0.5, 0.875, 1.25)
    step = model.step(
        target=displacement,
        K=stiffness,
        M=mass,
        C=damping,
        F=model.external_force(displacement),
        frequencies=frequencies,
        responses=(response,),
        solver_options=solvers.direct_solver(),
        progress=False,
    )
    simulation = step.solve_result()

    actual = np.asarray(simulation.histories["tip_x_REAL"].values) + 1j * np.asarray(
        simulation.histories["tip_x_IMAG"].values
    )
    expected = []
    for frequency in frequencies:
        omega = 2.0 * np.pi * frequency
        effective_young = young * (1.0 + 1j * omega * beta)
        wave_number = np.sqrt(
            density * (omega**2 - 1j * omega * alpha) / effective_young
        )
        expected.append(
            traction
            * np.tan(wave_number * length)
            / (effective_young * wave_number)
        )
    np.testing.assert_allclose(actual, expected, rtol=4.0e-3)
    assert np.max(
        simulation.histories["relative_cycle_energy_balance_error"].values
    ) < 1.0e-9
    assert np.max(simulation.histories["relative_residual_norm"].values) < 1.0e-9
    backend = step.point_step.summary()["backend_execution"]
    assert backend["matrix_layout"] == "monolithic"
    assert backend["component_residuals_available"] is False
    gathered = comm.allgather(actual)
    for value in gathered:
        np.testing.assert_allclose(value, actual, rtol=1.0e-12, atol=1.0e-14)


def test_nafems_r0016_test5h_direct_sweep_matches_with_two_ranks():
    """Promote Test 5H only when its complete sweep is rank consistent."""

    if MPI.COMM_WORLD.size != 2:
        pytest.skip("NAFEMS Test 5H MPI regression requires two ranks")

    benchmark, _simulation = benchmarks.nafems_r0016_test5h_benchmark(
        comm=MPI.COMM_WORLD
    )
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.nafems_r0016_test5h_forced_vibration"
    )

    assert benchmark.acceptable
    assert benchmark.mpi_ranks == 2
    for name in (
        "peak_frequency_hz",
        "peak_displacement_m",
        "peak_recovered_s11_pa",
    ):
        golden.quantity(name).assert_accepts(benchmark.quantities[name])
    gathered = MPI.COMM_WORLD.allgather(benchmark.quantities)
    assert all(item == gathered[0] for item in gathered)
