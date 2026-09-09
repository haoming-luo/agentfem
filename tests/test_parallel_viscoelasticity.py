from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from mpi4py import MPI
import pytest

from agentfem import benchmarks, fields, mesh, models, results, studies
from agentfem.constitutive import IsotropicGeneralizedMaxwell


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
