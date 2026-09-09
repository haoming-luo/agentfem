"""External direct steady-state forced-vibration verification."""

from __future__ import annotations

from dataclasses import replace
import os

from mpi4py import MPI
import numpy as np
import pytest

from agentfem import benchmarks, provenance


def test_nafems_r0016_test5h_rejects_changed_public_frequency_axis():
    with pytest.raises(ValueError, match="50-point"):
        benchmarks.nafems_r0016_test5h_benchmark(
            frequencies=np.linspace(40.0, 45.0, 49),
            comm=MPI.COMM_SELF,
        )


@pytest.mark.parametrize("cells", ((4, 2, 1), (5, 3, 1), (5, 2, 2)))
def test_nafems_r0016_test5h_rejects_ambiguous_sampling_or_support_mesh(cells):
    with pytest.raises(ValueError, match="must be (odd|even)"):
        benchmarks.nafems_r0016_test5h_benchmark(
            cells=cells,
            comm=MPI.COMM_SELF,
        )


def test_nafems_r0016_test5h_public_forced_vibration_benchmark():
    benchmark, simulation = benchmarks.nafems_r0016_test5h_benchmark(
        comm=MPI.COMM_SELF,
    )
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.nafems_r0016_test5h_forced_vibration"
    )

    assert benchmark.acceptable
    assert benchmark.quantities["relative_peak_frequency_error"] < 0.01
    assert benchmark.quantities["relative_peak_displacement_error"] < 0.02
    assert benchmark.quantities["relative_peak_stress_error"] < 0.03
    assert benchmark.quantities["maximum_relative_residual_norm"] < 1.0e-8
    assert benchmark.quantities["maximum_relative_cycle_energy_balance_error"] < 1.0e-8
    assert benchmark.quantities["relative_loaded_area_error"] < 1.0e-10
    assert benchmark.quantities["loaded_area_m2"] == pytest.approx(20.0)
    assert benchmark.quantities["load_resultant_amplitude_n"] == pytest.approx(10.0e6)
    assert benchmark.quantities[
        "rayleigh_damping_ratio_at_reference_peak"
    ] == pytest.approx(0.0199964, rel=2.0e-5)

    # The card owns AgentFEM's regression values for this declared mesh. They
    # remain distinct from the independently asserted public NAFEMS values.
    for name in (
        "peak_frequency_hz",
        "peak_displacement_m",
        "peak_recovered_s11_pa",
    ):
        golden.quantity(name).assert_accepts(benchmark.quantities[name])
    assert benchmark.quantities["peak_recovered_s11_frequency_hz"] == pytest.approx(
        42.55102040816327, abs=1.0e-12
    )

    assert benchmark.extraction["tolerance_authority"].startswith("AgentFEM")
    assert "no Abaqus element identity" in benchmark.extraction["discretization"]
    assert benchmark.extraction["frequency_point_count"] == 50
    assert benchmark.extraction["frequency_axis"][0] == pytest.approx(40.0)
    assert benchmark.extraction["frequency_axis"][-1] == pytest.approx(45.0)
    assert 0 < benchmark.extraction["displacement_peak_index"] < 49
    assert 0 < benchmark.extraction["stress_peak_index"] < 49
    assert benchmark.extraction["peak_index_separation_bins"] <= 1
    assert (
        benchmark.quantities["peak_index_separation_bins"]
        == (benchmark.extraction["peak_index_separation_bins"])
    )
    assert benchmark.quantities["peak_frequency_separation_hz"] <= 5.0 / 49.0 + 1.0e-14
    assert benchmark.extraction["reference_values"] == {
        "peak_frequency_hz": 42.65,
        "peak_displacement_m": 13.45e-3,
        "peak_extreme_fibre_s11_pa": 241.9e6,
    }
    assert benchmark.extraction["phasor_convention"] == "exp(+i*omega*t)"
    assert "r0016" in benchmark.extraction["source"]["nafems_r0016"]
    assert simulation.metadata["external_benchmark"]["acceptable"] is True


def test_nafems_r0016_test5h_spatial_refinement_certificate_from_frozen_evidence():
    levels = _frozen_refinement_levels()
    certificate = benchmarks.certify_nafems_r0016_test5h_spatial_convergence(levels)
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.nafems_r0016_test5h_spatial_refinement"
    )

    assert certificate.accepted
    assert certificate.successive_changes_decrease
    assert certificate.all_levels_pass_external_comparison
    assert certificate.cells == ((5, 2, 1), (9, 4, 3), (13, 6, 5))
    assert certificate.as_dict()["observed_order"] is None
    assert certificate.as_dict()["uniform_refinement"] is False
    assert certificate.peak_index_separations_bins == (0, 0, 0)
    assert certificate.as_dict()["residual_tolerance"] == 1.0e-8
    assert certificate.as_dict()["energy_tolerance"] == 1.0e-8
    assert certificate.as_dict()["maximum_peak_index_separation_bins"] == 1
    golden.quantity("peak_frequencies_hz").assert_accepts(
        certificate.peak_frequencies_hz
    )
    golden.quantity("peak_displacements_m").assert_accepts(
        certificate.peak_displacements_m
    )
    golden.quantity("peak_recovered_s11_pa").assert_accepts(
        certificate.peak_recovered_s11_pa
    )
    golden.quantity("finest_pair_relative_changes").assert_accepts(
        tuple(
            certificate.finest_pair_relative_changes[name]
            for name in (
                "peak_frequency_hz",
                "peak_displacement_m",
                "peak_recovered_s11_pa",
            )
        )
    )


def test_nafems_r0016_test5h_spatial_refinement_certificate_fails_closed():
    levels = _frozen_refinement_levels()

    with pytest.raises(ValueError, match="At least three"):
        benchmarks.certify_nafems_r0016_test5h_spatial_convergence(levels[:2])

    changed = list(levels)
    changed[-1] = replace(
        changed[-1],
        extraction={
            **changed[-1].extraction,
            "refinement_contract_sha256": "sha256:changed",
        },
    )
    with pytest.raises(ValueError, match="fingerprint does not match"):
        benchmarks.certify_nafems_r0016_test5h_spatial_convergence(changed)

    strict = benchmarks.certify_nafems_r0016_test5h_spatial_convergence(
        levels,
        refinement_relative_tolerances={"peak_recovered_s11_pa": 0.001},
    )
    assert not strict.accepted


def test_nafems_r0016_test5h_certificate_recomputes_scientific_evidence():
    levels = _frozen_refinement_levels()

    empty_tolerances = list(levels)
    empty_tolerances[1] = replace(empty_tolerances[1], tolerances={})
    with pytest.raises(ValueError, match="tolerance schema"):
        benchmarks.certify_nafems_r0016_test5h_spatial_convergence(empty_tolerances)

    relaxed_tolerances = list(levels)
    relaxed_tolerances[1] = replace(
        relaxed_tolerances[1],
        tolerances={
            **relaxed_tolerances[1].tolerances,
            "relative_peak_stress_error": 1.0,
        },
    )
    with pytest.raises(ValueError, match="changed frozen.*tolerance"):
        benchmarks.certify_nafems_r0016_test5h_spatial_convergence(relaxed_tolerances)

    inconsistent_error = list(levels)
    inconsistent_error[1] = replace(
        inconsistent_error[1],
        quantities={
            **inconsistent_error[1].quantities,
            "relative_peak_displacement_error": 0.0,
        },
    )
    with pytest.raises(ValueError, match="derived.*inconsistent"):
        benchmarks.certify_nafems_r0016_test5h_spatial_convergence(inconsistent_error)

    false_size = list(levels)
    false_size[1] = replace(
        false_size[1],
        extraction={
            **false_size[1].extraction,
            "characteristic_cell_size_m": 0.01,
        },
    )
    with pytest.raises(ValueError, match="characteristic cell size"):
        benchmarks.certify_nafems_r0016_test5h_spatial_convergence(false_size)

    false_width = list(levels)
    false_width[1] = replace(
        false_width[1],
        extraction={**false_width[1].extraction, "cell_widths_m": (0.1, 0.1, 0.1)},
    )
    with pytest.raises(ValueError, match="cell widths disagree"):
        benchmarks.certify_nafems_r0016_test5h_spatial_convergence(false_width)


def test_nafems_r0016_test5h_certificate_rejects_mutated_contract_and_nonfinite():
    levels = _frozen_refinement_levels()

    mutated_contract = list(levels)
    contract = {
        **mutated_contract[1].extraction["refinement_contract"],
        "load": "mutated after observing the result",
    }
    mutated_contract[1] = replace(
        mutated_contract[1],
        extraction={
            **mutated_contract[1].extraction,
            "refinement_contract": contract,
            "refinement_contract_sha256": provenance.content_fingerprint(contract),
        },
    )
    with pytest.raises(ValueError, match="changed the frozen.*model contract"):
        benchmarks.certify_nafems_r0016_test5h_spatial_convergence(mutated_contract)

    nonfinite = list(levels)
    nonfinite[1] = replace(
        nonfinite[1],
        quantities={
            **nonfinite[1].quantities,
            "maximum_relative_residual_norm": np.nan,
        },
    )
    with pytest.raises(ValueError, match="non-finite"):
        benchmarks.certify_nafems_r0016_test5h_spatial_convergence(nonfinite)

    with pytest.raises(ValueError, match="residual_tolerance"):
        benchmarks.certify_nafems_r0016_test5h_spatial_convergence(
            levels, residual_tolerance=np.inf
        )
    with pytest.raises(ValueError, match="energy_tolerance"):
        benchmarks.certify_nafems_r0016_test5h_spatial_convergence(
            levels, energy_tolerance=np.nan
        )


def test_nafems_r0016_test5h_certificate_rejects_separated_peak_semantics():
    levels = _frozen_refinement_levels()
    separated = list(levels)
    extraction = dict(separated[1].extraction)
    stress_index = extraction["displacement_peak_index"] + 2
    extraction.update(
        {
            "stress_peak_index": stress_index,
            "peak_index_separation_bins": 2,
        }
    )
    axis = extraction["frequency_axis"]
    quantities = {
        **separated[1].quantities,
        "peak_recovered_s11_frequency_hz": axis[stress_index],
        "peak_index_separation_bins": 2,
        "peak_frequency_separation_hz": abs(
            axis[stress_index] - separated[1].quantities["peak_frequency_hz"]
        ),
    }
    separated[1] = replace(separated[1], extraction=extraction, quantities=quantities)

    with pytest.raises(ValueError, match="more than one bin"):
        benchmarks.certify_nafems_r0016_test5h_spatial_convergence(separated)


@pytest.mark.skipif(
    os.environ.get("AGENTFEM_REQUIRE_TEST5H_REFINEMENT") != "1",
    reason="The complete three-level Test 5H refinement is an opt-in science gate.",
)
def test_nafems_r0016_test5h_spatial_refinement_execution():
    certificate, simulations = benchmarks.nafems_r0016_test5h_spatial_convergence(
        comm=MPI.COMM_SELF
    )

    assert certificate.accepted
    assert len(simulations) == 3
    assert all(
        simulation.metadata["spatial_refinement_certificate"]["accepted"]
        for simulation in simulations
    )


def _frozen_refinement_levels():
    golden = benchmarks.golden_benchmark(
        "agentfem.benchmark.nafems_r0016_test5h_spatial_refinement"
    )
    frequencies = tuple(golden.quantity("peak_frequencies_hz").expected)
    displacements = tuple(golden.quantity("peak_displacements_m").expected)
    stresses = tuple(golden.quantity("peak_recovered_s11_pa").expected)
    cells = ((5, 2, 1), (9, 4, 3), (13, 6, 5))
    axis = np.linspace(40.0, 45.0, 50).tolist()
    contract = {
        "geometry_m": (10.0, 2.0, 2.0),
        "element": "complete quadratic Q2 hexahedron",
        "material": {
            "young_pa": 200.0e9,
            "poisson": 0.3,
            "density_kg_m3": 8000.0,
        },
        "rayleigh_damping": {
            "mass_coefficient_per_s": 5.36,
            "stiffness_coefficient_s": 7.46e-5,
        },
        "load": "-0.5 MPa pressure amplitude on the complete top face",
        "support": "published solid-model Test 5H support contract",
        "frequency_axis_hz": axis,
        "response_point_m": (5.0, 2.0, 1.0),
        "stress_recovery": "continuous-P1 global L2 recovery",
        "phasor_convention": "exp(+i*omega*t)",
    }
    contract_fingerprint = provenance.content_fingerprint(contract)
    axis_fingerprint = provenance.content_fingerprint(axis)
    residuals = (2.4e-11, 1.1e-10, 7.63e-10)
    energy_errors = (1.3e-11, 9.12e-11, 7.86e-10)
    levels = []
    for level_cells, frequency, displacement, stress, residual, energy in zip(
        cells,
        frequencies,
        displacements,
        stresses,
        residuals,
        energy_errors,
    ):
        widths = (
            10.0 / level_cells[0],
            2.0 / level_cells[1],
            2.0 / level_cells[2],
        )
        peak_index = int(np.argmin(np.abs(np.asarray(axis) - frequency)))
        quantities = {
            "relative_peak_frequency_error": abs(frequency - 42.65) / 42.65,
            "relative_peak_displacement_error": (
                abs(displacement - 13.45e-3) / 13.45e-3
            ),
            "relative_peak_stress_error": abs(stress - 241.9e6) / 241.9e6,
            "maximum_relative_residual_norm": residual,
            "maximum_relative_cycle_energy_balance_error": energy,
            "relative_loaded_area_error": 0.0,
            "peak_frequency_hz": frequency,
            "peak_displacement_m": displacement,
            "peak_recovered_s11_pa": stress,
            "peak_recovered_s11_frequency_hz": frequency,
            "peak_index_separation_bins": 0,
            "peak_frequency_separation_hz": 0.0,
            "loaded_area_m2": 20.0,
            "load_resultant_amplitude_n": 10.0e6,
        }
        levels.append(
            benchmarks.ForcedVibrationBenchmark(
                name="NAFEMS R0016 Test 5H forced vibration",
                reference="frozen public Test 5H refinement evidence",
                mpi_ranks=1,
                quantities=quantities,
                tolerances={
                    "relative_peak_frequency_error": 0.01,
                    "relative_peak_displacement_error": 0.02,
                    "relative_peak_stress_error": 0.03,
                    "maximum_relative_residual_norm": 1.0e-8,
                    "maximum_relative_cycle_energy_balance_error": 1.0e-8,
                    "relative_loaded_area_error": 1.0e-10,
                    "peak_index_separation_bins": 1.0,
                },
                extraction={
                    "cells": level_cells,
                    "cell_widths_m": widths,
                    "characteristic_cell_size_m": float(np.linalg.norm(widths)),
                    "frequency_axis": axis,
                    "frequency_axis_sha256": axis_fingerprint,
                    "displacement_peak_index": peak_index,
                    "stress_peak_index": peak_index,
                    "peak_index_separation_bins": 0,
                    "maximum_peak_index_separation_bins": 1,
                    "comparison_stress_recovery": ("continuous-P1 global L2 recovery"),
                    "refinement_contract": contract,
                    "refinement_contract_sha256": contract_fingerprint,
                    "reference_values": {
                        "peak_frequency_hz": 42.65,
                        "peak_displacement_m": 13.45e-3,
                        "peak_extreme_fibre_s11_pa": 241.9e6,
                    },
                },
            )
        )
    return tuple(levels)
