"""Public direct steady-state harmonic-response benchmarks.

The NAFEMS R0016 Test 5H beam is a three-dimensional bending problem with
Rayleigh damping.  It exercises the generic ``K/M/C/F`` operator route; it is
not a generalized-Maxwell fit and it is not a modal-superposition analysis.
The implementation freezes the public physical problem while naming
AgentFEM's own Q2-hexahedral discretization and stress-recovery semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from mpi4py import MPI
import numpy as np

from .. import fields, mesh, models, operators, results, solvers, studies, verification
from ..provenance import content_fingerprint
from ..constitutive import elasticity


NAFEMS_R0016_URL = "https://www.nafems.org/publications/resource_center/r0016/"
ABAQUS_TEST5H_URL = (
    "https://docs.software.vt.edu/abaqusv2025/English/"
    "SIMACAEBMKRefMap/simabmk-c-forcedvibrationtest5h.htm"
)
ABAQUS_C3D10_DECK_SHA256 = (
    "4e5b52c3cf6f4ef391e8ba5a5de301c3d8013d716a611fa0d7e24f6fc9ed0a79"
)
ABAQUS_C3D27_DECK_SHA256 = (
    "14a50519f66b27f5d144d88118ad0463e970d08ed5103b6cf15c6535d7bf473b"
)
_TEST5H_NAME = "NAFEMS R0016 Test 5H forced vibration"
_TEST5H_GEOMETRY_M = (10.0, 2.0, 2.0)
_TEST5H_REFERENCE_VALUES = {
    "peak_frequency_hz": 42.65,
    "peak_displacement_m": 13.45e-3,
    "peak_extreme_fibre_s11_pa": 241.9e6,
}
_TEST5H_EXPECTED_LOADED_AREA_M2 = 20.0
_TEST5H_PRESSURE_AMPLITUDE_PA = 0.5e6
_TEST5H_MAXIMUM_PEAK_INDEX_SEPARATION_BINS = 1
_TEST5H_EXTERNAL_TOLERANCES = {
    "relative_peak_frequency_error": 0.01,
    "relative_peak_displacement_error": 0.02,
    "relative_peak_stress_error": 0.03,
    "maximum_relative_residual_norm": 1.0e-8,
    "maximum_relative_cycle_energy_balance_error": 1.0e-8,
    "relative_loaded_area_error": 1.0e-10,
    "peak_index_separation_bins": float(_TEST5H_MAXIMUM_PEAK_INDEX_SEPARATION_BINS),
}


def _test5h_frequency_axis() -> np.ndarray:
    return np.linspace(40.0, 45.0, 50)


def _test5h_refinement_contract(axis) -> dict[str, object]:
    """Return the one physical/extraction contract shared by every mesh."""

    return {
        "geometry_m": _TEST5H_GEOMETRY_M,
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
        "frequency_axis_hz": np.asarray(axis, dtype=float).tolist(),
        "response_point_m": (5.0, 2.0, 1.0),
        "stress_recovery": "continuous-P1 global L2 recovery",
        "phasor_convention": "exp(+i*omega*t)",
    }


@dataclass(frozen=True)
class ForcedVibrationBenchmark:
    """Rank-independent evidence for one public forced-vibration problem."""

    name: str
    reference: str
    mpi_ranks: int
    quantities: dict[str, float]
    tolerances: dict[str, float]
    extraction: dict[str, object]

    @property
    def acceptable(self) -> bool:
        return all(
            np.isfinite(self.quantities[name]) and self.quantities[name] <= limit
            for name, limit in self.tolerances.items()
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "reference": self.reference,
            "mpi_ranks": self.mpi_ranks,
            "acceptable": self.acceptable,
            "quantities": dict(self.quantities),
            "tolerances": dict(self.tolerances),
            "extraction": dict(self.extraction),
        }


@dataclass(frozen=True)
class ForcedVibrationConvergenceCertificate:
    """Spatial-refinement stability evidence for a harmonic benchmark.

    The certificate compares the final two of at least three successively
    refined meshes.  It intentionally reports no observed order: the public
    Test 5H frequency axis is discrete and the supported structured meshes do
    not form one constant-ratio refinement family.
    """

    benchmark_name: str
    cells: tuple[tuple[int, int, int], ...]
    cell_widths_m: tuple[tuple[float, float, float], ...]
    characteristic_sizes_m: tuple[float, ...]
    peak_frequencies_hz: tuple[float, ...]
    peak_displacements_m: tuple[float, ...]
    peak_recovered_s11_pa: tuple[float, ...]
    successive_relative_changes: dict[str, tuple[float, ...]]
    finest_pair_relative_changes: dict[str, float]
    refinement_relative_tolerances: dict[str, float]
    maximum_relative_residual_norms: tuple[float, ...]
    maximum_relative_cycle_energy_balance_errors: tuple[float, ...]
    peak_index_separations_bins: tuple[int, ...]
    external_comparison_tolerances: dict[str, float]
    residual_tolerance: float
    energy_tolerance: float
    frequency_axis_sha256: str
    model_contract_sha256: str
    successive_changes_decrease: bool
    all_levels_pass_external_comparison: bool
    accepted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": ("agentfem.forced-vibration-spatial-refinement-certificate.v1"),
            "benchmark_name": self.benchmark_name,
            "cells": [list(level) for level in self.cells],
            "cell_widths_m": [list(level) for level in self.cell_widths_m],
            "characteristic_size_definition": "Euclidean cell-box diagonal",
            "characteristic_sizes_m": list(self.characteristic_sizes_m),
            "peak_frequencies_hz": list(self.peak_frequencies_hz),
            "peak_displacements_m": list(self.peak_displacements_m),
            "peak_recovered_s11_pa": list(self.peak_recovered_s11_pa),
            "successive_relative_changes": {
                name: list(values)
                for name, values in self.successive_relative_changes.items()
            },
            "finest_pair_relative_changes": dict(self.finest_pair_relative_changes),
            "refinement_relative_tolerances": dict(self.refinement_relative_tolerances),
            "maximum_relative_residual_norms": list(
                self.maximum_relative_residual_norms
            ),
            "maximum_relative_cycle_energy_balance_errors": list(
                self.maximum_relative_cycle_energy_balance_errors
            ),
            "peak_index_separations_bins": list(self.peak_index_separations_bins),
            "maximum_peak_index_separation_bins": (
                _TEST5H_MAXIMUM_PEAK_INDEX_SEPARATION_BINS
            ),
            "external_comparison_tolerances": dict(self.external_comparison_tolerances),
            "residual_tolerance": self.residual_tolerance,
            "energy_tolerance": self.energy_tolerance,
            "frequency_axis_sha256": self.frequency_axis_sha256,
            "model_contract_sha256": self.model_contract_sha256,
            "uniform_refinement": False,
            "successive_changes_decrease": self.successive_changes_decrease,
            "all_levels_pass_external_comparison": (
                self.all_levels_pass_external_comparison
            ),
            "accepted": self.accepted,
            "claim": "peak_observable_spatial_refinement_stability",
            "scope": (
                "last-pair stability of the discrete Test 5H peak frequency, "
                "displacement and declared recovered stress"
            ),
            "observed_order": None,
            "limitations": (
                "No constant refinement ratio is assumed.",
                "The peak frequency is selected on the frozen 50-point axis.",
                "Acceptance is an AgentFEM gate, not a NAFEMS tolerance.",
                "No GCI, continuum-error estimate or asymptotic range is claimed.",
            ),
        }


@dataclass
class _PreparedStressSampler:
    """Recover one harmonic stress trace without rebuilding its mass solve."""

    material: object
    study: object
    domain: object
    point: tuple[float, float, float]
    component: tuple[int, int]
    family: str
    degree: int
    name: str
    _real: object | None = None
    _imaginary: object | None = None

    def __call__(self, step) -> complex:
        if self._real is None or self._imaginary is None:
            row, column = (int(value) for value in self.component)
            self._real = results.prepare_projection(
                elasticity.stress(step.solution_real, self.material, study=self.study)[
                    row, column
                ],
                domain=self.domain,
                family=self.family,
                degree=self.degree,
                name=f"{self.name}_REAL_SAMPLE",
            )
            self._imaginary = results.prepare_projection(
                elasticity.stress(
                    step.solution_imaginary, self.material, study=self.study
                )[row, column],
                domain=self.domain,
                family=self.family,
                degree=self.degree,
                name=f"{self.name}_IMAG_SAMPLE",
            )
        real = self._real.solve()
        imaginary = self._imaginary.solve()
        return complex(
            results.probe(real, at=self.point),
            results.probe(imaginary, at=self.point),
        )

    def to_ir(self) -> dict[str, object]:
        return {
            "kind": "prepared_harmonic_stress_probe",
            "point": self.point,
            "component": self.component,
            "family": self.family,
            "degree": self.degree,
            "name": self.name,
            "recovery": "global_l2_projection_with_reused_mass_operator",
        }

    def close(self) -> None:
        for projector in (self._real, self._imaginary):
            if projector is not None:
                projector.close()


def nafems_r0016_test5h_benchmark(
    *,
    cells=(5, 2, 1),
    frequencies=None,
    comm=MPI.COMM_WORLD,
) -> tuple[ForcedVibrationBenchmark, object]:
    """Solve the public NAFEMS R0016 Test 5H physical problem.

    The reference reports a 42.65 Hz response peak, 13.45 mm vertical
    displacement and 241.9 MPa extreme-fibre bending stress.  NAFEMS does not
    prescribe acceptance tolerances; the limits below are AgentFEM promotion
    gates fixed independently of the numerical result.

    AgentFEM uses a complete quadratic hexahedral displacement field on a
    structured mesh.  It does not claim element-by-element identity with the
    Abaqus C3D10/C3D20/C3D27 comparison rows.  Two stress traces are retained:
    an unsmoothed discontinuous-Q2 cell-side field and an explicitly labelled
    continuous-P1 L2 recovery used for the external scalar comparison.
    """

    selected_cells = tuple(int(value) for value in cells)
    if len(selected_cells) != 3 or any(value <= 0 for value in selected_cells):
        raise ValueError("cells must contain three positive integers.")
    if selected_cells[0] % 2 == 0 or selected_cells[2] % 2 == 0:
        raise ValueError(
            "The axial and through-width cell counts must be odd so the "
            "frozen point lies inside one cell in x and z for deterministic "
            "one-sided stress sampling."
        )
    if selected_cells[1] % 2:
        raise ValueError(
            "The height cell count must be even so the y=1 support lines exist."
        )
    expected_axis = _test5h_frequency_axis()
    axis = (
        expected_axis
        if frequencies is None
        else np.asarray(tuple(frequencies), dtype=float)
    )
    if axis.shape != (50,) or not np.allclose(
        axis, expected_axis, rtol=0.0, atol=1.0e-13
    ):
        raise ValueError(
            "NAFEMS R0016 Test 5H requires the published 50-point 40--45 Hz axis."
        )
    cell_widths = (
        10.0 / selected_cells[0],
        2.0 / selected_cells[1],
        2.0 / selected_cells[2],
    )
    refinement_contract = _test5h_refinement_contract(axis)

    domain = mesh.cuboid(
        (0.0, 0.0, 0.0),
        (10.0, 2.0, 2.0),
        selected_cells,
        comm=comm,
        cell_type="hexahedron",
    )
    study = studies.harmonic_solid(dimension=3, name="NAFEMS R0016 Test 5H")
    model = models.create(study=study, mesh=domain, name="nafems_r0016_test5h")
    displacement = model.field(fields.displacement(domain, degree=2))
    material = elasticity.isotropic_elastic(
        young=200.0e9,
        poisson=0.3,
        density=8000.0,
        name="NAFEMS R0016 Test 5H isotropic solid",
    )
    model.material(material)

    tolerance = 1.0e-11
    left = mesh.face(domain, axis="x", value=0.0, name="left_end", tag=1)
    right = mesh.face(domain, axis="x", value=10.0, name="right_end", tag=2)
    top = mesh.face(domain, axis="y", value=2.0, name="top_pressure", tag=3)
    model.fix(displacement, on=left, component=2, value=0.0)
    model.fix(displacement, on=right, component=2, value=0.0)
    model.fix(
        displacement,
        location=lambda x: (
            np.isclose(x[0], 0.0, rtol=0.0, atol=tolerance)
            & np.isclose(x[1], 1.0, rtol=0.0, atol=tolerance)
        ),
        components=(0, 1),
        value=0.0,
        name="left_midline_pin",
    )
    model.fix(
        displacement,
        location=lambda x: (
            np.isclose(x[0], 10.0, rtol=0.0, atol=tolerance)
            & np.isclose(x[1], 1.0, rtol=0.0, atol=tolerance)
        ),
        component=1,
        value=0.0,
        name="right_midline_roller",
    )
    model.traction((0.0, -0.5e6, 0.0), on=top, name="harmonic_top_pressure")

    stiffness = model.stiffness(displacement)
    mass = model.mass(displacement)
    damping = operators.rayleigh_damping(
        mass,
        stiffness,
        mass_coefficient=5.36,
        stiffness_coefficient=7.46e-5,
    )
    force = model.external_force(displacement)
    point = (5.0, 2.0, 1.0)
    response_displacement = results.harmonic_response(
        "MIDSPAN_UY",
        partial(_point_displacement, point=point, component=1),
        unit="m",
        description="Complex vertical displacement at the frozen midspan point.",
    )
    cell_stress_sampler = _PreparedStressSampler(
        material=material,
        study=study,
        domain=domain,
        point=point,
        component=(0, 0),
        family="DG",
        degree=2,
        name="MIDSPAN_S11_CELL_SIDE",
    )
    recovered_stress_sampler = _PreparedStressSampler(
        material=material,
        study=study,
        domain=domain,
        point=point,
        component=(0, 0),
        family="Lagrange",
        degree=1,
        name="MIDSPAN_S11_NAFEMS_RECOVERED",
    )
    response_cell_stress = results.harmonic_response(
        "MIDSPAN_S11_CELL_SIDE",
        cell_stress_sampler,
        unit="Pa",
        description="Unsmoothed discontinuous-Q2 cell-side axial stress.",
    )
    response_recovered_stress = results.harmonic_response(
        "MIDSPAN_S11_NAFEMS_RECOVERED",
        recovered_stress_sampler,
        unit="Pa",
        description="Continuous-P1 L2 recovered axial stress for comparison.",
    )
    step = model.step(
        target=displacement,
        K=stiffness,
        M=mass,
        C=damping,
        F=force,
        frequencies=axis,
        responses=(
            response_displacement,
            response_cell_stress,
            response_recovered_stress,
        ),
        solver_options=solvers.direct_solver(),
        name="nafems_r0016_test5h_direct_harmonic",
    )
    try:
        simulation = step.solve_result()
    finally:
        cell_stress_sampler.close()
        recovered_stress_sampler.close()

    displacement_amplitude = simulation.histories["MIDSPAN_UY_AMPLITUDE"].values
    recovered_stress = simulation.histories[
        "MIDSPAN_S11_NAFEMS_RECOVERED_AMPLITUDE"
    ].values
    displacement_peak_index = int(np.argmax(displacement_amplitude))
    stress_peak_index = int(np.argmax(recovered_stress))
    peak_index_separation = abs(displacement_peak_index - stress_peak_index)
    if peak_index_separation > _TEST5H_MAXIMUM_PEAK_INDEX_SEPARATION_BINS:
        raise RuntimeError(
            "NAFEMS R0016 Test 5H requires the displacement and recovered-stress "
            "peaks to identify the same discrete resonance within one frequency "
            f"bin; got indices {displacement_peak_index} and {stress_peak_index}."
        )
    peak_frequency = float(axis[displacement_peak_index])
    peak_displacement = float(displacement_amplitude[displacement_peak_index])
    peak_stress = float(recovered_stress[stress_peak_index])
    reference_frequency = _TEST5H_REFERENCE_VALUES["peak_frequency_hz"]
    reference_displacement = _TEST5H_REFERENCE_VALUES["peak_displacement_m"]
    reference_stress = _TEST5H_REFERENCE_VALUES["peak_extreme_fibre_s11_pa"]
    loaded_area = float(results.integral(1.0, measure=top.measure, comm=comm))
    expected_area = _TEST5H_EXPECTED_LOADED_AREA_M2
    load_resultant = _TEST5H_PRESSURE_AMPLITUDE_PA * loaded_area
    reference_angular_frequency = 2.0 * np.pi * reference_frequency
    reference_damping_ratio = 0.5 * (
        5.36 / reference_angular_frequency + 7.46e-5 * reference_angular_frequency
    )
    max_residual = float(np.max(simulation.histories["relative_residual_norm"].values))
    max_energy_error = float(
        np.max(simulation.histories["relative_cycle_energy_balance_error"].values)
    )
    benchmark = ForcedVibrationBenchmark(
        name=_TEST5H_NAME,
        reference=(
            "NAFEMS R0016 Test 5H; Abaqus Benchmarks Guide, Forced vibration "
            "of a simply supported beam"
        ),
        mpi_ranks=int(comm.size),
        quantities={
            "relative_peak_frequency_error": abs(peak_frequency - reference_frequency)
            / reference_frequency,
            "relative_peak_displacement_error": abs(
                peak_displacement - reference_displacement
            )
            / reference_displacement,
            "relative_peak_stress_error": abs(peak_stress - reference_stress)
            / reference_stress,
            "maximum_relative_residual_norm": max_residual,
            "maximum_relative_cycle_energy_balance_error": max_energy_error,
            "relative_loaded_area_error": abs(loaded_area - expected_area)
            / expected_area,
            "peak_frequency_hz": peak_frequency,
            "peak_displacement_m": peak_displacement,
            "peak_recovered_s11_pa": peak_stress,
            "peak_recovered_s11_frequency_hz": float(axis[stress_peak_index]),
            "peak_index_separation_bins": peak_index_separation,
            "peak_frequency_separation_hz": abs(
                float(axis[stress_peak_index]) - peak_frequency
            ),
            "loaded_area_m2": loaded_area,
            "load_resultant_amplitude_n": load_resultant,
            "rayleigh_damping_ratio_at_reference_peak": reference_damping_ratio,
        },
        tolerances=dict(_TEST5H_EXTERNAL_TOLERANCES),
        extraction={
            "cells": selected_cells,
            "cell_widths_m": cell_widths,
            "characteristic_cell_size_m": float(np.linalg.norm(cell_widths)),
            "refinement_contract": refinement_contract,
            "refinement_contract_sha256": content_fingerprint(refinement_contract),
            "frequency_axis_sha256": content_fingerprint(axis.tolist()),
            "frequency_axis": axis.tolist(),
            "frequency_point_count": int(axis.size),
            "primary_peak": "discrete maximum over the published 50 points",
            "displacement_peak_index": displacement_peak_index,
            "stress_peak_index": stress_peak_index,
            "peak_index_separation_bins": peak_index_separation,
            "maximum_peak_index_separation_bins": (
                _TEST5H_MAXIMUM_PEAK_INDEX_SEPARATION_BINS
            ),
            "point": point,
            "displacement": "absolute complex UY amplitude",
            "stress": "absolute complex longitudinal S11 elastic-stress amplitude",
            "comparison_stress_recovery": "continuous-P1 global L2 recovery",
            "audit_stress_recovery": "unsmoothed discontinuous-Q2 cell side",
            "phasor_convention": "exp(+i*omega*t)",
            "reference_load_convention": "sinusoidal; amplitude comparison only",
            "source_response_location": (
                "midspan top edge; the public solid decks use an available "
                "through-width node that depends on the element mesh"
            ),
            "discretization": (
                "AgentFEM complete-Q2 hexahedra; no Abaqus element identity claimed"
            ),
            "tolerance_authority": "AgentFEM promotion contract, not NAFEMS",
            "reference_values": {
                "peak_frequency_hz": reference_frequency,
                "peak_displacement_m": reference_displacement,
                "peak_extreme_fibre_s11_pa": reference_stress,
            },
            "source": {
                "nafems_r0016": NAFEMS_R0016_URL,
                "abaqus_test_5h": ABAQUS_TEST5H_URL,
                "abaqus_c3d10_input_sha256": ABAQUS_C3D10_DECK_SHA256,
                "abaqus_c3d27_input_sha256": ABAQUS_C3D27_DECK_SHA256,
            },
        },
    )
    simulation.add_quantity(
        "nafems_peak_frequency",
        peak_frequency,
        unit="Hz",
        kind="benchmark",
    )
    simulation.add_quantity(
        "nafems_peak_midspan_displacement",
        peak_displacement,
        unit="m",
        kind="benchmark",
    )
    simulation.add_quantity(
        "nafems_peak_recovered_s11",
        peak_stress,
        unit="Pa",
        kind="benchmark",
    )
    simulation.metadata["external_benchmark"] = benchmark.as_dict()
    return benchmark, simulation


def _validated_test5h_level(
    level: ForcedVibrationBenchmark,
    *,
    index: int,
    expected_axis: np.ndarray,
    expected_contract: dict[str, object],
) -> dict[str, object]:
    """Recompute one refinement level's frozen Test 5H evidence."""

    if level.name != _TEST5H_NAME:
        raise ValueError(f"Level {index} is not the frozen Test 5H benchmark.")

    required_tolerances = _TEST5H_EXTERNAL_TOLERANCES
    if set(level.tolerances) != set(required_tolerances):
        raise ValueError(
            f"Level {index} changed the required Test 5H tolerance schema."
        )
    for name, expected in required_tolerances.items():
        actual = float(level.tolerances[name])
        if not np.isfinite(actual) or actual != expected:
            raise ValueError(
                f"Level {index} changed frozen Test 5H tolerance {name!r}."
            )

    required_quantities = {
        *required_tolerances,
        "peak_frequency_hz",
        "peak_displacement_m",
        "peak_recovered_s11_pa",
        "peak_recovered_s11_frequency_hz",
        "peak_frequency_separation_hz",
        "loaded_area_m2",
        "load_resultant_amplitude_n",
    }
    missing_quantities = required_quantities.difference(level.quantities)
    if missing_quantities:
        raise ValueError(
            f"Level {index} is missing required Test 5H quantities: "
            + ", ".join(sorted(missing_quantities))
        )
    quantities = {name: float(level.quantities[name]) for name in required_quantities}
    nonfinite = sorted(
        name for name, value in quantities.items() if not np.isfinite(value)
    )
    if nonfinite:
        raise ValueError(
            f"Level {index} has non-finite Test 5H quantities: " + ", ".join(nonfinite)
        )
    if any(
        quantities[name] < 0.0
        for name in (
            *required_tolerances,
            "peak_frequency_separation_hz",
        )
    ):
        raise ValueError(f"Level {index} records a negative Test 5H error quantity.")
    if any(
        quantities[name] <= 0.0
        for name in (
            "peak_frequency_hz",
            "peak_displacement_m",
            "peak_recovered_s11_pa",
            "loaded_area_m2",
            "load_resultant_amplitude_n",
        )
    ):
        raise ValueError(f"Level {index} records a non-positive Test 5H observable.")

    extraction = level.extraction
    raw_cells = tuple(extraction.get("cells", ()))
    if len(raw_cells) != 3 or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        for value in raw_cells
    ):
        raise ValueError(f"Level {index} does not record three positive mesh counts.")
    level_cells = tuple(int(value) for value in raw_cells)
    if level_cells[0] % 2 == 0 or level_cells[2] % 2 == 0 or level_cells[1] % 2:
        raise ValueError(
            f"Level {index} mesh counts violate the frozen Test 5H sampling/support contract."
        )

    axis = np.asarray(extraction.get("frequency_axis", ()), dtype=float)
    if axis.shape != expected_axis.shape or not np.allclose(
        axis, expected_axis, rtol=0.0, atol=1.0e-13
    ):
        raise ValueError(f"Level {index} changed the frozen frequency axis.")
    actual_axis_fingerprint = content_fingerprint(axis.tolist())
    if extraction.get("frequency_axis_sha256") != actual_axis_fingerprint:
        raise ValueError(
            f"Level {index} frequency-axis fingerprint does not match its values."
        )

    if extraction.get("comparison_stress_recovery") != (
        "continuous-P1 global L2 recovery"
    ):
        raise ValueError(f"Level {index} changed the stress recovery contract.")
    contract = extraction.get("refinement_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Level {index} is missing its Test 5H model contract.")
    actual_contract_fingerprint = content_fingerprint(contract)
    if extraction.get("refinement_contract_sha256") != actual_contract_fingerprint:
        raise ValueError(
            f"Level {index} refinement-contract fingerprint does not match its contract."
        )
    expected_contract_fingerprint = content_fingerprint(expected_contract)
    if actual_contract_fingerprint != expected_contract_fingerprint:
        raise ValueError(f"Level {index} changed the frozen Test 5H model contract.")

    reference_values = extraction.get("reference_values")
    if content_fingerprint(reference_values) != content_fingerprint(
        _TEST5H_REFERENCE_VALUES
    ):
        raise ValueError(f"Level {index} changed the public Test 5H references.")

    level_widths = tuple(float(value) for value in extraction.get("cell_widths_m", ()))
    if len(level_widths) != 3 or not np.all(np.isfinite(level_widths)):
        raise ValueError(f"Level {index} does not record finite cell widths.")
    expected_widths = tuple(
        extent / count for extent, count in zip(_TEST5H_GEOMETRY_M, level_cells)
    )
    if not np.allclose(level_widths, expected_widths, rtol=1.0e-13, atol=0.0):
        raise ValueError(
            f"Level {index} cell widths disagree with its geometry and mesh counts."
        )
    characteristic_size = float(extraction["characteristic_cell_size_m"])
    expected_size = float(np.linalg.norm(expected_widths))
    if not np.isfinite(characteristic_size) or not np.isclose(
        characteristic_size, expected_size, rtol=1.0e-13, atol=0.0
    ):
        raise ValueError(
            f"Level {index} characteristic cell size disagrees with its cell widths."
        )

    peak_indices = {}
    for name in ("displacement_peak_index", "stress_peak_index"):
        value = extraction.get(name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not 0 <= int(value) < expected_axis.size
        ):
            raise ValueError(f"Level {index} has an invalid {name}.")
        peak_indices[name] = int(value)
    peak_index_separation = abs(
        peak_indices["displacement_peak_index"] - peak_indices["stress_peak_index"]
    )
    if peak_index_separation > _TEST5H_MAXIMUM_PEAK_INDEX_SEPARATION_BINS:
        raise ValueError(
            f"Level {index} displacement and stress peaks differ by more than one bin."
        )
    if extraction.get("maximum_peak_index_separation_bins") != (
        _TEST5H_MAXIMUM_PEAK_INDEX_SEPARATION_BINS
    ):
        raise ValueError(f"Level {index} changed the Test 5H peak-index contract.")
    if extraction.get("peak_index_separation_bins") != peak_index_separation:
        raise ValueError(
            f"Level {index} records an inconsistent peak-index separation."
        )

    expected_peak_frequency = float(
        expected_axis[peak_indices["displacement_peak_index"]]
    )
    expected_stress_frequency = float(expected_axis[peak_indices["stress_peak_index"]])
    expected_frequency_separation = abs(
        expected_stress_frequency - expected_peak_frequency
    )
    derived = {
        "peak_frequency_hz": expected_peak_frequency,
        "peak_recovered_s11_frequency_hz": expected_stress_frequency,
        "peak_index_separation_bins": float(peak_index_separation),
        "peak_frequency_separation_hz": expected_frequency_separation,
        "relative_peak_frequency_error": abs(
            quantities["peak_frequency_hz"]
            - _TEST5H_REFERENCE_VALUES["peak_frequency_hz"]
        )
        / _TEST5H_REFERENCE_VALUES["peak_frequency_hz"],
        "relative_peak_displacement_error": abs(
            quantities["peak_displacement_m"]
            - _TEST5H_REFERENCE_VALUES["peak_displacement_m"]
        )
        / _TEST5H_REFERENCE_VALUES["peak_displacement_m"],
        "relative_peak_stress_error": abs(
            quantities["peak_recovered_s11_pa"]
            - _TEST5H_REFERENCE_VALUES["peak_extreme_fibre_s11_pa"]
        )
        / _TEST5H_REFERENCE_VALUES["peak_extreme_fibre_s11_pa"],
        "relative_loaded_area_error": abs(
            quantities["loaded_area_m2"] - _TEST5H_EXPECTED_LOADED_AREA_M2
        )
        / _TEST5H_EXPECTED_LOADED_AREA_M2,
        "load_resultant_amplitude_n": (
            _TEST5H_PRESSURE_AMPLITUDE_PA * quantities["loaded_area_m2"]
        ),
    }
    for name, expected in derived.items():
        if not np.isclose(quantities[name], expected, rtol=1.0e-12, atol=1.0e-15):
            raise ValueError(
                f"Level {index} derived Test 5H quantity {name!r} is inconsistent."
            )

    return {
        "cells": level_cells,
        "cell_widths": level_widths,
        "characteristic_size": characteristic_size,
        "contract_fingerprint": actual_contract_fingerprint,
        "axis_fingerprint": actual_axis_fingerprint,
        "quantities": quantities,
        "peak_index_separation": peak_index_separation,
    }


def certify_nafems_r0016_test5h_spatial_convergence(
    levels,
    *,
    refinement_relative_tolerances=None,
    residual_tolerance: float = 1.0e-8,
    energy_tolerance: float = 1.0e-8,
) -> ForcedVibrationConvergenceCertificate:
    """Certify a declared three-or-more-level Test 5H refinement sequence.

    Only the finest-pair relative changes are acceptance quantities.  Earlier
    levels remain in the evidence so refinement histories cannot be reduced to
    one favourable pair.  The function requires the frozen frequency and
    extraction contracts to match across every level.
    """

    selected = tuple(levels)
    if len(selected) < 3:
        raise ValueError("At least three Test 5H refinement levels are required.")
    selected_residual_tolerance = float(residual_tolerance)
    selected_energy_tolerance = float(energy_tolerance)
    if (
        not np.isfinite(selected_residual_tolerance)
        or selected_residual_tolerance <= 0.0
    ):
        raise ValueError("residual_tolerance must be finite and positive.")
    if not np.isfinite(selected_energy_tolerance) or selected_energy_tolerance <= 0.0:
        raise ValueError("energy_tolerance must be finite and positive.")

    expected_axis = _test5h_frequency_axis()
    expected_contract = _test5h_refinement_contract(expected_axis)
    cells: list[tuple[int, int, int]] = []
    cell_widths: list[tuple[float, float, float]] = []
    characteristic_sizes: list[float] = []
    contract_fingerprints: list[str] = []
    axis_fingerprints: list[str] = []
    validated_quantities: list[dict[str, float]] = []
    peak_index_separations: list[int] = []
    for index, level in enumerate(selected):
        if not isinstance(level, ForcedVibrationBenchmark):
            raise TypeError("levels must contain ForcedVibrationBenchmark instances.")
        evidence = _validated_test5h_level(
            level,
            index=index,
            expected_axis=expected_axis,
            expected_contract=expected_contract,
        )
        cells.append(evidence["cells"])
        cell_widths.append(evidence["cell_widths"])
        characteristic_sizes.append(evidence["characteristic_size"])
        contract_fingerprints.append(evidence["contract_fingerprint"])
        axis_fingerprints.append(evidence["axis_fingerprint"])
        validated_quantities.append(evidence["quantities"])
        peak_index_separations.append(evidence["peak_index_separation"])

    if any(
        fine >= coarse
        for coarse, fine in zip(characteristic_sizes[:-1], characteristic_sizes[1:])
    ):
        raise ValueError(
            "characteristic cell size must decrease strictly through the refinement sequence."
        )
    if any(
        any(fine_axis <= coarse_axis for coarse_axis, fine_axis in zip(coarse, fine))
        for coarse, fine in zip(cells[:-1], cells[1:])
    ):
        raise ValueError("Every cell-count axis must increase through refinement.")
    if len(set(contract_fingerprints)) != 1:
        raise ValueError("Test 5H refinement model contracts differ.")
    if len(set(axis_fingerprints)) != 1:
        raise ValueError("Test 5H refinement frequency identities differ.")

    values = {
        "peak_frequency_hz": tuple(
            level["peak_frequency_hz"] for level in validated_quantities
        ),
        "peak_displacement_m": tuple(
            level["peak_displacement_m"] for level in validated_quantities
        ),
        "peak_recovered_s11_pa": tuple(
            level["peak_recovered_s11_pa"] for level in validated_quantities
        ),
    }
    tolerances = {
        "peak_frequency_hz": 0.01,
        "peak_displacement_m": 0.01,
        "peak_recovered_s11_pa": 0.01,
    }
    if refinement_relative_tolerances is not None:
        unknown = set(refinement_relative_tolerances).difference(tolerances)
        if unknown:
            raise ValueError(
                "Unknown Test 5H convergence observable(s): "
                + ", ".join(sorted(unknown))
            )
        tolerances.update(
            {
                name: float(value)
                for name, value in refinement_relative_tolerances.items()
            }
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in tolerances.values()):
        raise ValueError("Refinement tolerances must be finite and positive.")

    convergence_studies = {
        name: verification.convergence_study(
            name=f"nafems_r0016_test5h_{name}",
            observable=name,
            samples=tuple(
                verification.ConvergenceSample(
                    characteristic_size=size,
                    value=value,
                    label=f"cells={level_cells}",
                )
                for size, value, level_cells in zip(characteristic_sizes, series, cells)
            ),
        )
        for name, series in values.items()
    }
    successive_changes = {
        name: tuple(
            _relative_change(coarse, fine)
            for coarse, fine in zip(series[:-1], series[1:])
        )
        for name, series in values.items()
    }
    finest_changes = {
        name: study.finest_relative_change
        for name, study in convergence_studies.items()
    }
    successive_decrease = all(
        all(fine <= coarse for coarse, fine in zip(changes[:-1], changes[1:]))
        for changes in successive_changes.values()
    )
    residuals = tuple(
        level["maximum_relative_residual_norm"] for level in validated_quantities
    )
    energy_errors = tuple(
        level["maximum_relative_cycle_energy_balance_error"]
        for level in validated_quantities
    )
    external_comparison = all(
        all(level[name] <= limit for name, limit in _TEST5H_EXTERNAL_TOLERANCES.items())
        for level in validated_quantities
    )
    accepted = (
        external_comparison
        and successive_decrease
        and all(finest_changes[name] <= tolerances[name] for name in values)
        and all(value <= selected_residual_tolerance for value in residuals)
        and all(value <= selected_energy_tolerance for value in energy_errors)
    )
    return ForcedVibrationConvergenceCertificate(
        benchmark_name=_TEST5H_NAME,
        cells=tuple(cells),
        cell_widths_m=tuple(cell_widths),
        characteristic_sizes_m=tuple(characteristic_sizes),
        peak_frequencies_hz=values["peak_frequency_hz"],
        peak_displacements_m=values["peak_displacement_m"],
        peak_recovered_s11_pa=values["peak_recovered_s11_pa"],
        successive_relative_changes=successive_changes,
        finest_pair_relative_changes=finest_changes,
        refinement_relative_tolerances=tolerances,
        maximum_relative_residual_norms=residuals,
        maximum_relative_cycle_energy_balance_errors=energy_errors,
        peak_index_separations_bins=tuple(peak_index_separations),
        external_comparison_tolerances=dict(_TEST5H_EXTERNAL_TOLERANCES),
        residual_tolerance=selected_residual_tolerance,
        energy_tolerance=selected_energy_tolerance,
        frequency_axis_sha256=axis_fingerprints[0],
        model_contract_sha256=contract_fingerprints[0],
        successive_changes_decrease=successive_decrease,
        all_levels_pass_external_comparison=external_comparison,
        accepted=accepted,
    )


def nafems_r0016_test5h_spatial_convergence(
    *,
    cells=((5, 2, 1), (9, 4, 3), (13, 6, 5)),
    comm=MPI.COMM_WORLD,
):
    """Execute the declared Test 5H spatial-refinement stability study."""

    benchmarks = []
    simulations = []
    for level_cells in cells:
        benchmark, simulation = nafems_r0016_test5h_benchmark(
            cells=level_cells,
            comm=comm,
        )
        benchmarks.append(benchmark)
        simulations.append(simulation)
    certificate = certify_nafems_r0016_test5h_spatial_convergence(benchmarks)
    for simulation in simulations:
        simulation.metadata["spatial_refinement_certificate"] = certificate.as_dict()
    return certificate, tuple(simulations)


def _relative_change(coarse: float, fine: float) -> float:
    scale = abs(float(fine))
    if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
        raise ValueError(
            "Refinement observables must have a finite nonzero fine value."
        )
    return abs(float(fine) - float(coarse)) / scale


def _point_displacement(step, *, point, component: int) -> complex:
    real = np.asarray(results.probe(step.solution_real, at=point), dtype=float)
    imaginary = np.asarray(
        results.probe(step.solution_imaginary, at=point), dtype=float
    )
    return complex(real[int(component)], imaginary[int(component)])


__all__ = [
    "ForcedVibrationBenchmark",
    "ForcedVibrationConvergenceCertificate",
    "certify_nafems_r0016_test5h_spatial_convergence",
    "nafems_r0016_test5h_benchmark",
    "nafems_r0016_test5h_spatial_convergence",
]
