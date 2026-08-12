"""Machine-readable verification inventory.

The registry does not replace tests. It links every present capability claim
to a reference problem, a criterion, executable evidence, and a maturity
boundary so that functionality cannot silently outrun verification.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    """One verification obligation and its executable evidence."""

    identifier: str
    capability: str
    level: str
    reference: str
    criterion: str
    automated_test: str
    status: str = "automated"

    def as_dict(self) -> dict[str, str]:
        return {
            "identifier": self.identifier,
            "capability": self.capability,
            "level": self.level,
            "reference": self.reference,
            "criterion": self.criterion,
            "automated_test": self.automated_test,
            "status": self.status,
        }


_BENCHMARKS = (
    BenchmarkSpec(
        identifier="cae_reliability_cliffs",
        capability="scientific_verification",
        level="workflow",
        reference="knowledge/benchmarks/cae_reliability_cliffs.json",
        criterion=(
            "orientation covariance, discretization convergence, theory "
            "applicability, and dataset trust gates remain explicit"
        ),
        automated_test=(
            "tests/test_verification.py tests/test_release_goldens.py "
            "tests/test_campaigns.py"
        ),
        status="partial_automated_suite",
    ),
    BenchmarkSpec(
        identifier="linear_static_cantilever",
        capability="linear_elasticity",
        level="finite_element",
        reference="knowledge/benchmarks/linear_static_cantilever.json",
        criterion="fixed Q2 release mesh reproduces the versioned maximum displacement",
        automated_test="python examples/static_elasticity_2d.py",
        status="release_regression",
    ),
    BenchmarkSpec(
        identifier="elasticity_foundation",
        capability="linear_elasticity",
        level="finite_element",
        reference="knowledge/benchmarks/elasticity_foundation.json",
        criterion=(
            "a displacement-controlled 3D patch reproduces constant strain and "
            "stress, a two-material series bar preserves regional fields, and "
            "named-boundary reactions close the applied load in serial and MPI"
        ),
        automated_test=(
            "tests/test_results.py -k 'displacement_controlled_3d or "
            "two_material_elastic_bar'; tests/test_parallel_results.py"
        ),
        status="release_regression",
    ),
    BenchmarkSpec(
        identifier="operator_contracts",
        capability="operator_systems",
        level="interface",
        reference="knowledge/benchmarks/operator_contracts.json",
        criterion=(
            "operator roles match UFL arity, first-order systems reject role "
            "mismatch, and residual linearization records K_t = dR/du"
        ),
        automated_test="tests/test_operators.py",
    ),
    BenchmarkSpec(
        identifier="neo_hookean_energy_gradient",
        capability="neo_hookean",
        level="material_point",
        reference="DOLFINx hyperelasticity demo; compressible Neo-Hookean energy",
        criterion="analytical first Piola stress matches a centered energy derivative",
        automated_test="tests/test_constitutive_models.py::test_neo_hookean_nominal_stress_is_energy_derivative",
    ),
    BenchmarkSpec(
        identifier="neo_hookean_displacement_patch",
        capability="neo_hookean",
        level="finite_element",
        reference="displacement-controlled homogeneous patch",
        criterion="PETSc SNES converges and prescribed displacement is recovered",
        automated_test="tests/test_constitutive_models.py::test_neo_hookean_model_step_solves_a_displacement_controlled_patch",
    ),
    BenchmarkSpec(
        identifier="neo_hookean_release",
        capability="neo_hookean",
        level="finite_element",
        reference="knowledge/benchmarks/neo_hookean_release.json",
        criterion=(
            "versioned energy, first-Piola stress, positive J, and a "
            "displacement-controlled FEM patch remain consistent"
        ),
        automated_test="tests/test_constitutive_models.py -k neo_hookean",
        status="release_regression",
    ),
    BenchmarkSpec(
        identifier="plane_stress_thin_3d_crosscheck",
        capability="neo_hookean_plane_stress",
        level="finite_element",
        reference="knowledge/benchmarks/plane_stress_thin_3d_crosscheck.json",
        criterion=(
            "condensed plane-stress nominal stress and energy match a thin "
            "three-dimensional affine FEM patch with traction-free transverse stress"
        ),
        automated_test=(
            "tests/test_dynamic_fracture_benchmarks.py::"
            "test_plane_stress_matches_thin_three_dimensional_affine_patch"
        ),
        status="experimental_geometry_crosscheck",
    ),
    BenchmarkSpec(
        identifier="distributed_cohesive_force",
        capability="dynamic_cohesive_fracture",
        level="workflow",
        reference="knowledge/benchmarks/distributed_cohesive_force.json",
        criterion=(
            "two MPI ranks assemble every physical interface facet once, close "
            "force and energy, and restart the split-interface Explicit state on one rank"
        ),
        automated_test=(
            "tests/test_parallel_cohesive.py; "
            "tests/portable_cohesive_dynamics_driver.py"
        ),
        status="experimental_mpi_reference",
    ),
    BenchmarkSpec(
        identifier="c3d10h_periodic_cell",
        capability="mixed_neo_hookean",
        level="workflow",
        reference="knowledge/benchmarks/c3d10h_periodic_cell.json",
        criterion=(
            "a direct C3D10H source selects P2/DG0 mixed periodic equilibrium "
            "and satisfies its versioned "
            "stress, volume, positive-J, and equation-mismatch contract"
        ),
        automated_test=(
            "tests/test_abaqus_interop.py -k direct_c3d10h; "
            "tests/test_engineering_workflows.py -k mixed_hybrid_affine"
        ),
        status="manual_large_mesh_release_regression",
    ),
    BenchmarkSpec(
        identifier="j2_radial_return",
        capability="j2_plasticity",
        level="material_point",
        reference="closest-point radial return for Mises plasticity",
        criterion="updated stress lies on the hardened yield surface",
        automated_test="tests/test_constitutive_models.py::test_j2_radial_return_lands_on_the_hardened_yield_surface",
    ),
    BenchmarkSpec(
        identifier="j2_algorithmic_tangent",
        capability="j2_plasticity",
        level="material_point",
        reference="directional derivative of the radial-return mapping",
        criterion="analytical algorithmic tangent matches a centered derivative",
        automated_test="tests/test_p1_platform.py::test_j2_algorithmic_tangent_matches_return_map_derivative",
    ),
    BenchmarkSpec(
        identifier="j2_global_restart",
        capability="j2_plasticity",
        level="finite_element",
        reference="knowledge/benchmarks/j2_global_restart.json",
        criterion=(
            "uninterrupted and checkpoint/restarted paths recover U, PEEQ, and "
            "history; a uniaxial patch matches the analytical hardening path"
        ),
        automated_test="tests/test_p1_platform.py -k global_j2",
    ),
    BenchmarkSpec(
        identifier="j2_multielement_patch",
        capability="j2_plasticity",
        level="finite_element",
        reference="knowledge/benchmarks/j2_multielement_patch.json",
        criterion=(
            "a multi-element 3D displacement-controlled patch reproduces the "
            "analytical isotropic-hardening path, uniform PEEQ, and work-energy closure"
        ),
        automated_test=(
            "tests/test_p1_platform.py::"
            "test_global_j2_multielement_patch_matches_the_uniaxial_golden"
        ),
    ),
    BenchmarkSpec(
        identifier="j2_nonuniform_bending",
        capability="j2_plasticity",
        level="finite_element",
        reference="knowledge/benchmarks/j2_nonuniform_bending.json",
        criterion=(
            "a displacement-controlled 3D cantilever retains simultaneous "
            "elastic/plastic points, nonuniform PEEQ, and work-energy closure"
        ),
        automated_test=(
            "tests/test_p1_platform.py::"
            "test_global_j2_nonuniform_bending_path_localizes_plastic_state"
        ),
    ),
    BenchmarkSpec(
        identifier="thermoelastic_free_expansion",
        capability="thermoelasticity",
        level="finite_element",
        reference="knowledge/benchmarks/thermoelastic_free_expansion.json",
        criterion="free displacement equals alpha DeltaT L",
        automated_test="tests/test_p1_platform.py::test_thermoelastic_material_arrhenius_creep_and_free_expansion",
    ),
    BenchmarkSpec(
        identifier="transient_heat_release",
        capability="transient_heat",
        level="finite_element",
        reference="knowledge/benchmarks/transient_heat_release.json",
        criterion="five implicit-Euler increments reproduce the versioned mean temperature",
        automated_test="AGENTFEM_RELEASE_SMOKE=1 python examples/transient_heat_2d.py",
        status="release_regression",
    ),
    BenchmarkSpec(
        identifier="wave_release",
        capability="structural_dynamics",
        level="finite_element",
        reference="knowledge/benchmarks/wave_release.json",
        criterion=(
            "a reduced two-material explicit workflow reproduces wave speeds, "
            "Courant number, receiver response, and periodic compatibility"
        ),
        automated_test=(
            "tests/test_release_goldens.py::"
            "test_wave_release_patch_matches_versioned_golden"
        ),
        status="release_regression",
    ),
    BenchmarkSpec(
        identifier="cohesive_mode_i_v0",
        capability="dynamic_cohesive_fracture",
        level="material_point_and_interface_element",
        reference="docs/dynamic_cohesive_fracture_architecture.md#v0----local-mathematics",
        criterion=(
            "bilinear envelope area equals fracture energy; irreversible "
            "loading paths, closure, precrack, equal-opposite facet forces, "
            "and interface splitting identities remain exact"
        ),
        automated_test="tests/test_interfaces.py",
        status="experimental_regression",
    ),
    BenchmarkSpec(
        identifier="finite_strain_explicit_v0",
        capability="dynamic_cohesive_fracture",
        level="finite_element_foundation",
        reference="docs/dynamic_cohesive_fracture_architecture.md#bulk-dynamics",
        criterion=(
            "Neo-Hookean dynamics selects the Total-Lagrangian Explicit "
            "provider, preserves rigid-rotation objectivity, reports the "
            "constitutive energy, and cannot silently lower to linear elasticity"
        ),
        automated_test="tests/test_dynamic_fracture.py",
        status="experimental_regression",
    ),
    BenchmarkSpec(
        identifier="cohesive_global_state_v0",
        capability="dynamic_cohesive_fracture",
        level="serial_global_consumer",
        reference="docs/dynamic_cohesive_fracture_architecture.md#state-ownership",
        criterion=(
            "bulk and cohesive residuals share accepted-step commit/rollback, "
            "typed energy, stable-step evidence, and checkpoint state identity"
        ),
        automated_test="tests/test_global_cohesive_residual.py",
        status="experimental_regression",
    ),
    BenchmarkSpec(
        identifier="mixed_mode_bending_external_contract",
        capability="mixed_mode_cohesive_fracture",
        level="external_curve_contract",
        reference=(
            "knowledge/benchmarks/mixed_mode_bending_external_contract.json"
        ),
        criterion=(
            "a source-identified mixed-mode bending curve is compared on common "
            "crack-length coordinates using explicitly declared load, displacement, "
            "and mode-mix tolerances"
        ),
        automated_test="tests/test_mixed_mode_benchmark.py",
        status="contract_ready_external_data_pending",
    ),
    BenchmarkSpec(
        identifier="finite_strain_incremental_waves_v1",
        capability="dynamic_cohesive_fracture",
        level="constitutive_and_homogeneous_prestrain",
        reference="knowledge/benchmarks/finite_strain_incremental_waves_v1.json",
        criterion=(
            "the analytical Neo-Hookean material tangent matches finite "
            "differences, the unstretched acoustic tensor recovers c_s/c_d, "
            "and current/reference propagation directions transform consistently"
        ),
        automated_test=(
            "tests/test_dynamic_fracture.py -k "
            "'acoustic_tensor or prestrained_wave or material_tangent'"
        ),
        status="experimental_v1_automated",
    ),
    BenchmarkSpec(
        identifier="dynamic_fracture_energy_v2",
        capability="dynamic_cohesive_fracture",
        level="finite_element_energy_convergence",
        reference="knowledge/benchmarks/dynamic_fracture_energy_v2.json",
        criterion=(
            "no-fracture mechanical energy converges under mesh refinement; "
            "complete interface separation dissipates exactly Gamma and the "
            "full external/internal/kinetic/fracture ledger converges in time"
        ),
        automated_test=(
            "tests/test_dynamic_fracture_benchmarks.py::"
            "test_v2_cohesive_dissipation_is_exact_and_energy_error_converges"
        ),
        status="experimental_v2_automated",
    ),
    BenchmarkSpec(
        identifier="classical_sub_rayleigh_crack_v3",
        capability="dynamic_cohesive_fracture",
        level="finite_element_dynamic_crack_guardrail",
        reference="knowledge/benchmarks/classical_sub_rayleigh_crack_v3.json",
        criterion=(
            "a precracked cohesive strip advances multiple facets while its "
            "window-fitted speed remains below c_R under mesh, time-step, and "
            "declared mass-damping perturbations with typed energy closure"
        ),
        automated_test=(
            "tests/test_dynamic_fracture_benchmarks.py::"
            "test_v3_classical_crack_remains_sub_rayleigh_under_refinement_and_damping"
        ),
        status="experimental_v3_guardrail_automated",
    ),
    BenchmarkSpec(
        identifier="creep_closed_forms",
        capability="power_law_creep",
        level="material_point",
        reference="Abaqus verification of creep integration",
        criterion="constant-stress creep matches the integrated power law and relaxation decays",
        automated_test="tests/test_constitutive_models.py::test_power_law_creep_matches_constant_stress_and_relaxation_solutions",
    ),
    BenchmarkSpec(
        identifier="implicit_creep_relaxation",
        capability="power_law_creep",
        level="finite_element",
        reference="knowledge/benchmarks/implicit_creep_relaxation.json",
        criterion=(
            "a 3D constant-strain bar follows the analytical relaxation path; "
            "state-limit cutback and checkpoint restart preserve accepted state"
        ),
        automated_test="tests/test_p1_platform.py -k global_implicit_creep",
    ),
    BenchmarkSpec(
        identifier="creep_abaqus_constant_stress",
        capability="power_law_creep",
        level="finite_element",
        reference="knowledge/benchmarks/creep_abaqus_constant_stress.json",
        criterion=(
            "the global 3D time-hardening creep path retains the official held "
            "stress and approaches the published closed-form creep strain"
        ),
        automated_test=(
            "tests/test_p1_platform.py -k official_abaqus_constant_stress"
        ),
        status="external_verification",
    ),
    BenchmarkSpec(
        identifier="arrhenius_global_creep",
        capability="power_law_creep",
        level="finite_element",
        reference="knowledge/benchmarks/arrhenius_global_creep.json",
        criterion=(
            "a nonuniform finite-element temperature field is consumed at the "
            "creep quadrature identity and produces traceable nonuniform CEEQ"
        ),
        automated_test=(
            "tests/test_p1_platform.py::"
            "test_global_arrhenius_creep_consumes_nonuniform_temperature_field"
        ),
    ),
    BenchmarkSpec(
        identifier="creep_damage_material_paths",
        capability="creep_damage",
        level="material_point",
        reference="knowledge/benchmarks/creep_damage_material_paths.json",
        criterion=(
            "K-R exact updates are subdivision invariant, tensor flow is "
            "deviatoric, and modified-theta fitting recovers a synthetic curve"
        ),
        automated_test="tests/test_constitutive_models.py",
    ),
    BenchmarkSpec(
        identifier="creep_hot_wall_release",
        capability="creep_damage",
        level="workflow",
        reference="knowledge/benchmarks/creep_hot_wall_release.json",
        criterion=(
            "the reduced sequential FEM-to-material-point assessment "
            "reproduces versioned stress, rupture-time, and damage observables"
        ),
        automated_test=(
            "AGENTFEM_RELEASE_SMOKE=1 python "
            "examples/creep_hot_wall_assessment.py"
        ),
        status="release_regression",
    ),
    BenchmarkSpec(
        identifier="rainflow_miner_history",
        capability="stress_life_fatigue",
        level="postprocess",
        reference="rainflow cycle counting followed by Palmgren-Miner summation",
        criterion="counted cycles reproduce analytical repeated-cycle damage",
        automated_test="tests/test_constitutive_models.py::test_rainflow_history_to_miner_damage_and_goodman_correction",
    ),
    BenchmarkSpec(
        identifier="cyclic_cohesive_material_path",
        capability="cyclic_cohesive_fatigue",
        level="global_lifecycle_and_facet",
        reference=(
            "irreversible bilinear monotonic limit plus constant-extrema "
            "analytical cycle-block evolution"
        ),
        criterion=(
            "monotonic recovery is exact, sub-threshold cycles are inactive, "
            "closure cannot heal, cycle jump equals exact cycles at fixed "
            "extrema, global feedback triggers atomic cutback, restart retains "
            "bulk/interface/cycle state, named 3D interfaces share one solver "
            "mesh, and the facet consumer degrades force with nonnegative dissipation"
        ),
        automated_test="tests/test_fatigue_fracture.py",
        status="experimental_foundation",
    ),
    BenchmarkSpec(
        identifier="external_mesh_named_sets",
        capability="external_mesh_interoperability",
        level="interface",
        reference="meshio cell blocks and named set conversion",
        criterion="volume and boundary memberships survive in XDMF tags and manifest",
        automated_test="tests/test_mesh_formats.py::test_conversion_can_preserve_boundary_sets_in_a_separate_xdmf",
    ),
    BenchmarkSpec(
        identifier="abaqus_equation_affine_reduction",
        capability="abaqus_periodic_equations",
        level="interface",
        reference="Abaqus/Standard first-term equation elimination",
        criterion=(
            "continued *EQUATION terms preserve source labels; chained affine "
            "relations reconstruct prescribed offsets exactly and cycles fail"
        ),
        automated_test="tests/test_abaqus_interop.py",
    ),
    BenchmarkSpec(
        identifier="abaqus_c3d10h_periodic_finite_deformation",
        capability="abaqus_periodic_equations",
        level="finite_element",
        reference="examples/abaqus_c3d10h_periodic_cell; 3D porous periodic cell",
        criterion=(
            "14,942 C3D10H nodes and 8,781 cells import directly; all 4,212 equations "
            "have zero mismatch; automatic increments reach the target with "
            "positive sampled J and recorded convergence evidence"
        ),
        automated_test=(
            "examples/abaqus_c3d10h_periodic_cell/case.py "
            "--displacement 0.20"
        ),
        status="manual_regression",
    ),
)


def list_benchmarks(*, capability: str | None = None) -> tuple[BenchmarkSpec, ...]:
    """Return all benchmarks or those for one capability."""

    if capability is None:
        return _BENCHMARKS
    selected = str(capability).lower().replace("-", "_")
    return tuple(item for item in _BENCHMARKS if item.capability == selected)


def benchmark(identifier: str) -> BenchmarkSpec:
    """Return one benchmark by stable identifier."""

    for item in _BENCHMARKS:
        if item.identifier == identifier:
            return item
    raise KeyError(
        f"Unknown benchmark {identifier!r}; "
        f"available={tuple(item.identifier for item in _BENCHMARKS)}."
    )
