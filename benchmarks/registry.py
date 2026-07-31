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
        criterion="uninterrupted and checkpoint/restarted paths recover U and PEEQ",
        automated_test="tests/test_p1_platform.py::test_global_j2_checkpoint_restart_matches_uninterrupted_path",
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
        identifier="creep_closed_forms",
        capability="power_law_creep",
        level="material_point",
        reference="Abaqus verification of creep integration",
        criterion="constant-stress creep matches the integrated power law and relaxation decays",
        automated_test="tests/test_constitutive_models.py::test_power_law_creep_matches_constant_stress_and_relaxation_solutions",
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
        identifier="abaqus_c3d10_periodic_finite_deformation",
        capability="abaqus_periodic_equations",
        level="finite_element",
        reference="examples/abaqus_c3d10_periodic_cell; 3D porous periodic cell",
        criterion=(
            "14,942 C3D10 nodes and 8,781 cells import; all 4,212 equations "
            "have zero mismatch; automatic increments reach the target with "
            "positive sampled J and recorded convergence evidence"
        ),
        automated_test=(
            "examples/abaqus_c3d10_periodic_cell/agentfem_periodic_hyperelastic.py "
            "--stretch 1.20"
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
