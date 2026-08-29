"""Deterministic evidence driver for the first multi-void J2 RVE Golden.

The driver deliberately separates three kinds of evidence:

* fixed-stack Golden candidates: macroscopic stress, physically weighted PEEQ
  mean/P95, and meshed solid fraction;
* mesh-sensitive diagnostics: PEEQ P99/maximum and minimum local ``J``;
* invariant gates: periodicity, Hill--Mandel work, stored-energy
  decomposition, field availability, and physical-weight semantics.

One deterministic realization is a software-regression asset, not a
statistically representative porous material.  Numerical Golden values should
only be promoted after successive-mesh and two-rank evidence have been stored.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import (
    checkpointing,
    constitutive,
    fields,
    models,
    provenance,
    results,
    solvers,
    steps,
    studies,
)

from periodic_void_fixture import (
    PERIODIC_VOID_GMSH_OPTIONS,
    periodic_multi_spherical_void_cell,
    sample_hard_core_spherical_voids,
)


SCHEMA = "agentfem.multi-void-rve-golden-candidate.v1"
DEFAULT_MESH_SIZE = 0.16
DEFAULT_INCREMENTS = 2
DEFAULT_STRETCH = 1.004
MATERIAL_INPUT = {
    "young": 200000.0,
    "poisson": 0.3,
    "yield_stress": 200.0,
    "hardening_modulus": 2000.0,
    "tangent_relative_step": 2.0e-6,
}
REQUIRED_FIELDS = {
    "Displacement",
    "F",
    "P",
    "S",
    "MISES",
    "SENER",
    "ELENER",
    "HARDENER",
    "FP",
    "PEEQ",
}


@dataclass(frozen=True)
class CandidateCase:
    """One built case shared by Golden and checkpoint evidence."""

    fixture: object
    step: object
    periodicity: object
    solver_options: object


def deterministic_realization():
    """Return the versioned four-void identity used by this driver."""

    return sample_hard_core_spherical_voids(
        side_length=1.0,
        count=4,
        radius=0.09,
        seed=1729,
        minimum_inter_void_clearance=0.04,
        minimum_boundary_clearance=0.03,
        maximum_attempts=500,
    )


def evidence_policy() -> dict[str, object]:
    """Machine-readable promotion policy for the first multi-void Golden."""

    return {
        "fixed_stack_golden_candidates": [
            "homogenized_first_piola_stress",
            "peeq_mean",
            "peeq_p95",
            "solid_reference_fraction",
        ],
        "mesh_sensitive_diagnostics": [
            "peeq_p99",
            "peeq_maximum",
            "minimum_quadrature_j",
        ],
        "threshold_invariants": [
            "periodic_pairing",
            "periodic_equation_mismatch",
            "periodic_realization_clearance",
            "void_surface_identity",
            "geometry_representation",
            "positive_deformation_jacobian",
            "plastic_flow_reached",
            "hill_mandel_work",
            "stored_energy_components",
            "physical_weighted_statistics",
            "provider_fields",
        ],
        "invariant_limits": {
            "periodic_pairing_per_side_length": 1.0e-12,
            "periodic_equation_mismatch_per_side_length": 1.0e-10,
            "minimum_quadrature_j": 0.99,
            "minimum_plastic_flow": 1.0e-4,
            "maximum_hill_mandel_relative_error": 1.0e-8,
            "maximum_sener_component_closure_error": 1.0e-12,
            "maximum_geometry_relative_error": 1.0e-2,
        },
        "excluded_from_golden": {
            "peeq_maximum": "localization and mesh sensitive",
            "peeq_p99": "retained until multi-void refinement is demonstrated",
            "minimum_quadrature_j": "safety gate rather than target value",
            "hill_mandel_relative_error": "consistency residual, expected near zero",
            "stored_energy_closure": "identity residual, expected near zero",
            "runtime_seconds": "hardware and load dependent",
        },
        "promotion_requires": [
            "successive-mesh stability",
            "two-rank invariant equivalence",
            "checkpoint/restart equivalence on the realized mesh",
        ],
    }


def candidate_solver_options():
    """Return the single versioned nonlinear-solver contract."""

    return solvers.newton(
        relative_tolerance=2.0e-7,
        absolute_tolerance=1.0e-8,
        maximum_iterations=25,
        line_search="backtracking",
        linear_solver=solvers.direct_solver(),
    )


def build_candidate_case(
    *,
    comm=MPI.COMM_WORLD,
    mesh_size: float = DEFAULT_MESH_SIZE,
    increments: int = DEFAULT_INCREMENTS,
    progress: bool = False,
    name: str = "finite_strain_j2_deterministic_multi_void_candidate",
) -> CandidateCase:
    """Build the one scientific input consumed by all promotion drivers."""

    fixture = periodic_multi_spherical_void_cell(
        comm,
        realization=deterministic_realization(),
        mesh_size=float(mesh_size),
        stretch=DEFAULT_STRETCH,
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name=name,
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(**MATERIAL_INPUT)
    )
    periodicity = model.constraint(fixture.constraint(displacement))
    solver_options = candidate_solver_options()
    step = model.step(
        target=displacement,
        material=material,
        constraints=periodicity,
        incrementation=steps.fixed(int(increments)),
        solver_options=solver_options,
        quadrature_degree=1,
        output_every=1,
        progress=bool(progress),
    )
    return CandidateCase(
        fixture=fixture,
        step=step,
        periodicity=periodicity,
        solver_options=solver_options,
    )


def run_candidate(
    *,
    comm=MPI.COMM_WORLD,
    mesh_size: float = DEFAULT_MESH_SIZE,
    increments: int = DEFAULT_INCREMENTS,
    progress: bool = False,
) -> dict[str, object]:
    """Run the deterministic realization and return JSON-safe evidence."""

    started = time.perf_counter()
    realization = deterministic_realization()
    policy = evidence_policy()
    limits = policy["invariant_limits"]
    case = build_candidate_case(
        comm=comm,
        mesh_size=mesh_size,
        increments=increments,
        progress=bool(progress),
    )
    fixture = case.fixture
    step = case.step
    periodicity = case.periodicity
    solver_options = case.solver_options
    result = step.solve_result()
    frames = results.homogenize_periodic_path(
        step.snapshots,
        step.state_transaction.material,
        constraint=periodicity,
    )
    hill_mandel = results.hill_mandel_periodic_path(
        step.snapshots,
        step.state_transaction.material,
        constraint=periodicity,
        frames=frames,
    )
    statistics = step.response.state.committed[
        "equivalent_plastic_strain"
    ].weighted_statistics(quantiles=(0.95, 0.99))

    local_minimum_j = min(
        float(np.min(np.linalg.det(snapshot.fields["F"].owned_values)))
        for snapshot in step.snapshots
    )
    minimum_j = float(comm.allreduce(local_minimum_j, op=MPI.MIN))
    local_energy_closure = max(
        float(
            np.max(
                np.abs(
                    snapshot.fields["SENER"].owned_values
                    - snapshot.fields["ELENER"].owned_values
                    - snapshot.fields["HARDENER"].owned_values
                ),
                initial=0.0,
            )
        )
        for snapshot in step.snapshots
    )
    energy_closure = float(comm.allreduce(local_energy_closure, op=MPI.MAX))
    final = frames[-1]
    exact_solid_fraction = fixture.exact_solid_volume / fixture.cell_reference_volume
    accepted_load_factors = [
        float(snapshot.load_factor) for snapshot in step.snapshots
    ]
    constraint_identity = periodicity.scientific_identity()
    scientific_input = {
        "realization": realization.scientific_identity(),
        "mesh": {
            "generator": "gmsh-periodic-first-order-tetrahedra",
            "nominal_size": float(mesh_size),
            "exact_solid_fraction": exact_solid_fraction,
            "generation_options": PERIODIC_VOID_GMSH_OPTIONS,
        },
        "material": dict(MATERIAL_INPUT),
        "macroscopic_deformation_gradient": (
            fixture.deformation_gradient.tolist()
        ),
        "constraint_fingerprint": constraint_identity["fingerprint"],
        "constraint_equation_count": int(constraint_identity["equation_count"]),
        "increments": int(increments),
        "quadrature_degree": 1,
        "solver": solver_options.summary(),
    }
    global_cells = int(fixture.domain.topology.index_map(3).size_global)
    maximum_hill_mandel_error = max(
        float(item.relative_error) for item in hill_mandel
    )
    fields_present = sorted(set(result.fields))
    gates = {
        "step_completed": result.status == "completed",
        "global_newton_converged": bool(step.last_solve_info.converged),
        "final_load_factor": bool(
            np.isclose(step.accepted_load_factor, 1.0, rtol=0.0, atol=1.0e-14)
        ),
        "accepted_path_monotone": bool(
            accepted_load_factors[0] == 0.0
            and accepted_load_factors[-1] == 1.0
            and np.all(np.diff(accepted_load_factors) > 0.0)
        ),
        "periodic_pairing": (
            float(fixture.periodic_pairing_error)
            < limits["periodic_pairing_per_side_length"] * fixture.side_length
        ),
        "periodic_equation_mismatch": (
            float(periodicity.mismatch())
            < limits["periodic_equation_mismatch_per_side_length"]
            * fixture.side_length
        ),
        "periodic_realization_clearance": (
            realization.observed_periodic_inter_void_clearance is not None
            and realization.observed_periodic_inter_void_clearance
            >= realization.minimum_inter_void_clearance
        ),
        "void_surface_identity": fixture.void_surface_count
        == len(realization.spheres),
        "geometry_representation": (
            abs(final.solid_reference_fraction - exact_solid_fraction)
            / exact_solid_fraction
            < limits["maximum_geometry_relative_error"]
        ),
        "positive_deformation_jacobian": (
            minimum_j > limits["minimum_quadrature_j"]
        ),
        "plastic_flow_reached": (
            float(statistics.maximum) > limits["minimum_plastic_flow"]
        ),
        "hill_mandel_work": (
            maximum_hill_mandel_error
            < limits["maximum_hill_mandel_relative_error"]
        ),
        "stored_energy_components": (
            energy_closure
            < limits["maximum_sener_component_closure_error"]
        ),
        "provider_fields": REQUIRED_FIELDS <= set(result.fields),
        "weighted_sample_count": int(statistics.sample_count) == global_cells,
        "weighted_measure_matches_homogenization": bool(
            np.isclose(
                statistics.total_weight,
                final.solid_reference_fraction,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ),
        "weighted_statistics_semantics": (
            statistics.location == "quadrature_points"
            and statistics.representation == "raw_quadrature_values"
            and statistics.operation == "physical_weighted_distribution"
        ),
    }
    return {
        "schema": SCHEMA,
        "status": (
            "candidate_requires_refinement_mpi_and_restart"
            if all(gates.values())
            else "failed_invariant_gate"
        ),
        "accepted_invariant_gates": bool(all(gates.values())),
        "policy": policy,
        "execution": {
            "rank_count": int(comm.size),
            "runtime_seconds": float(time.perf_counter() - started),
            "global_cells": global_cells,
            "accepted_load_factors": accepted_load_factors,
            "fields": fields_present,
            "restart_exercised": False,
            "runtime_manifest": provenance.runtime_manifest(),
        },
        "identities": {
            "realization": realization.scientific_identity(),
            "portable_mesh": checkpointing.mesh_portable_identity(fixture.domain),
            "constraint": constraint_identity,
            "scientific_input": scientific_input,
            "scientific_input_fingerprint": provenance.content_fingerprint(
                scientific_input
            ),
        },
        "quantities": {
            "homogenized_first_piola_stress": (
                np.asarray(final.first_piola_stress, dtype=float).tolist()
            ),
            "peeq_mean": float(statistics.mean),
            "peeq_p95": float(statistics.quantiles[0.95]),
            "peeq_p99": float(statistics.quantiles[0.99]),
            "solid_reference_fraction": float(final.solid_reference_fraction),
        },
        "diagnostics": {
            "peeq_maximum": float(statistics.maximum),
            "minimum_quadrature_j": minimum_j,
            "maximum_hill_mandel_relative_error": maximum_hill_mandel_error,
            "maximum_sener_component_closure_error": energy_closure,
            "geometry_relative_error": float(
                abs(final.solid_reference_fraction - exact_solid_fraction)
                / exact_solid_fraction
            ),
            "weighted_sample_count": int(statistics.sample_count),
            "weighted_total_measure": float(statistics.total_weight),
            "void_surface_count": int(fixture.void_surface_count),
            "periodic_inter_void_clearance": float(
                realization.observed_periodic_inter_void_clearance
            ),
        },
        "gates": gates,
    }


def compare_refinement_evidence(
    coarse: dict[str, object],
    medium: dict[str, object],
    fine: dict[str, object],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    """Compare three evidence records and fail closed on incompatible inputs.

    The default limits are provisional software-promotion thresholds, not an
    external RVE accuracy standard.  Geometry error need not decrease
    monotonically for unstructured curved-boundary meshes; the finest error
    and successive solid-volume change are checked explicitly instead.
    """

    limits = {
        "macro_first_piola": 5.0e-3,
        "peeq_mean": 2.0e-3,
        "peeq_p95": 1.5e-2,
        "solid_reference_fraction": 2.0e-3,
        "finest_geometry_relative_error": 1.0e-2,
    }
    if thresholds is not None:
        unknown = set(thresholds) - set(limits)
        if unknown:
            raise ValueError(
                "Unknown multi-void refinement threshold(s): "
                + ", ".join(sorted(unknown))
            )
        limits.update({name: float(value) for name, value in thresholds.items()})
    if any(not np.isfinite(value) or value <= 0.0 for value in limits.values()):
        raise ValueError("Refinement thresholds must be finite and positive.")

    levels = (coarse, medium, fine)
    required_sections = {
        "schema",
        "accepted_invariant_gates",
        "execution",
        "identities",
        "quantities",
        "diagnostics",
    }
    for index, evidence in enumerate(levels):
        if not isinstance(evidence, dict) or not required_sections <= set(evidence):
            raise ValueError(f"Refinement level {index} is not complete evidence.")
        if evidence["schema"] != SCHEMA:
            raise ValueError(f"Refinement level {index} has an incompatible schema.")

    realization_fingerprints = [
        item["identities"]["realization"]["fingerprint"] for item in levels
    ]
    if len(set(realization_fingerprints)) != 1:
        raise ValueError("Refinement evidence uses different void realizations.")
    mesh_sizes = [
        float(item["identities"]["scientific_input"]["mesh"]["nominal_size"])
        for item in levels
    ]
    global_cells = [int(item["execution"]["global_cells"]) for item in levels]
    if not (mesh_sizes[0] > mesh_sizes[1] > mesh_sizes[2]):
        raise ValueError("Refinement mesh sizes must be strictly decreasing.")
    if not (global_cells[0] < global_cells[1] < global_cells[2]):
        raise ValueError("Refinement global cell counts must be strictly increasing.")

    def scalar_change(left: float, right: float) -> float:
        return float(
            abs(float(left) - float(right))
            / max(abs(float(right)), np.finfo(float).tiny)
        )

    def tensor_change(left, right) -> float:
        first = np.asarray(left, dtype=float).reshape(-1)
        second = np.asarray(right, dtype=float).reshape(-1)
        return float(
            np.linalg.norm(first - second)
            / max(np.linalg.norm(second), np.finfo(float).tiny)
        )

    intervals = []
    for left, right in zip(levels, levels[1:]):
        intervals.append(
            {
                "from_mesh_size": float(
                    left["identities"]["scientific_input"]["mesh"]["nominal_size"]
                ),
                "to_mesh_size": float(
                    right["identities"]["scientific_input"]["mesh"]["nominal_size"]
                ),
                "macro_first_piola": tensor_change(
                    left["quantities"]["homogenized_first_piola_stress"],
                    right["quantities"]["homogenized_first_piola_stress"],
                ),
                "peeq_mean": scalar_change(
                    left["quantities"]["peeq_mean"],
                    right["quantities"]["peeq_mean"],
                ),
                "peeq_p95": scalar_change(
                    left["quantities"]["peeq_p95"],
                    right["quantities"]["peeq_p95"],
                ),
                "peeq_p99_diagnostic": scalar_change(
                    left["quantities"]["peeq_p99"],
                    right["quantities"]["peeq_p99"],
                ),
                "peeq_maximum_diagnostic": scalar_change(
                    left["diagnostics"]["peeq_maximum"],
                    right["diagnostics"]["peeq_maximum"],
                ),
                "solid_reference_fraction": scalar_change(
                    left["quantities"]["solid_reference_fraction"],
                    right["quantities"]["solid_reference_fraction"],
                ),
            }
        )
    latest = intervals[-1]
    gates = {
        "all_level_invariants": all(
            bool(item["accepted_invariant_gates"]) for item in levels
        ),
        "macro_first_piola": latest["macro_first_piola"]
        < limits["macro_first_piola"],
        "peeq_mean": latest["peeq_mean"] < limits["peeq_mean"],
        "peeq_p95": latest["peeq_p95"] < limits["peeq_p95"],
        "solid_reference_fraction": latest["solid_reference_fraction"]
        < limits["solid_reference_fraction"],
        "finest_geometry_relative_error": float(
            fine["diagnostics"]["geometry_relative_error"]
        )
        < limits["finest_geometry_relative_error"],
    }
    return {
        "schema": "agentfem.multi-void-rve-refinement-certificate.v1",
        "accepted": bool(all(gates.values())),
        "realization_fingerprint": realization_fingerprints[0],
        "mesh_sizes": mesh_sizes,
        "global_cells": global_cells,
        "thresholds": limits,
        "interval_relative_changes": intervals,
        "geometry_relative_errors": [
            float(item["diagnostics"]["geometry_relative_error"])
            for item in levels
        ],
        "gates": gates,
        "wording": (
            "Successive-refinement stability under provisional software "
            "promotion limits; not formal asymptotic convergence or GCI."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-size", type=float, default=DEFAULT_MESH_SIZE)
    parser.add_argument("--increments", type=int, default=DEFAULT_INCREMENTS)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    options = _parse_args()
    evidence = run_candidate(
        mesh_size=options.mesh_size,
        increments=options.increments,
        progress=options.progress,
    )
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        payload = json.dumps(evidence, indent=2, sort_keys=True)
        if options.output is not None:
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
    if not evidence["accepted_invariant_gates"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
