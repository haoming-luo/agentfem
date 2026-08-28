"""Two-rank Golden acceptance for a true-geometry finite-strain J2 RVE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import (
    benchmarks,
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

from periodic_void_fixture import periodic_spherical_void_cell


BENCHMARK_ID = "agentfem.benchmark.finite_strain_j2_periodic_void"
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


def _benchmark_card() -> dict[str, object]:
    path = (
        Path(benchmarks.__file__).resolve().parents[1]
        / "knowledge"
        / "benchmarks"
        / "finite_strain_j2_periodic_void.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _json_value(value):
    selected = np.asarray(value)
    if selected.ndim == 0:
        return selected.item()
    return selected.tolist()


def _golden_comparisons(golden, actual) -> dict[str, dict[str, object]]:
    comparisons = {}
    for quantity in golden.quantities:
        actual_value = actual[quantity.name]
        actual_array = np.asarray(actual_value, dtype=float)
        expected_array = np.asarray(quantity.expected, dtype=float)
        denominator = max(
            float(np.linalg.norm(expected_array.reshape(-1))),
            np.finfo(float).tiny,
        )
        comparisons[quantity.name] = {
            "actual": _json_value(actual_value),
            "expected": _json_value(quantity.expected),
            "relative_error": float(
                np.linalg.norm((actual_array - expected_array).reshape(-1))
                / denominator
            ),
            "maximum_absolute_error": float(
                np.max(np.abs(actual_array - expected_array), initial=0.0)
            ),
            "relative_tolerance": float(quantity.relative_tolerance),
            "absolute_tolerance": float(quantity.absolute_tolerance),
            "accepted": bool(quantity.accepts(actual_value)),
        }
    return comparisons


def main(*, invariants_only: bool = False) -> None:
    comm = MPI.COMM_WORLD
    card = _benchmark_card()
    golden = benchmarks.golden_benchmark(BENCHMARK_ID)
    reference_input = card["regression_identity"]["scientific_input"]
    geometry_input = reference_input["geometry"]
    material_input = reference_input["material"]
    macro_deformation = np.asarray(
        reference_input["macroscopic_deformation_gradient"],
        dtype=float,
    )
    stretch = float(macro_deformation[0, 0])
    expected_macro_deformation = np.diag(
        (stretch, 1.0 / np.sqrt(stretch), 1.0 / np.sqrt(stretch))
    )
    if not np.allclose(
        macro_deformation,
        expected_macro_deformation,
        rtol=0.0,
        atol=2.0e-15,
    ):
        raise RuntimeError("The reference fixture requires isochoric diagonal Fbar.")
    fixture = periodic_spherical_void_cell(
        comm,
        side_length=float(geometry_input["side_length"]),
        void_radius=float(geometry_input["void_radius"]),
        mesh_size=float(geometry_input["mesh_size"]),
        stretch=stretch,
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=3,
        ),
        mesh=fixture.domain,
        name="parallel_finite_strain_j2_real_void_smoke",
    )
    displacement = model.field(fields.displacement(fixture.domain))
    material = model.material(
        constitutive.finite_strain_j2_logarithmic(
            young=float(material_input["young"]),
            poisson=float(material_input["poisson"]),
            yield_stress=float(material_input["yield_stress"]),
            hardening_modulus=float(material_input["hardening_modulus"]),
            tangent_relative_step=float(
                material_input["tangent_relative_step"]
            ),
        )
    )
    periodicity = model.constraint(fixture.constraint(displacement))
    newton_input = reference_input["newton"]
    solver_options = solvers.newton(
        relative_tolerance=float(newton_input["relative_tolerance"]),
        absolute_tolerance=float(newton_input["absolute_tolerance"]),
        maximum_iterations=int(newton_input["maximum_iterations"]),
        line_search=str(newton_input["line_search"]),
        linear_solver=solvers.direct_solver(),
    )
    quadrature_degree = int(reference_input["quadrature_degree"])
    step = model.step(
        target=displacement,
        material=material,
        constraints=periodicity,
        incrementation=steps.fixed(int(reference_input["increments"])),
        solver_options=solver_options,
        quadrature_degree=quadrature_degree,
        output_every=1,
        progress=False,
    )
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
    local_minimum_j = min(
        float(np.min(np.linalg.det(snapshot.fields["F"].owned_values)))
        for snapshot in step.snapshots
    )
    minimum_j = float(comm.allreduce(local_minimum_j, op=MPI.MIN))
    peeq = step.response.state.committed["equivalent_plastic_strain"]
    statistics = peeq.weighted_statistics(quantiles=(0.95, 0.99))
    maximum_peeq = statistics.maximum
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
    macro_first_piola = np.asarray(final.first_piola_stress, dtype=float)
    actual_quantities = {
        "homogenized_first_piola_stress": macro_first_piola.reshape(-1),
        "peeq_mean": float(statistics.mean),
        "peeq_p95": float(statistics.quantiles[0.95]),
        "peeq_p99": float(statistics.quantiles[0.99]),
        "solid_reference_fraction": float(final.solid_reference_fraction),
    }
    golden_comparisons = _golden_comparisons(golden, actual_quantities)

    regression_identity = card["regression_identity"]
    expected_mesh_identity = regression_identity["mesh_identity"]
    mesh_identity = checkpointing.mesh_portable_identity(fixture.domain)
    constraint_identity = periodicity.scientific_identity()
    expected_scientific_input = regression_identity["scientific_input"]
    accepted_load_factors = [
        float(snapshot.load_factor) for snapshot in step.snapshots
    ]
    expected_load_factors = [
        float(value)
        for value in expected_scientific_input["accepted_load_factors"]
    ]
    solver_summary = solver_options.summary()
    linear_summary = solver_summary["linear_solver"]
    scientific_input = {
        "geometry": {
            "side_length": fixture.side_length,
            "void_radius": fixture.void_radius,
            "mesh_size": float(geometry_input["mesh_size"]),
            "exact_solid_volume": fixture.exact_solid_volume,
        },
        "material": {
            name: float(material_input[name])
            for name in (
                "young",
                "poisson",
                "yield_stress",
                "hardening_modulus",
                "tangent_relative_step",
            )
        },
        "macroscopic_deformation_gradient": (
            fixture.deformation_gradient.tolist()
        ),
        "constraint_fingerprint": constraint_identity["fingerprint"],
        "equation_count": int(constraint_identity["equation_count"]),
        "increments": int(reference_input["increments"]),
        "accepted_load_factors": accepted_load_factors,
        "quadrature_degree": quadrature_degree,
        "newton": {
            "relative_tolerance": float(
                solver_summary["relative_tolerance"]
            ),
            "absolute_tolerance": float(
                solver_summary["absolute_tolerance"]
            ),
            "maximum_iterations": int(
                solver_summary["maximum_iterations"]
            ),
            "line_search": solver_summary["line_search"],
            "linear_solver": (
                f"{linear_summary['ksp_type']}_{linear_summary['pc_type']}"
            ),
        },
    }
    scientific_input_fingerprint = provenance.content_fingerprint(
        scientific_input
    )
    expected_statistics = regression_identity["weighted_statistics"]
    fields_present = sorted(set(result.fields))
    gates = {
        "two_mpi_ranks": int(comm.size) == 2,
        "step_completed": result.status == "completed",
        "global_newton_converged": bool(step.last_solve_info.converged),
        "final_load_factor": float(step.accepted_load_factor) == 1.0,
        "accepted_load_factors": accepted_load_factors
        == expected_load_factors,
        "portable_mesh_identity": mesh_identity == expected_mesh_identity,
        "constraint_fingerprint": constraint_identity["fingerprint"]
        == expected_scientific_input["constraint_fingerprint"],
        "constraint_equation_count": int(constraint_identity["equation_count"])
        == int(expected_scientific_input["equation_count"]),
        "scientific_input": scientific_input == expected_scientific_input,
        "scientific_input_fingerprint": scientific_input_fingerprint
        == regression_identity["scientific_input_fingerprint"],
        "periodic_pairing": float(fixture.periodic_pairing_error)
        < 1.0e-12 * fixture.side_length,
        "periodic_equation_mismatch": float(periodicity.mismatch())
        < 1.0e-10 * fixture.side_length,
        "positive_deformation_jacobian": minimum_j > 0.99,
        "plastic_flow_reached": float(maximum_peeq) > 1.0e-4,
        "hill_mandel_work": max(
            item.relative_error for item in hill_mandel
        )
        < 1.0e-8,
        "stored_energy_components": energy_closure < 1.0e-12,
        "provider_fields": REQUIRED_FIELDS <= set(result.fields),
        "weighted_sample_count": int(statistics.sample_count)
        == int(expected_statistics["sample_count"]),
        "weighted_total_measure": bool(
            np.isclose(
                statistics.total_weight,
                expected_statistics["total_weight"],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ),
        "weighted_measure_matches_homogenization": bool(
            np.isclose(
                statistics.total_weight,
                final.solid_reference_fraction,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ),
        "weighted_statistics_semantics": (
            statistics.location == expected_statistics["location"]
            and statistics.representation
            == expected_statistics["representation"]
            and statistics.operation == expected_statistics["operation"]
        ),
    }
    golden_accepted = {
        name: bool(item["accepted"])
        for name, item in golden_comparisons.items()
    }
    reference_gate_names = {
        "portable_mesh_identity",
        "constraint_fingerprint",
        "constraint_equation_count",
        "scientific_input",
        "scientific_input_fingerprint",
        "weighted_sample_count",
        "weighted_total_measure",
    }
    required_gates = {
        name: passed
        for name, passed in gates.items()
        if not invariants_only or name not in reference_gate_names
    }
    accepted = all(required_gates.values()) and (
        invariants_only or all(golden_accepted.values())
    )
    evidence = {
        "schema": "agentfem.benchmark-evidence.v1",
        "benchmark": BENCHMARK_ID,
        "reference_version": golden.reference_version,
        "mode": (
            "cross-platform-invariants"
            if invariants_only
            else "fixed-stack-golden"
        ),
        "accepted": bool(accepted),
        "execution": {
            "rank_count": int(comm.size),
            "status": result.status,
            "converged": bool(step.last_solve_info.converged),
            "global_cells": int(
                fixture.domain.topology.index_map(3).size_global
            ),
            "accepted_load_factors": accepted_load_factors,
            "fields": fields_present,
        },
        "identities": {
            "portable_mesh": {
                "actual": mesh_identity,
                "expected": expected_mesh_identity,
                "accepted": bool(gates["portable_mesh_identity"]),
            },
            "constraint": {
                "actual_fingerprint": constraint_identity["fingerprint"],
                "expected_fingerprint": expected_scientific_input[
                    "constraint_fingerprint"
                ],
                "equation_count": int(constraint_identity["equation_count"]),
                "expected_equation_count": int(
                    expected_scientific_input["equation_count"]
                ),
                "accepted": bool(
                    gates["constraint_fingerprint"]
                    and gates["constraint_equation_count"]
                ),
            },
            "scientific_input": {
                "actual": scientific_input,
                "expected": expected_scientific_input,
                "actual_fingerprint": scientific_input_fingerprint,
                "expected_fingerprint": regression_identity[
                    "scientific_input_fingerprint"
                ],
                "accepted": bool(
                    gates["scientific_input"]
                    and gates["scientific_input_fingerprint"]
                ),
            },
        },
        "observables": {
            "homogenized_first_piola_stress": macro_first_piola.tolist(),
            "peeq_weighted_statistics": statistics.summary(),
            "peeq_maximum_diagnostic": float(maximum_peeq),
            "solid_reference_fraction": float(
                final.solid_reference_fraction
            ),
            "periodic_pairing_error": float(
                fixture.periodic_pairing_error
            ),
            "periodic_equation_mismatch": float(periodicity.mismatch()),
            "minimum_quadrature_j": minimum_j,
            "maximum_hill_mandel_relative_error": float(
                max(item.relative_error for item in hill_mandel)
            ),
            "maximum_sener_component_closure_error": energy_closure,
        },
        "golden": golden_comparisons,
        "gates": gates,
        "failed_checks": sorted(
            [name for name, passed in required_gates.items() if not passed]
            + [
                f"golden:{name}"
                for name, passed in golden_accepted.items()
                if not passed and not invariants_only
            ]
        ),
        "non_gating_reference_mismatches": sorted(
            name
            for name in reference_gate_names
            if not gates[name]
        ),
    }
    if comm.rank == 0:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    comm.Barrier()
    if not accepted:
        raise RuntimeError(
            "The distributed periodic-void J2 Golden contract failed: "
            + ", ".join(evidence["failed_checks"])
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the distributed periodic-void J2 acceptance case."
    )
    parser.add_argument(
        "--invariants-only",
        action="store_true",
        help=(
            "Gate cross-platform MPI physics invariants without requiring the "
            "Darwin/arm64 fixed-stack Golden identity."
        ),
    )
    arguments = parser.parse_args()
    main(invariants_only=arguments.invariants_only)
