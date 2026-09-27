"""Run the AgentFEM candidate for the Lewandowski et al. beam gate.

This is an opt-in evidence driver, not a unit test.  It writes the candidate
curve and an assessment that remains incomplete unless an independently
executed, pinned MGIS/FEniCS curve and all promotion evidence are supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from mpi4py import MPI

import agentfem
from agentfem import (
    constitutive,
    fields,
    mesh,
    models,
    results,
    solvers,
    steps,
    studies,
)
from agentfem.provenance import runtime_manifest

from lewandowski_2023_self_weight_beam_fixture import (
    CANDIDATE_ABSOLUTE_RESIDUAL_TOLERANCE,
    CANDIDATE_LINE_SEARCH,
    CANDIDATE_MAXIMUM_ITERATIONS,
    CANDIDATE_RELATIVE_RESIDUAL_TOLERANCE,
    DEFINITION,
    UPSTREAM_BEHAVIOUR_SHA256,
    UPSTREAM_COMMIT,
    UPSTREAM_SOLVER_SHA256,
    assess_external_curve,
)


def _candidate_step(
    comm,
    *,
    subdivisions,
    increments,
    adaptive=False,
    progress=False,
    line_search=CANDIDATE_LINE_SEARCH,
    maximum_iterations=CANDIDATE_MAXIMUM_ITERATIONS,
    absolute_tolerance=CANDIDATE_ABSOLUTE_RESIDUAL_TOLERANCE,
    relative_tolerance=CANDIDATE_RELATIVE_RESIDUAL_TOLERANCE,
    tangent_evaluation="analytic_spectral",
):
    definition = DEFINITION
    domain = mesh.cuboid(
        (0.0, -0.5 * definition.width, -0.5 * definition.height),
        (definition.length, 0.5 * definition.width, 0.5 * definition.height),
        subdivisions,
        comm=comm,
        cell_type="tetrahedron",
    )
    study = studies.nonlinear_static(physics="solid_mechanics", dimension=3)
    model = models.create(
        study=study,
        mesh=domain,
        name="lewandowski_2023_self_weight_beam_candidate",
    )
    displacement = model.field(
        fields.displacement(domain, degree=definition.displacement_degree)
    )
    left = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], 0.0),
        name="left_clamp",
        tag=1,
    )
    right = mesh.boundary(
        domain,
        lambda x: np.isclose(x[0], definition.length),
        name="right_symmetry",
        tag=2,
    )
    model.fix(displacement, on=left, value=(0.0, 0.0, 0.0))
    model.fix(displacement, on=right, component=0, value=0.0)
    material = constitutive.finite_strain_j2_logarithmic(
        young=definition.young,
        poisson=definition.poisson,
        yield_stress=definition.yield_stress,
        hardening_modulus=definition.hardening_modulus,
        tangent_evaluation=tangent_evaluation,
    )
    model.material(material)
    model.body_force(
        (0.0, 0.0, -definition.maximum_body_force),
        target=displacement,
        name="self_weight_body_force",
    )
    if adaptive:
        incrementation = steps.automatic(
            initial=1.0 / increments,
            minimum=1.0 / (64.0 * increments),
            maximum=1.0 / increments,
            max_increments=16 * increments,
            max_cutbacks=8,
            cutback_factor=0.5,
            growth_factor=1.5,
            fast_iterations=4,
            slow_iterations=10,
        )
    else:
        incrementation = steps.fixed(increments)
    step = model.step(
        target=displacement,
        material=material,
        incrementation=incrementation,
        solver_options=solvers.newton(
            relative_tolerance=float(relative_tolerance),
            absolute_tolerance=float(absolute_tolerance),
            maximum_iterations=int(maximum_iterations),
            line_search=line_search,
        ),
        progress=progress,
        name="lewandowski_2023_self_weight_beam_candidate",
    )
    return step, displacement


def _read_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, names=True, delimiter=",")
    names = tuple(table.dtype.names or ())
    required = ("load_factor", "downward_displacement_m")
    if not set(required).issubset(names):
        raise ValueError(f"Reference CSV must contain columns {required}.")
    return (
        np.atleast_1d(table["load_factor"]).astype(float),
        np.atleast_1d(table["downward_displacement_m"]).astype(float),
    )


def _write_candidate_curve(
    output: Path,
    load_factors: list[float],
    downward_displacements: list[float],
) -> None:
    """Atomically publish the accepted prefix of a long candidate run."""

    output.mkdir(parents=True, exist_ok=True)
    destination = output / "candidate_curve.csv"
    temporary = output / ".candidate_curve.csv.tmp"
    np.savetxt(
        temporary,
        np.column_stack((load_factors, downward_displacements)),
        delimiter=",",
        header="load_factor,downward_displacement_m",
        comments="",
    )
    temporary.replace(destination)


def _increment_performance(records) -> dict[str, object]:
    """Summarize rank-reduced nonlinear stage timings without hiding detail."""

    selected = tuple(records)
    names = (
        "total_seconds",
        "material_update_seconds",
        "residual_assembly_seconds",
        "tangent_assembly_seconds",
        "linear_solve_seconds",
        "line_search_seconds",
    )
    return {
        "schema": "agentfem.finite-strain-j2-increment-performance.v1",
        "timing_basis": "maximum_rank_wall_clock_perf_counter",
        "accepted_increment_count": len(selected),
        "totals": {
            name: float(sum(getattr(record, name, 0.0) for record in selected))
            for name in names
        },
        "linear_solve_calls": int(
            sum(getattr(record, "linear_solve_calls", 0) for record in selected)
        ),
        "linear_iterations": int(
            sum(getattr(record, "linear_iterations", 0) for record in selected)
        ),
        "increments": [
            {
                "increment": int(record.increment),
                "load_factor": float(record.load_factor),
                **{name: float(getattr(record, name, 0.0)) for name in names},
                "linear_solve_calls": int(
                    getattr(record, "linear_solve_calls", 0)
                ),
                "linear_iterations": int(
                    getattr(record, "linear_iterations", 0)
                ),
                "linear_converged_reasons": list(
                    getattr(record, "linear_converged_reasons", ())
                ),
            }
            for record in selected
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--subdivisions", type=int, nargs=3, default=DEFINITION.subdivisions)
    parser.add_argument("--increments", type=int, default=DEFINITION.increments)
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Allow fail-closed cutbacks from a maximum 1/increments load step.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--maximum-iterations",
        type=int,
        default=CANDIDATE_MAXIMUM_ITERATIONS,
        help="Maximum Newton corrections allowed for each fixed increment.",
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=CANDIDATE_ABSOLUTE_RESIDUAL_TOLERANCE,
        help="Absolute global force-residual tolerance for Newton convergence.",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=CANDIDATE_RELATIVE_RESIDUAL_TOLERANCE,
        help="Relative global force-residual tolerance for Newton convergence.",
    )
    parser.add_argument(
        "--line-search",
        choices=("backtracking", "basic"),
        default=CANDIDATE_LINE_SEARCH,
        help=(
            "Newton globalization policy. The pinned upstream beam uses full "
            "Newton, so 'basic' is the benchmark default."
        ),
    )
    parser.add_argument(
        "--tangent-evaluation",
        choices=("analytic_spectral", "central_difference"),
        default="analytic_spectral",
        help="Algorithmic dP/dF implementation used by the candidate material.",
    )
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--promotion-evidence-json", type=Path)
    arguments = parser.parse_args()
    if any(value <= 0 for value in arguments.subdivisions):
        raise ValueError("All subdivisions must be positive.")
    if arguments.increments < 5:
        raise ValueError("At least five load increments are required.")
    if arguments.maximum_iterations <= 0:
        raise ValueError("maximum-iterations must be positive.")
    if arguments.absolute_tolerance <= 0.0:
        raise ValueError("absolute-tolerance must be positive.")
    if arguments.relative_tolerance <= 0.0:
        raise ValueError("relative-tolerance must be positive.")

    comm = MPI.COMM_WORLD
    started = time.perf_counter()
    step, displacement = _candidate_step(
        comm,
        subdivisions=tuple(arguments.subdivisions),
        increments=arguments.increments,
        adaptive=arguments.adaptive,
        progress=arguments.progress,
        line_search=arguments.line_search,
        maximum_iterations=arguments.maximum_iterations,
        absolute_tolerance=arguments.absolute_tolerance,
        relative_tolerance=arguments.relative_tolerance,
        tangent_evaluation=arguments.tangent_evaluation,
    )
    factors = np.linspace(0.0, 1.0, arguments.increments + 1)
    downward = [0.0]
    accepted_factors = [0.0]
    if comm.rank == 0:
        _write_candidate_curve(arguments.output, accepted_factors, downward)
    failure = None
    try:
        for factor in factors[1:]:
            step.solve(until=float(factor))
            value = results.probe(displacement, at=DEFINITION.observer)
            downward.append(-float(value[2]))
            accepted_factors.append(float(factor))
            if comm.rank == 0:
                _write_candidate_curve(arguments.output, accepted_factors, downward)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"

    reference_load = None
    reference_displacement = None
    evidence = {}
    source = {}
    declared_reference_curve_sha256 = None
    actual_reference_curve_sha256 = None
    if arguments.reference_csv is not None:
        reference_load, reference_displacement = _read_reference(arguments.reference_csv)
        actual_reference_curve_sha256 = hashlib.sha256(
            arguments.reference_csv.read_bytes()
        ).hexdigest()
    if arguments.promotion_evidence_json is not None:
        promotion = json.loads(arguments.promotion_evidence_json.read_text())
        evidence = dict(promotion.get("evidence", {}))
        source = dict(promotion.get("source", {}))
        declared_reference_curve_sha256 = promotion.get(
            "reference_curve_sha256"
        )
    if comm.rank == 0:
        candidate_path = arguments.output / "candidate_curve.csv"
        candidate_curve_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    else:
        candidate_curve_sha256 = None
    candidate_curve_sha256 = comm.bcast(candidate_curve_sha256, root=0)
    elapsed_seconds = float(comm.allreduce(time.perf_counter() - started, op=MPI.MAX))
    if failure is None:
        assessment = assess_external_curve(
            candidate_load_factors=accepted_factors,
            candidate_displacements=downward,
            reference_load_factors=reference_load,
            reference_displacements=reference_displacement,
            reference_source_commit=source.get("commit"),
            reference_solver_sha256=source.get("solver_sha256"),
            reference_behaviour_sha256=source.get("behaviour_sha256"),
            reference_curve_sha256=actual_reference_curve_sha256,
            declared_reference_curve_sha256=declared_reference_curve_sha256,
            convergence_evidence=evidence,
        )
    else:
        assessment = {
            "status": "failed",
            "accepted": False,
            "reason": "candidate_solve_failed_before_complete_curve",
            "accepted_prefix_points": len(accepted_factors),
            "last_accepted_load_factor": float(accepted_factors[-1]),
            "complete_curve_comparison_performed": False,
        }
    manifest = {
        "schema": "agentfem.external-benchmark-candidate.v1",
        "benchmark": "lewandowski_2023_self_weight_beam",
        "status": "failed" if failure is not None else assessment["status"],
        "runtime": {
            "agentfem_version": agentfem.__version__,
            "agentfem_import_path": str(Path(agentfem.__file__).resolve()),
            "manifest": runtime_manifest(),
        },
        "scientific_definition": DEFINITION.summary(),
        "candidate": {
            "formulation": (
                "multiplicative_Fp_quadratic_Hencky_Kirchhoff_J2_"
                "linear_isotropic_hardening"
            ),
            "subdivisions": tuple(arguments.subdivisions),
            "increments": arguments.increments,
            "incrementation": "automatic_cutback" if arguments.adaptive else "fixed",
            "line_search": arguments.line_search,
            "maximum_iterations_limit": arguments.maximum_iterations,
            "absolute_residual_tolerance": arguments.absolute_tolerance,
            "relative_residual_tolerance": arguments.relative_tolerance,
            "tangent_evaluation": arguments.tangent_evaluation,
            "mpi_ranks": comm.size,
            "curve_file": "candidate_curve.csv",
            "curve_sha256": candidate_curve_sha256,
            "points": len(accepted_factors),
            "final_downward_displacement_m": float(downward[-1]),
            "elapsed_seconds_max_rank": elapsed_seconds,
            "accepted_increments": len(step.accepted_increments),
            "attempted_increments": len(step.attempted_increments),
            "maximum_newton_iterations": max(
                (record.iterations for record in step.accepted_increments),
                default=0,
            ),
            "performance": _increment_performance(step.accepted_increments),
            "final_plastic_points": int(step.state_transaction.last_plastic_points),
            "step": step.summary(),
        },
        "reference": {
            "formulation": (
                "MFront_total_Hencky_Hooke_Mises_linear_isotropic_hardening"
            ),
            "commit": UPSTREAM_COMMIT,
            "solver_sha256": UPSTREAM_SOLVER_SHA256,
            "behaviour_sha256": UPSTREAM_BEHAVIOUR_SHA256,
            "curve_supplied": reference_load is not None,
            "curve_sha256": actual_reference_curve_sha256,
            "declared_curve_sha256": declared_reference_curve_sha256,
        },
        "assessment": assessment,
        "failure": (
            {
                "message": failure,
                "last_attempt": (
                    step.attempted_increments[-1].as_dict()
                    if step.attempted_increments
                    else None
                ),
                "accepted_load_factor": float(step.accepted_load_factor),
            }
            if failure is not None
            else None
        ),
    }
    if comm.rank == 0:
        _write_candidate_curve(arguments.output, accepted_factors, downward)
        assessment_path = arguments.output / "assessment.json"
        assessment_temporary = arguments.output / ".assessment.json.tmp"
        assessment_temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        assessment_temporary.replace(assessment_path)
    comm.barrier()
    if failure is not None:
        raise RuntimeError(failure)


if __name__ == "__main__":
    main()
