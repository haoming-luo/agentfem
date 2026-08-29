"""Same-rank checkpoint/restart equivalence for the deterministic void RVE.

The uninterrupted and restarted branches rebuild the same deterministic Gmsh
realization independently.  The comparison covers the nodal solution, the
accepted solution boundary, every committed quadrature-state variable, the
accepted first-Piola response, derived constitutive fields, the final
homogenized frame, measured macroscopic deformation gradient, and the complete
accepted increment history.

This driver is an executable promotion asset rather than a default unit test.
It accepts a checkpoint root as its positional CLI argument.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import (
    checkpointing,
    results,
)

from multi_void_rve_golden_driver import (
    CandidateCase,
    DEFAULT_INCREMENTS,
    DEFAULT_MESH_SIZE,
    build_candidate_case,
    deterministic_realization,
)


SCHEMA = "agentfem.multi-void-rve-same-rank-restart.v1"
ABSOLUTE_TOLERANCE = 2.0e-11
RELATIVE_TOLERANCE = 2.0e-10


def _capture(case: CandidateCase) -> dict[str, object]:
    step = case.step
    transaction = step.state_transaction
    snapshots = transaction.response.state.snapshot()
    frame = results.homogenize_periodic_path(
        step.snapshots,
        transaction.material,
        constraint=case.periodicity,
    )[-1]
    derived = transaction.snapshot_runtime_state()
    return {
        "solution": step.solution.x.array.copy(),
        "accepted_solution": transaction.accepted_solution.x.array.copy(),
        "state": {name: np.asarray(value).copy() for name, value in snapshots.items()},
        "state_vectors": transaction.response.state.committed_state_vectors().copy(),
        "first_piola": transaction.response.first_piola_stress.values.copy(),
        "deformation_gradient": transaction.deformation_gradient.values.copy(),
        "cauchy_stress": transaction.response.cauchy_stress.values.copy(),
        "equivalent_stress": transaction.equivalent_stress.values.copy(),
        "strain_energy_density": (
            transaction.response.strain_energy_density.values.copy()
        ),
        "tangent": transaction.response.tangent.values.copy(),
        "trial_state": {
            name: np.asarray(value).copy()
            for name, value in derived["trial_state"].items()
        },
        "frame": frame.as_dict(),
        "measured_macro_deformation_gradient": (
            case.periodicity.measured_deformation_gradient(step.solution)
        ),
        "accepted_history": [
            item.as_dict() for item in step.accepted_increments
        ],
        "attempted_history": [
            item.as_dict() for item in step.attempted_increments
        ],
        "accepted_load_factor": float(step.accepted_load_factor),
        "transaction_accepted_factor": float(transaction.accepted_factor),
        "mesh_identity": checkpointing.mesh_portable_identity(
            case.fixture.domain
        ),
        "constraint_identity": case.periodicity.scientific_identity(),
    }


def _array_error(reference, candidate) -> dict[str, object]:
    first = np.asarray(reference, dtype=float)
    second = np.asarray(candidate, dtype=float)
    if first.shape != second.shape:
        return {
            "passed": False,
            "shape_reference": first.shape,
            "shape_candidate": second.shape,
            "maximum_absolute": None,
            "maximum_relative": None,
        }
    difference = np.abs(first - second)
    maximum_absolute = float(np.max(difference, initial=0.0))
    scale = np.maximum(np.maximum(np.abs(first), np.abs(second)), 1.0)
    maximum_relative = float(np.max(difference / scale, initial=0.0))
    return {
        "passed": bool(
            np.allclose(
                first,
                second,
                rtol=RELATIVE_TOLERANCE,
                atol=ABSOLUTE_TOLERANCE,
            )
        ),
        "shape_reference": first.shape,
        "shape_candidate": second.shape,
        "maximum_absolute": maximum_absolute,
        "maximum_relative": maximum_relative,
    }


def _compare_mapping(reference, candidate, *, prefix: str) -> dict[str, object]:
    if set(reference) != set(candidate):
        return {
            f"{prefix}.keys": {
                "passed": False,
                "reference": tuple(sorted(reference)),
                "candidate": tuple(sorted(candidate)),
            }
        }
    checks = {}
    for name in sorted(reference):
        first = reference[name]
        second = candidate[name]
        key = f"{prefix}.{name}"
        if isinstance(first, dict) and isinstance(second, dict):
            checks.update(_compare_mapping(first, second, prefix=key))
        elif (
            first is None
            or second is None
            or isinstance(first, (str, bool, np.bool_))
            or isinstance(second, (str, bool, np.bool_))
        ):
            checks[key] = {
                "passed": first == second,
                "reference": first,
                "candidate": second,
            }
        else:
            checks[key] = _array_error(first, second)
    return checks


def _compare_frame(reference, candidate) -> dict[str, object]:
    checks = {}
    for name in sorted(reference):
        first = reference[name]
        second = candidate[name]
        if first is None or second is None:
            checks[f"macro_frame.{name}"] = {
                "passed": first is None and second is None,
                "reference": first,
                "candidate": second,
            }
        else:
            checks[f"macro_frame.{name}"] = _array_error(first, second)
    return checks


def _compare_history(reference, candidate, *, prefix: str) -> dict[str, object]:
    checks = {
        f"{prefix}.length": {
            "passed": len(reference) == len(candidate),
            "reference": len(reference),
            "candidate": len(candidate),
        }
    }
    if len(reference) != len(candidate):
        return checks
    exact_names = {
        "increment",
        "attempt",
        "converged",
        "iterations",
        "reduced_dofs",
        "message",
    }
    for index, (first, second) in enumerate(zip(reference, candidate, strict=True)):
        if set(first) != set(second):
            checks[f"{prefix}.{index}.keys"] = {
                "passed": False,
                "reference": tuple(sorted(first)),
                "candidate": tuple(sorted(second)),
            }
            continue
        for name in sorted(first):
            key = f"{prefix}.{index}.{name}"
            if name in exact_names or isinstance(first[name], str):
                checks[key] = {
                    "passed": first[name] == second[name],
                    "reference": first[name],
                    "candidate": second[name],
                }
            elif name == "checks":
                checks.update(
                    _compare_mapping(first[name], second[name], prefix=key)
                )
            else:
                checks[key] = _array_error(first[name], second[name])
    return checks


def compare(reference, restarted) -> dict[str, object]:
    """Return a complete, JSON-safe same-rank equivalence assessment."""

    checks = {
        "solution": _array_error(reference["solution"], restarted["solution"]),
        "accepted_solution": _array_error(
            reference["accepted_solution"], restarted["accepted_solution"]
        ),
        "state_vectors": _array_error(
            reference["state_vectors"], restarted["state_vectors"]
        ),
        "first_piola": _array_error(
            reference["first_piola"], restarted["first_piola"]
        ),
        "deformation_gradient": _array_error(
            reference["deformation_gradient"],
            restarted["deformation_gradient"],
        ),
        "cauchy_stress": _array_error(
            reference["cauchy_stress"], restarted["cauchy_stress"]
        ),
        "equivalent_stress": _array_error(
            reference["equivalent_stress"], restarted["equivalent_stress"]
        ),
        "strain_energy_density": _array_error(
            reference["strain_energy_density"],
            restarted["strain_energy_density"],
        ),
        "tangent": _array_error(reference["tangent"], restarted["tangent"]),
        "measured_macro_deformation_gradient": _array_error(
            reference["measured_macro_deformation_gradient"],
            restarted["measured_macro_deformation_gradient"],
        ),
        "accepted_load_factor": _array_error(
            reference["accepted_load_factor"], restarted["accepted_load_factor"]
        ),
        "transaction_accepted_factor": _array_error(
            reference["transaction_accepted_factor"],
            restarted["transaction_accepted_factor"],
        ),
        "mesh_identity": {
            "passed": reference["mesh_identity"] == restarted["mesh_identity"],
        },
        "constraint_identity": {
            "passed": (
                reference["constraint_identity"]
                == restarted["constraint_identity"]
            ),
        },
    }
    checks.update(_compare_mapping(reference["state"], restarted["state"], prefix="state"))
    checks.update(
        _compare_mapping(
            reference["trial_state"],
            restarted["trial_state"],
            prefix="trial_state",
        )
    )
    checks.update(_compare_frame(reference["frame"], restarted["frame"]))
    checks.update(
        _compare_history(
            reference["accepted_history"],
            restarted["accepted_history"],
            prefix="accepted_history",
        )
    )
    checks.update(
        _compare_history(
            reference["attempted_history"],
            restarted["attempted_history"],
            prefix="attempted_history",
        )
    )
    failed = tuple(name for name, value in checks.items() if not value["passed"])
    return {
        "schema": SCHEMA,
        "passed": not failed,
        "failed_checks": failed,
        "tolerances": {
            "absolute": ABSOLUTE_TOLERANCE,
            "relative": RELATIVE_TOLERANCE,
            "scope": "same_rank_same_scientific_input",
        },
        "checks": checks,
    }


def run(
    checkpoint_root: Path,
    *,
    comm=MPI.COMM_WORLD,
    mesh_size: float = DEFAULT_MESH_SIZE,
    increments: int = DEFAULT_INCREMENTS,
    progress: bool = False,
) -> dict[str, object]:
    """Execute uninterrupted and midpoint-restarted branches."""

    selected_increments = int(increments)
    if selected_increments < 2 or selected_increments % 2:
        raise ValueError("Restart evidence requires a positive even increment count.")
    midpoint = 0.5

    uninterrupted = build_candidate_case(
        comm=comm,
        mesh_size=mesh_size,
        increments=selected_increments,
        progress=progress,
        name="finite_strain_j2_deterministic_multi_void_restart_reference",
    )
    uninterrupted.step.solve()
    uninterrupted_state = _capture(uninterrupted)

    partial = build_candidate_case(
        comm=comm,
        mesh_size=mesh_size,
        increments=selected_increments,
        progress=progress,
        name="finite_strain_j2_deterministic_multi_void_restart_partial",
    )
    partial.step.solve(until=midpoint)
    selected_root = Path(checkpoint_root)
    if comm.rank == 0:
        selected_root.parent.mkdir(parents=True, exist_ok=True)
    comm.barrier()
    manifest = partial.step.save_checkpoint(selected_root)

    restarted = build_candidate_case(
        comm=comm,
        mesh_size=mesh_size,
        increments=selected_increments,
        progress=progress,
        name="finite_strain_j2_deterministic_multi_void_restart_resumed",
    )
    restarted.step.load_checkpoint(manifest)
    restored_coordinate = float(restarted.step.accepted_load_factor)
    restored_history = [
        item.as_dict() for item in restarted.step.accepted_increments
    ]
    restarted.step.solve()
    restarted_state = _capture(restarted)
    assessment = compare(uninterrupted_state, restarted_state)
    return {
        **assessment,
        "status": "accepted" if assessment["passed"] else "failed",
        "execution": {
            "mpi_ranks": int(comm.size),
            "mesh_size": float(mesh_size),
            "increments": selected_increments,
            "midpoint": midpoint,
            "checkpoint_manifest": str(manifest),
            "restored_coordinate": restored_coordinate,
            "restored_history": restored_history,
            "uninterrupted_history": uninterrupted_state["accepted_history"],
            "restarted_history": restarted_state["accepted_history"],
        },
        "identity": {
            "realization": deterministic_realization().scientific_identity(),
            "mesh": uninterrupted_state["mesh_identity"],
            "constraint": uninterrupted_state["constraint_identity"],
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--mesh-size", type=float, default=DEFAULT_MESH_SIZE)
    parser.add_argument("--increments", type=int, default=DEFAULT_INCREMENTS)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    options = _parse_args()
    evidence = run(
        options.checkpoint_root,
        mesh_size=options.mesh_size,
        increments=options.increments,
        progress=options.progress,
    )
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        payload = json.dumps(evidence, indent=2, sort_keys=True, default=str)
        if options.output is not None:
            options.output.parent.mkdir(parents=True, exist_ok=True)
            options.output.write_text(payload + "\n", encoding="utf-8")
        if not options.quiet:
            print(payload)
    if not evidence["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
