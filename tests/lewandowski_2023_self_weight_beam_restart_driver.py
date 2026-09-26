"""Checkpoint/restart evidence for the Lewandowski finite-strain J2 beam.

This opt-in driver solves the same public beam twice on the same MPI layout:
once without interruption and once through a midpoint portable checkpoint.  It
compares the complete load-displacement curve, primary solution, committed
constitutive state and provider-owned quadrature response.  The JSON result is
a content-bound promotion input; running the driver is not itself a maturity
claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from agentfem import results
from agentfem.provenance import runtime_manifest

from lewandowski_2023_self_weight_beam_driver import _candidate_step
from lewandowski_2023_self_weight_beam_fixture import DEFINITION


SCHEMA = "agentfem.lewandowski-2023-beam-restart-equivalence.v2"
ABSOLUTE_TOLERANCE = 2.0e-11
PRIMARY_NORMALIZED_TOLERANCE = 2.0e-10
RESPONSE_NORMALIZED_TOLERANCE = 1.0e-8


def _capture(step) -> dict[str, object]:
    transaction = step.state_transaction
    return {
        "solution": step.solution.x.array.copy(),
        "accepted_solution": transaction.accepted_solution.x.array.copy(),
        "state": {
            name: np.asarray(value).copy()
            for name, value in transaction.response.state.snapshot().items()
        },
        "first_piola": transaction.response.first_piola_stress.values.copy(),
        "deformation_gradient": transaction.deformation_gradient.values.copy(),
        "cauchy_stress": transaction.response.cauchy_stress.values.copy(),
        "equivalent_stress": transaction.equivalent_stress.values.copy(),
        "strain_energy_density": (
            transaction.response.strain_energy_density.values.copy()
        ),
        "tangent": transaction.response.tangent.values.copy(),
        "accepted_load_factor": float(step.accepted_load_factor),
    }


def _global_array_error(
    reference,
    candidate,
    *,
    comm,
    normalized_tolerance: float = PRIMARY_NORMALIZED_TOLERANCE,
) -> dict[str, object]:
    """Compare one distributed channel using its global physical scale.

    A single elementwise relative tolerance is not meaningful across the
    dimensionless primary solution, Pa-valued stresses, energy densities and
    Pa-valued fourth-order tangent.  The restart contract therefore retains a
    strict absolute floor and normalizes the largest error by the global
    magnitude of the channel being compared.
    """

    first = np.asarray(reference, dtype=float)
    second = np.asarray(candidate, dtype=float)
    local_shape_match = first.shape == second.shape
    shape_match = bool(comm.allreduce(local_shape_match, op=MPI.LAND))
    if not shape_match:
        return {
            "passed": False,
            "maximum_absolute": None,
            "maximum_relative": None,
            "reason": "rank_local_shape_mismatch",
        }
    difference = np.abs(first - second)
    local_absolute = float(np.max(difference, initial=0.0))
    local_scale = float(
        max(
            np.max(np.abs(first), initial=0.0),
            np.max(np.abs(second), initial=0.0),
        )
    )
    maximum_absolute = float(comm.allreduce(local_absolute, op=MPI.MAX))
    reference_scale = float(comm.allreduce(local_scale, op=MPI.MAX))
    maximum_normalized = (
        maximum_absolute / reference_scale
        if reference_scale > np.finfo(float).tiny
        else maximum_absolute
    )
    passed = bool(
        maximum_absolute
        <= ABSOLUTE_TOLERANCE + normalized_tolerance * reference_scale
    )
    return {
        "passed": passed,
        "maximum_absolute": maximum_absolute,
        "maximum_normalized": maximum_normalized,
        "reference_scale": reference_scale,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "normalized_tolerance": normalized_tolerance,
    }


def _solve_curve(step, displacement, factors, *, prefix=None):
    accepted = [0.0] if prefix is None else list(prefix[0])
    downward = [0.0] if prefix is None else list(prefix[1])
    for factor in factors:
        step.solve(until=float(factor))
        value = results.probe(displacement, at=DEFINITION.observer)
        accepted.append(float(factor))
        downward.append(-float(value[2]))
    return accepted, downward


def run(
    checkpoint_root: Path,
    *,
    comm=MPI.COMM_WORLD,
    subdivisions=DEFINITION.subdivisions,
    increments: int = DEFINITION.increments,
    progress: bool = False,
) -> dict[str, object]:
    """Execute uninterrupted and midpoint-restarted branches."""

    selected_increments = int(increments)
    if selected_increments < 6 or selected_increments % 2:
        raise ValueError("Restart evidence requires an even increment count of at least 6.")
    factors = np.linspace(0.0, 1.0, selected_increments + 1)
    midpoint_index = selected_increments // 2
    midpoint = float(factors[midpoint_index])

    uninterrupted, uninterrupted_u = _candidate_step(
        comm,
        subdivisions=tuple(subdivisions),
        increments=selected_increments,
        progress=progress,
    )
    reference_load, reference_u = _solve_curve(
        uninterrupted,
        uninterrupted_u,
        factors[1:],
    )
    reference_state = _capture(uninterrupted)

    partial, partial_u = _candidate_step(
        comm,
        subdivisions=tuple(subdivisions),
        increments=selected_increments,
        progress=progress,
    )
    restarted_load, restarted_u = _solve_curve(
        partial,
        partial_u,
        factors[1 : midpoint_index + 1],
    )
    selected_root = Path(checkpoint_root)
    if comm.rank == 0:
        selected_root.parent.mkdir(parents=True, exist_ok=True)
    comm.barrier()
    checkpoint = partial.save_checkpoint(selected_root)

    restarted, restarted_field = _candidate_step(
        comm,
        subdivisions=tuple(subdivisions),
        increments=selected_increments,
        progress=progress,
    )
    restarted.load_checkpoint(checkpoint)
    restored_factor = float(restarted.accepted_load_factor)
    restarted_load, restarted_u = _solve_curve(
        restarted,
        restarted_field,
        factors[midpoint_index + 1 :],
        prefix=(restarted_load, restarted_u),
    )
    restarted_state = _capture(restarted)

    checks = {
        "load_path": _global_array_error(reference_load, restarted_load, comm=comm),
        "displacement_curve": _global_array_error(reference_u, restarted_u, comm=comm),
        "solution": _global_array_error(
            reference_state["solution"], restarted_state["solution"], comm=comm
        ),
        "accepted_solution": _global_array_error(
            reference_state["accepted_solution"],
            restarted_state["accepted_solution"],
            comm=comm,
        ),
        "first_piola": _global_array_error(
            reference_state["first_piola"],
            restarted_state["first_piola"],
            comm=comm,
            normalized_tolerance=RESPONSE_NORMALIZED_TOLERANCE,
        ),
        "deformation_gradient": _global_array_error(
            reference_state["deformation_gradient"],
            restarted_state["deformation_gradient"],
            comm=comm,
        ),
        "cauchy_stress": _global_array_error(
            reference_state["cauchy_stress"],
            restarted_state["cauchy_stress"],
            comm=comm,
            normalized_tolerance=RESPONSE_NORMALIZED_TOLERANCE,
        ),
        "equivalent_stress": _global_array_error(
            reference_state["equivalent_stress"],
            restarted_state["equivalent_stress"],
            comm=comm,
            normalized_tolerance=RESPONSE_NORMALIZED_TOLERANCE,
        ),
        "strain_energy_density": _global_array_error(
            reference_state["strain_energy_density"],
            restarted_state["strain_energy_density"],
            comm=comm,
            normalized_tolerance=RESPONSE_NORMALIZED_TOLERANCE,
        ),
        "tangent": _global_array_error(
            reference_state["tangent"],
            restarted_state["tangent"],
            comm=comm,
            normalized_tolerance=RESPONSE_NORMALIZED_TOLERANCE,
        ),
        "accepted_load_factor": _global_array_error(
            reference_state["accepted_load_factor"],
            restarted_state["accepted_load_factor"],
            comm=comm,
        ),
    }
    reference_states = reference_state["state"]
    restarted_states = restarted_state["state"]
    state_names_match = set(reference_states) == set(restarted_states)
    checks["state_schema"] = {"passed": state_names_match}
    if state_names_match:
        for name in sorted(reference_states):
            checks[f"state.{name}"] = _global_array_error(
                reference_states[name],
                restarted_states[name],
                comm=comm,
                normalized_tolerance=(
                    RESPONSE_NORMALIZED_TOLERANCE
                    if name == "plastic_dissipation"
                    else PRIMARY_NORMALIZED_TOLERANCE
                ),
            )
    restored_midpoint = abs(restored_factor - midpoint) <= 1.0e-14
    checks["restored_midpoint"] = {
        "passed": restored_midpoint,
        "expected": midpoint,
        "actual": restored_factor,
    }
    failed = tuple(name for name, check in checks.items() if not check["passed"])
    runtime = runtime_manifest()
    checkpoint_manifest = Path(checkpoint)
    checkpoint_digest = None
    if comm.rank == 0:
        checkpoint_digest = hashlib.sha256(checkpoint_manifest.read_bytes()).hexdigest()
    checkpoint_digest = comm.bcast(checkpoint_digest, root=0)
    return {
        "schema": SCHEMA,
        "status": "accepted" if not failed else "failed",
        "passed": not failed,
        "failed_checks": failed,
        "tolerances": {
            "absolute": ABSOLUTE_TOLERANCE,
            "primary_normalized": PRIMARY_NORMALIZED_TOLERANCE,
            "response_normalized": RESPONSE_NORMALIZED_TOLERANCE,
            "normalization": "global_maximum_magnitude_per_physical_channel",
            "scope": "same_rank_same_mesh_complete_state",
        },
        "execution": {
            "mpi_ranks": int(comm.size),
            "subdivisions": tuple(int(value) for value in subdivisions),
            "increments": selected_increments,
            "midpoint": midpoint,
            "restored_factor": restored_factor,
            "checkpoint_manifest": str(checkpoint_manifest),
            "checkpoint_manifest_sha256": checkpoint_digest,
        },
        "runtime": runtime,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--subdivisions",
        type=int,
        nargs=3,
        default=DEFINITION.subdivisions,
    )
    parser.add_argument("--increments", type=int, default=DEFINITION.increments)
    parser.add_argument("--progress", action="store_true")
    arguments = parser.parse_args()
    report = run(
        arguments.checkpoint_root,
        subdivisions=tuple(arguments.subdivisions),
        increments=arguments.increments,
        progress=arguments.progress,
    )
    if MPI.COMM_WORLD.rank == 0:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(arguments.output)
        print(
            "Lewandowski beam restart equivalence "
            f"| {report['status'].upper()} | evidence={arguments.output}"
        )


if __name__ == "__main__":
    main()
