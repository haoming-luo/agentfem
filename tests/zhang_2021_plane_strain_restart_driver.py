"""Portable restart evidence for the exact Zhang Q2/DPC1 periodic cell.

``write`` records a fully accepted midpoint checkpoint.  ``read`` rebuilds the
same scientific problem, resumes the checkpoint (including with a different
MPI rank count), and compares its partition-independent final evidence with an
uninterrupted solve on the reader communicator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mpi4py import MPI
import numpy as np

from agentfem import models, operators, results, solvers, steps, studies

from zhang_2021_periodic_composite_fixture import (
    zhang_2021_plane_strain_composite,
)


SCHEMA = "agentfem.zhang-2021-plane-strain-restart.v1"
ABSOLUTE_TOLERANCE = 2.0e-10
RELATIVE_TOLERANCE = 2.0e-9


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_artifacts(manifest_path: Path) -> dict[str, object]:
    """Verify and describe the complete portable checkpoint artifact graph."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = {
        "manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "size": manifest_path.stat().st_size,
        }
    }
    for name in ("nodal_state", "quadrature_state"):
        reference = manifest.get(name)
        if not isinstance(reference, dict):
            raise RuntimeError(f"Checkpoint {name!r} reference is missing.")
        selected = manifest_path.parent / str(reference.get("path", ""))
        expected = reference.get("sha256")
        if not selected.is_file() or not isinstance(expected, str):
            raise RuntimeError(f"Checkpoint {name!r} artifact is missing: {selected}")
        actual = _sha256(selected)
        if actual != expected:
            raise RuntimeError(f"Checkpoint {name!r} SHA-256 mismatch: {selected}")
        artifacts[name] = {
            "path": str(selected),
            "sha256": actual,
            "size": selected.stat().st_size,
        }
    return {
        "writer_rank_count": int(manifest["writer_rank_count"]),
        "portable": bool(manifest["portable"]),
        "artifacts": artifacts,
    }


def _fingerprint(record: object) -> str:
    encoded = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _build(comm, *, mesh_size: float, increments: int, progress: bool = False,
           geometry_source: str = "section-3.2.1-text"):
    fixture = zhang_2021_plane_strain_composite(
        comm, mesh_size=mesh_size, geometry_source=geometry_source,
    )
    model = models.create(
        study=studies.nonlinear_static(
            physics="solid_mechanics",
            dimension=2,
            assumption="plane_strain",
        ),
        mesh=fixture.domain,
        name="zhang_2021_plane_strain_restart",
    )
    target = model.field(fixture.mixed_field())
    matrix_region, inclusion_region = fixture.regions()
    matrix, inclusion = fixture.materials()
    model.material(matrix, region=matrix_region)
    model.material(inclusion, region=inclusion_region)
    periodicity = model.constraint(fixture.constraint(target))
    output = results.output_plan(
        Path("/private/tmp/agentfem-zhang-restart-unwritten"),
        requests=(results.periodic_cell_history(periodicity),),
        basename="zhang_2021_restart",
    )
    problem = model.step(
        target=target,
        constraints=periodicity,
        quadrature_degree=4,
        incrementation=steps.fixed(increments),
        solver_options=solvers.newton(
            relative_tolerance=1.0e-8,
            absolute_tolerance=1.0e-10,
            maximum_iterations=35,
            line_search="backtracking",
        ),
        output=output,
        progress=progress,
        name="zhang_2021_plane_strain_restart",
    )
    return fixture, target, periodicity, problem


def _capture(fixture, target, periodicity, problem) -> dict[str, object]:
    comm = fixture.domain.comm

    def synchronize(stage: str) -> None:
        comm.barrier()
        if comm.rank == 0:
            print(f"[restart evidence] {stage}", flush=True)

    recorder = problem.accepted_history_recorders["homogenized_history"]
    frame = recorder.frames[-1]
    last = problem.last_solve_info.increments[-1]
    synchronize("periodic mismatch")
    mismatch = float(periodicity.mismatch())
    synchronize("measured macro gradient")
    measured = np.asarray(
        periodicity.measured_deformation_gradient(target.displacement),
        dtype=float,
    )
    synchronize("mesh identity")
    mesh_identity = operators.mesh_executable_identity(fixture.domain)
    synchronize("capture complete")
    return {
        "geometry_source": fixture.geometry_source,
        "accepted_load_factor": float(problem.accepted_load_factor),
        "accepted_increment_factors": [
            float(item.load_factor) for item in problem.accepted_increments
        ],
        "attempted_increment_count": len(problem.attempted_increments),
        "first_piola": np.asarray(frame.first_piola_stress, dtype=float),
        "cauchy_stress": np.asarray(frame.cauchy_stress, dtype=float),
        "deformation_gradient": np.asarray(frame.deformation_gradient, dtype=float),
        "elastic_energy_density": float(frame.elastic_energy_density),
        "plastic_dissipation_density": float(frame.plastic_dissipation_density),
        "hardening_energy_density": float(frame.hardening_energy_density),
        "maximum_hill_mandel_relative_error": max(
            float(item.relative_error) for item in recorder.hill_mandel
        ),
        "periodic_equation_mismatch": mismatch,
        "measured_deformation_gradient": measured,
        "pressure_block_residual_norm": float(
            last.checks["pressure_block_residual_norm"]
        ),
        "maximum_quadrature_pressure_constraint_defect": float(
            last.checks["maximum_quadrature_pressure_projection_defect"]
        ),
        "global_cells": int(fixture.domain.topology.index_map(2).size_global),
        # The public executable mesh identity is one partition-independent
        # collective.  Do not re-enter the private checkpoint identity graph
        # here: checkpoint loading already validated that complete graph, and
        # calling it again from postprocessing duplicates several collectives.
        "mesh_identity": mesh_identity,
        "constraint_identity": periodicity.scientific_identity(),
    }


def _numeric_check(reference, candidate) -> dict[str, object]:
    first = np.asarray(reference, dtype=float)
    second = np.asarray(candidate, dtype=float)
    maximum_absolute = float(np.max(np.abs(first - second), initial=0.0))
    scale = np.maximum(np.maximum(np.abs(first), np.abs(second)), 1.0)
    maximum_relative = float(np.max(np.abs(first - second) / scale, initial=0.0))
    return {
        "passed": bool(
            first.shape == second.shape
            and np.allclose(
                first,
                second,
                rtol=RELATIVE_TOLERANCE,
                atol=ABSOLUTE_TOLERANCE,
            )
        ),
        "shape": first.shape,
        "maximum_absolute": maximum_absolute,
        "maximum_relative": maximum_relative,
    }


def _compare(reference, candidate) -> dict[str, object]:
    exact = (
        "geometry_source",
        "accepted_increment_factors",
        "attempted_increment_count",
        "global_cells",
        "mesh_identity",
        "constraint_identity",
    )
    checks = {
        name: {
            "passed": reference[name] == candidate[name],
            "reference": reference[name],
            "candidate": candidate[name],
        }
        for name in exact
    }
    for name in sorted(set(reference) - set(exact)):
        checks[name] = _numeric_check(reference[name], candidate[name])
    failed = tuple(name for name, check in checks.items() if not check["passed"])
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "tolerances": {
            "absolute": ABSOLUTE_TOLERANCE,
            "relative": RELATIVE_TOLERANCE,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "read"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mesh-size", type=float, default=0.30)
    parser.add_argument("--geometry-source", choices=("section-3.2.1-text", "figure-10a"),
                        default="section-3.2.1-text")
    parser.add_argument("--increments", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", action="store_true")
    arguments = parser.parse_args()
    if arguments.increments < 2 or arguments.increments % 2:
        parser.error("--increments must be a positive even integer")

    comm = MPI.COMM_WORLD
    if arguments.action == "write":
        _fixture, _target, _periodicity, problem = _build(
            comm,
            mesh_size=arguments.mesh_size,
            increments=arguments.increments,
            progress=arguments.progress,
            geometry_source=arguments.geometry_source,
        )
        problem.solve(until=0.5)
        manifest = problem.save_checkpoint(arguments.checkpoint)
        if comm.rank == 0:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload["writer_rank_count"] != comm.size or not payload["portable"]:
                raise RuntimeError("Checkpoint portability provenance is invalid.")
        return

    reference_case = _build(
        comm,
        mesh_size=arguments.mesh_size,
        increments=arguments.increments,
        progress=arguments.progress,
        geometry_source=arguments.geometry_source,
    )
    reference_case[-1].solve()
    reference = _capture(*reference_case)

    restarted_case = _build(
        comm,
        mesh_size=arguments.mesh_size,
        increments=arguments.increments,
        progress=arguments.progress,
        geometry_source=arguments.geometry_source,
    )
    restarted = restarted_case[-1]
    checkpoint = _checkpoint_artifacts(arguments.checkpoint)
    restarted.load_checkpoint(arguments.checkpoint)
    restored_coordinate = float(restarted.accepted_load_factor)
    restarted.solve()
    candidate = _capture(*restarted_case)
    comparison = _compare(reference, candidate)
    evidence = {
        "schema": SCHEMA,
        "status": "passed" if comparison["passed"] else "failed",
        "passed": comparison["passed"],
        "restart_equivalent": comparison["passed"],
        "content_bound": True,
        "benchmark_promotion_authorized": False,
        "geometry_source": arguments.geometry_source,
        "writer_rank_count": checkpoint["writer_rank_count"],
        "reader_rank_count": int(comm.size),
        "restored_coordinate": restored_coordinate,
        "checkpoint": checkpoint,
        "comparison": comparison,
        "decision_scope": (
            "portable Q9/DPC1 mixed-state checkpoint resumed across a changed "
            "MPI rank count and compared with an uninterrupted solve on the "
            "reader communicator; no Table 5 promotion authority"
        ),
    }
    evidence["fingerprint"] = _fingerprint(evidence)
    if comm.rank == 0:
        payload = json.dumps(evidence, indent=2, sort_keys=True, default=list)
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
    if not comparison["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
