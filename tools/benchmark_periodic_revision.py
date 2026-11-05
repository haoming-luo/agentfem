# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Same-fixture whole-path regression across two checked-out runtime revisions.

The parent never initializes MPI. Each serial child imports one requested
runtime, warms the complete path, then measures one fresh path. ABBA compares
net revision cost, including any new accepted-state diagnostics. It is distinct
from an isolated optimization benchmark and not an explicit-composite forecast.
"""

import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter


def source_hash(root):
    digest = sha256()
    for path in sorted((root / "src/agentfem").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def worker(root, increments):
    import agentfem
    from mpi4py import MPI
    import numpy as np
    from diagnose_periodic_contact_order import make_step

    if MPI.COMM_WORLD.size != 1 or not Path(agentfem.__file__).resolve().is_relative_to(root / "src"):
        raise RuntimeError("Require serial execution from the selected runtime source.")
    frozen = source_hash(root)
    records = []
    for _ in range(2):
        step = make_step(cells=4, degree=2, increments=increments, contact_degree=11, bulk_degree=6)
        gc.collect()
        stations, arrays = [], []
        started = perf_counter()
        for factor in (.25, .5, .75, 1.):
            step.solve(until=factor)
            contact = step.contact_provider.evaluate()
            arrays.extend((step.solution.x.array.copy(),
                           step.response.state.committed_state_vectors().copy(),
                           step.response.first_piola_stress.values.copy(),
                           step.response.tangent.values.copy()))
            stations.append(dict(factor=factor, force=contact.contact_force_on_structure.tolist(),
                                 potential=contact.potential_energy))
        seconds = perf_counter() - started
        digest = sha256()
        for array in arrays:
            if not np.isfinite(array).all():
                raise AssertionError("Nonfinite accepted field/state/stress/tangent.")
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        path = [(i.load_factor, i.iterations, i.accepted_step_lengths, i.linear_converged_reasons)
                for i in step.accepted_increments]
        records.append(dict(seconds=seconds, accepted=len(path),
                            newton_iterations=sum(i[1] for i in path),
                            physical_arrays_sha256=digest.hexdigest(), stations=stations,
                            path=path, dual_history=step.constraint_dual_history.records,
                            material_energy_history_available="material_energy" in step.accepted_history_recorders))
    if source_hash(root) != frozen:
        raise RuntimeError("Runtime source changed during measurement.")
    for key in ("physical_arrays_sha256", "stations", "path", "dual_history"):
        if records[0][key] != records[1][key]:
            raise AssertionError(f"Warm/measured path differs: {key}")
    return dict(runtime_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        runtime_source_sha256=frozen, warmup_seconds=records[0]["seconds"], **records[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--increments", type=int, default=64)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.increments < 4 or args.increments % 4:
        parser.error("Use increments divisible by four.")
    if args.worker:
        print(json.dumps(worker(args.worker.resolve(), args.increments), allow_nan=False))
        return
    if not args.baseline_source or not args.candidate_source or not args.output or args.output.exists():
        parser.error("Provide two source checkouts and a fresh output path.")
    roots = {"baseline": args.baseline_source.resolve(), "candidate": args.candidate_source.resolve()}
    if any(not (root / "src/agentfem/__init__.py").is_file() for root in roots.values()):
        parser.error("Each runtime must be an AgentFEM source checkout.")
    tool = Path(__file__).resolve()
    fixture = tool.with_name("diagnose_periodic_contact_order.py")
    report = dict(schema="agentfem.periodic-revision-regression.v1", status="running",
                  platform=platform.platform(), cells=4, increments=args.increments,
                  scope="serial_P2_J2_plane_penalty_MPC_net_revision_cost_not_general_speedup",
                  timing_boundary="solve plus four common observations; setup/imports excluded; whole-path warmup per child",
                  comparison="exact physical arrays and common histories on this machine; new diagnostic availability reported separately",
                  tool_sha256=sha256(tool.read_bytes()).hexdigest(),
                  fixture_sha256=sha256(fixture.read_bytes()).hexdigest(),
                  thread_environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "FI_PROVIDER")},
                  records=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        with temporary.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    save()
    try:
        for mode in ("baseline", "candidate", "candidate", "baseline"):
            environment = dict(os.environ, PYTHONPATH=str(roots[mode] / "src"))
            child = subprocess.run([sys.executable, str(tool), "--worker", str(roots[mode]),
                                    "--increments", str(args.increments)], env=environment,
                                   capture_output=True, text=True, timeout=300)
            if child.returncode:
                raise RuntimeError(f"{mode} child failed: {child.stdout[-2000:]}\n{child.stderr[-4000:]}")
            row = dict(mode=mode, **json.loads(child.stdout))
            if report["records"]:
                for key in ("physical_arrays_sha256", "stations", "path", "dual_history"):
                    if row[key] != report["records"][0][key]:
                        report["records"].append(row)
                        raise AssertionError(f"Cross-revision physical result changed: {key}")
            report["records"].append(row)
            save()
            print(json.dumps({k: row[k] for k in ("mode", "seconds", "runtime_commit", "accepted", "newton_iterations", "material_energy_history_available")}), flush=True)
        for mode in roots:
            hashes = {r["runtime_source_sha256"] for r in report["records"] if r["mode"] == mode}
            if len(hashes) != 1 or source_hash(roots[mode]) not in hashes:
                raise RuntimeError(f"{mode} runtime changed across comparisons.")
        if (sha256(tool.read_bytes()).hexdigest() != report["tool_sha256"]
                or sha256(fixture.read_bytes()).hexdigest() != report["fixture_sha256"]):
            raise RuntimeError("Common benchmark or fixture changed during comparison.")
        means = {mode: sum(r["seconds"] for r in report["records"] if r["mode"] == mode)/2 for mode in roots}
        report.update(status="completed", mean_seconds=means,
                      elapsed_reduction_fraction=1-means["candidate"]/means["baseline"])
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save()
        raise
    save()


if __name__ == "__main__":
    main()
