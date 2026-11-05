# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded output-cadence comparison; not a constitutive-kernel speed claim."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

import numpy as np

import agentfem
from agentfem import amplitudes, constitutive, fields, io, mesh, models, steps, studies


def instrument(function, profile, label):
    def measured(*args, **kwargs):
        start = perf_counter()
        result = function(*args, **kwargs)
        profile[f"{label}_seconds"] += perf_counter() - start
        profile[f"{label}_calls"] += 1
        return result

    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--increments", type=int, default=16)
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--no-profile", action="store_true", help="Confirm total times without per-call instrumentation.")
    parser.add_argument("--compare-preparation", action="store_true", help="Also force fresh geometry-expression preparation as an equivalent baseline.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.size, args.increments, args.every, args.pairs) < 1 or args.output.exists():
        parser.error("Use positive counts and a fresh report path.")
    example = Path(__file__).resolve().parents[1] / "examples/history_material/material.py"
    Material = runpy.run_path(str(example))["HistoryMaterial"]
    measure = constitutive.QuadratureField.owned_physical_weights

    def rebuilding_measure(field, *a, **kw):
        field._measure_expression_cache = None
        return measure(field, *a, **kw)

    def build(mechanism, every):
        domain = mesh.rectangle((0., 0.), (1., 1.), (args.size, args.size))
        model = models.create(study=studies.static_solid(dimension=2, assumption="plane_strain", nonlinear=True), mesh=domain)
        u = model.field(fields.displacement(domain))
        model.material(Material(mechanism=mechanism))
        model.clamp(u, on=mesh.face(domain, axis="x", value=0))
        model.fix(u, on=mesh.face(domain, axis="x", value=1), component=0, value=0.12)
        return model.step(
            target=u, duration=2., time_unit="s", progress=False,
            amplitude=amplitudes.tabular((0., .25, .5, .75, 1.), (0., 1., 1., -.2, 0.)),
            incrementation=steps.fixed(args.increments), output_every=every,
        )

    records = []
    # All generated frame data are owned by this temporary directory and cleaned
    # after numerical comparisons. Only the requested small report is retained.
    with tempfile.TemporaryDirectory(prefix="agentfem-output-cadence-") as temporary:
        root = Path(temporary)
        for mechanism in ("viscoelastic", "damage"):
            # Warm one complete path: material branches, assembly and output.
            warm = build(mechanism, 1)
            warm.solve_result(output=root / f"warm-{mechanism}" / "results.xdmf")
            reference_u = warm.solution.x.array.copy()
            reference_z = warm.state.committed_state_vectors().copy()
            reference_force = warm.reaction_field().x.array.copy()
            reference_accepted = list(warm.accepted_increments)
            for pair in range(args.pairs):
                preparations = ("rebuild", "reuse") if args.compare_preparation else ("reuse",)
                order = [(every, preparation) for every in (1, args.every) for preparation in preparations]
                if pair % 2:
                    order.reverse()
                for position, (every, preparation) in enumerate(order):
                    step = build(mechanism, every)
                    target = root / f"{mechanism}-{pair}-{position}" / "results.xdmf"
                    profile = {"recovery_seconds": 0., "write_seconds": 0., "recovery_calls": 0, "write_calls": 0}
                    timed_recovery = instrument(step._visual_fields, profile, "recovery")
                    timed_write = instrument(io.ResultWriter.write, profile, "write")
                    with ExitStack() as stack:
                        if preparation == "rebuild":
                            stack.enter_context(patch.object(constitutive.QuadratureField, "owned_physical_weights", rebuilding_measure))
                        if not args.no_profile:
                            stack.enter_context(patch.object(step, "_visual_fields", timed_recovery))
                            stack.enter_context(patch.object(io.ResultWriter, "write", timed_write))
                        start = perf_counter()
                        result = step.solve_result(output=target)
                        total = perf_counter() - start
                    if args.no_profile:
                        profile = {key: None for key in profile}
                    np.testing.assert_array_equal(step.solution.x.array, reference_u)
                    np.testing.assert_array_equal(step.state.committed_state_vectors(), reference_z)
                    np.testing.assert_array_equal(step.reaction_field().x.array, reference_force)
                    assert step.accepted_increments == reference_accepted
                    records.append({
                        "mechanism": mechanism, "pair": pair, "output_every": every,
                        "measure_preparation": preparation,
                        "total_seconds": total, **profile,
                        "other_seconds": None if args.no_profile else total - profile["recovery_seconds"] - profile["write_seconds"],
                        "output_bytes": sum(path.stat().st_size for path in target.parent.rglob("*") if path.is_file()),
                        "frames": result.metadata["field_output"]["frame_count"],
                        "accepted_increments": len(step.accepted_increments),
                        "attempts": len(step.attempted_increments),
                        "same_host_exact_state_force_and_ledger": True,
                    })
    report = {
        "scope": "nonuniform educational finite-history solve_result; output policy comparison, not solver acceleration",
        "excluded": ["Step construction", "cold JIT", "checkpoint writes", "manifest publication", "external reader", "composite simulation"],
        "timing": "synchronous recovery and write timers; other_seconds includes equilibrium and remaining result/lifecycle work, not pure solver time",
        "warmup": "one complete dense-output path per material, excluded",
        "instrumented": not args.no_profile,
        "size": args.size, "requested_increments": args.increments,
        "environment": {"platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "agentfem": agentfem.__version__,
                        "thread_environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "benchmark_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "material_sha256": sha256(example.read_bytes()).hexdigest(),
        "implementation_sha256": {name: sha256((Path(agentfem.__file__).parent / name).read_bytes()).hexdigest() for name in
                                  ("mechanics/finite_strain_material.py", "constitutive/quadrature.py", "results/recovery.py", "io.py", "results/_live_xdmf.py")},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
