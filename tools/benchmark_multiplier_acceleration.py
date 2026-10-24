# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Repeat the public cyclic plane example with unchanged AL acceptance gates.

Run with PYTHONPATH=src from the checkout. Serial, one thread; no industrial
contact/damage speed claim. Setup and solve are timed separately. Warm both
policies, alternate measured run order, compare every accepted displacement
snapshot plus final material/pressure and all available energy-history values.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import runpy
import subprocess
from time import perf_counter

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.cells < 1 or args.repeats < 2 or args.output.exists():
        parser.error("Use positive cells, at least two repeats and a fresh output path.")
    root = Path(__file__).resolve().parents[1]
    example = root / "examples/finite_j2_translating_augmented_plane.py"
    build = runpy.run_path(str(example))["make_step"]
    records, reference, reference_layout = [], None, None
    for repeat in range(-1, args.repeats):
        modes = ("none", "aitken") if repeat < 0 or repeat % 2 == 0 else ("aitken", "none")
        for mode in modes:
            started = perf_counter()
            _, step = build(cells=args.cells, increments=32, acceleration=mode)
            setup = perf_counter() - started
            started = perf_counter()
            step.solve()
            elapsed = perf_counter() - started
            energy_rows, energy_layout = [], []
            for frame in step.energy_history:
                values = frame.as_dict()
                pairs = values.pop("pair_tool_work")
                values.update({f"tool:{key}": value for key, value in pairs.items()})
                energy_layout.append(tuple((key, value is not None) for key, value in values.items()))
                energy_rows.append([value for value in values.values() if value is not None])
            if reference_layout is None:
                reference_layout = energy_layout
            if energy_layout != reference_layout:
                raise ValueError("Accepted energy channels or their availability changed.")
            physical = {
                "displacement_path": np.concatenate([s.solution.x.array for s in step.snapshots]),
                "material_state": step.response.state.committed_state_vectors().copy(),
                "pressure": step.contact_provider.accepted_point_samples()["normal_pressures"],
                "energy_path": np.asarray(energy_rows, dtype=float),
            }
            if reference is None:
                reference = {k: v.copy() for k, v in physical.items()}
            errors = {k: float(np.max(np.abs(v - reference[k]), initial=0.))
                      for k, v in physical.items()}
            tolerances = dict(displacement_path=3e-8, material_state=1e-7,
                              pressure=2e-5, energy_path=3e-7)
            for key in physical:
                np.testing.assert_allclose(physical[key], reference[key], atol=tolerances[key], rtol=0.)
            row = dict(
                mode=mode, repeat=repeat, warmup=repeat < 0, cells=args.cells, increments=32,
                setup_seconds=setup, solve_seconds=elapsed,
                augmentations=sum(i.normal_augmentations for i in step.accepted_increments),
                newton_iterations=sum(i.iterations for i in step.accepted_increments),
                linear_solves=sum(i.linear_solve_calls for i in step.accepted_increments),
                acceleration_resets=sum(i.normal_acceleration_resets for i in step.accepted_increments),
                errors=errors,
            )
            records.append(row)
            print(json.dumps(row), flush=True)
    medians = {mode: float(np.median([r["solve_seconds"] for r in records
                                    if r["mode"] == mode and not r["warmup"]]))
               for mode in ("none", "aitken")}
    sources = [Path(__file__), example,
               root / "src/agentfem/mechanics/_normal_multiplier_acceleration.py",
               root / "src/agentfem/mechanics/_normal_augmentation_procedure.py",
               root / "src/agentfem/boundary_models/normal_augmentation.py",
               root / "src/agentfem/boundary_models/_augmented_plane_trial_provider.py"]
    report = dict(
        schema="agentfem.multiplier-acceleration-screen.v1",
        scope="serial_cyclic_translating_plane_finite_J2_not_general_contact_damage_performance",
        platform=platform.platform(), records=records, median_solve_seconds=medians,
        solve_time_reduction=1.-medians["aitken"]/medians["none"],
        reference="same_model_unaccelerated_policy; independent_reference_checks_in_tests",
        comparison_tolerances=tolerances,
        git_base=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sources},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
