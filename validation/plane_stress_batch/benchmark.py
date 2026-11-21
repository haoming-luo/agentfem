# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded complete-Step timing using the independently tested material fixture.

Run from the checkout root with PYTHONPATH=src in fenicsx-env. Test fixtures
are intentionally outside the installed runtime; this is a developer benchmark.
"""
import sys
import time
import json
import platform
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_plane_stress_array import ArrayNeoHookean
from test_prescribed_material_fields import build
from agentfem import __version__


def main():
    records, final = [], {}
    for repeat, modes in enumerate(((False, True), (True, False), (False, True))):
        for mode in modes:
            start = time.perf_counter()
            step, _ = build(material=ArrayNeoHookean(array=mode), cells=(8, 8), increments=12)
            prepared = time.perf_counter()
            step.solve()
            ended = time.perf_counter()
            snapshot = step.accepted_fields
            final[mode] = {key: snapshot[key].copy() for key in ("F", "P", "S", "AGE", "SENER", "MATERIAL_DPDF")}
            record = dict(repeat=repeat, mode="array" if mode else "scalar", build_seconds=prepared-start,
                solve_seconds=ended-prepared, build_and_solve_seconds=ended-start,
                accepted=len(step.accepted_increments), attempts=len(step.attempted_increments),
                quadrature_points=len(snapshot.weights))
            print(json.dumps(record), flush=True)
            records.append(record)
    differences = {key: float(np.max(np.abs(final[True][key]-final[False][key]))) for key in final[True]}
    for key in final[True]:
        np.testing.assert_allclose(final[True][key], final[False][key], rtol=1e-9, atol=1e-10)
    files = ("src/agentfem/constitutive/plane_stress.py", "src/agentfem/constitutive/material_driver.py",
        "src/agentfem/mechanics/finite_strain_material.py", "tests/test_plane_stress_array.py",
        "tests/test_prescribed_material_fields.py", "validation/plane_stress_batch/benchmark.py")
    record = dict(scope="synthetic 2D plane-stress standard Step; no customer simulation",
        timing="build and completed solve; field output excluded; existing local JIT cache",
        runtime=dict(agentfem=__version__, python=platform.python_version(), platform=platform.platform()),
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        records=records, maximum_absolute_differences=differences)
    destination = Path(__file__).with_name("timing.json")
    destination.write_text(json.dumps(record, indent=2)+"\n")
    print(destination)


if __name__ == "__main__":
    main()
