# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare every retained channel of serial/MPI validation runs."""

from pathlib import Path
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("serial", type=Path)
parser.add_argument("parallel", type=Path)
parser.add_argument("--output", type=Path, default=Path(__file__).parent / "evidence")
args = parser.parse_args()
a = json.loads((args.serial / "records.json").read_text())
b = json.loads((args.parallel / "records.json").read_text())
keys = sorted(a["history"][0])
assert keys == sorted(b["history"][0])
x = np.array([[r[k] for k in keys] for r in a["history"]])
y = np.array([[r[k] for k in keys] for r in b["history"]])
np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-14)
report = {
    "max_absolute_difference": float(np.max(abs(x - y))),
    "serial_restart_max_abs": a["restart_max_abs"],
    "mpi2_restart_max_abs": b["restart_max_abs"],
    "columns": keys,
}
args.output.mkdir(parents=True, exist_ok=True)
for filename, payload in [
    ("serial_records.json", a),
    ("mpi2_records.json", b),
    ("mpi_comparison.json", report),
]:
    (args.output / filename).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(report, indent=2))
