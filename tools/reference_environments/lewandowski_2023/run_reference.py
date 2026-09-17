"""Execute the pinned upstream script and archive its numerical curve."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import subprocess

import numpy as np


EXPECTED = {
    "reference.py": "b2fdebe7edf9e10617f9cd4f350bfab119db9ca9441242959e5bf366a105e0d5",
    "LogarithmicStrainPlasticity.mfront": (
        "17456bf1b593eec205f7cc7fb06483fdaf67ec7415713aeb27f33db57bef1393"
    ),
}


def digest(path: Path) -> str:
    """Return the SHA-256 identity of one artifact."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


for filename, expected in EXPECTED.items():
    actual = digest(Path(filename))
    if actual != expected:
        raise RuntimeError(f"{filename} digest mismatch: {actual}")

subprocess.run(
    [
        "mfront",
        "--obuild",
        "--interface=generic",
        "LogarithmicStrainPlasticity.mfront",
    ],
    check=True,
)

namespace = runpy.run_path("reference.py", run_name="__main__")
results = np.asarray(namespace["results"], dtype=float)
curve = np.column_stack((results[:, 1], results[:, 0]))
np.savetxt(
    "reference_curve.csv",
    curve,
    delimiter=",",
    header="load_factor,downward_displacement_m",
    comments="",
)

packages = subprocess.run(
    ["micromamba", "list", "-n", "base", "--json"],
    check=True,
    capture_output=True,
    text=True,
)
manifest = {
    "schema": "agentfem.external-reference-execution.v1",
    "benchmark": "lewandowski_2023_self_weight_beam",
    "source_sha256": EXPECTED,
    "reference_curve_sha256": digest(Path("reference_curve.csv")),
    "points": int(curve.shape[0]),
    "packages": json.loads(packages.stdout),
}
Path("reference_execution.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
