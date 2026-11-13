"""Read checksum-verified BAM P92 records; no fitting or life qualification."""

import hashlib
from pathlib import Path

import numpy as np


SOURCE = "https://doi.org/10.5281/zenodo.7198219"
CHECKSUMS = {
    "RF": "bcd4654b4fe014e2c08ec49fd1c3a97b",
    "SLC": "e3b828fe33fcb68cb0c222842055d855",
    "SLR": "5214e2becfe6c797584e1f158c9a1f14",
}


def read_source(path, *, protocol):
    """Accept only the published file; return SI arrays with recorded gaps intact."""
    if protocol not in CHECKSUMS:
        raise ValueError("protocol must be RF, SLC or SLR")
    raw = Path(path).read_bytes()
    if hashlib.md5(raw, usedforsecurity=False).hexdigest() != CHECKSUMS[protocol]:
        raise ValueError("BAM source checksum mismatch: incomplete or changed file")
    lines = raw.decode("latin-1").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "[Data]")
    if lines[start + 1].split() != ["Time", "Cycle", "Stress", "Strain", "Temperature"]:
        raise ValueError("Unexpected BAM history columns")
    if lines[start + 2].split() != ["s", "No", "MPa", "%", "°C"]:
        raise ValueError("Unexpected BAM history units")
    rows = np.array([[float(v) for v in line.split()] for line in lines[start + 3:]
                     if line.strip()], dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 5 or not np.isfinite(rows).all():
        raise ValueError("Invalid BAM history values")
    if np.any(np.diff(rows[:, 0]) < 0) or np.any(np.diff(rows[:, 1]) < 0):
        raise ValueError("BAM history order is not monotone")
    if np.any(rows[:, 1] != np.floor(rows[:, 1])) or np.any(rows[:, 1] < 1):
        raise ValueError("BAM cycle identity must be positive integers")
    cycles = rows[:, 1].astype(np.int64)
    metadata = {}
    allowed = {"Specimen Designation (Internal)": "specimen_id",
               "Cycles to Failure, Nf": "failure_cycles",
               "Total No. of Cycles, N": "total_test_cycles"}
    for line in lines[:start]:
        fields = [value.strip() for value in line.split("\t") if value.strip()]
        if fields and fields[0] in allowed:
            metadata[allowed[fields[0]]] = fields[-1]
    # Keep only scientific identity; do not copy operator names or other header PII.
    return {
        "source": SOURCE, "protocol": protocol,
        "metadata": metadata,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "time_s": rows[:, 0], "cycle": cycles,
        "stress_Pa": rows[:, 2] * 1e6,
        "strain": rows[:, 3] / 100,
        "temperature_K": rows[:, 4] + 273.15,
        "recorded_cycles": np.unique(cycles),
        "engineering_qualified": False,
    }
