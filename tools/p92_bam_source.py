"""Read checksum-verified BAM P92 records; no fitting or life qualification."""

import hashlib
import argparse
import json
from pathlib import Path

import numpy as np


SOURCE = "https://doi.org/10.5281/zenodo.7198219"
CHECKSUMS = {
    "RF": "bcd4654b4fe014e2c08ec49fd1c3a97b",
    "SLC": "e3b828fe33fcb68cb0c222842055d855",
    "SLR": "5214e2becfe6c797584e1f158c9a1f14",
}


def life_observation(data):
    """Admit the source-defined outcome, not the final recorded test cycle."""
    from agentfem.assessments import LifeObservation

    metadata = data["metadata"]
    failure = float(metadata["failure_cycles"])
    total = float(metadata["total_test_cycles"])
    if not (0 < failure <= total and failure.is_integer() and total.is_integer()):
        raise ValueError("Invalid source failure/total cycle counts")
    protocol = data["protocol"]
    controls = {"RF": "strain-controlled symmetric 180 s peak holds",
                "SLR": "RF plus 1800 s zero-strain hold",
                "SLC": "RF plus 1800 s stress-controlled hold"}
    return LifeObservation(
        specimen_id=f"BAM-7198219-{metadata['specimen_id']}", life=failure,
        status="failure", life_unit="cycles", population="P92 virgin pipe parent metal",
        batch_id="BAM-7198219 source pipe cohort; heat number not supplied",
        failure_definition="10% maximum peak load drop relative to stabilized cyclic deformation trend",
        conditions={"temperature_K": 893.15, "mechanical_strain_amplitude": .002,
                    "protocol": protocol, "control": controls[protocol], "environment": "air"},
        source=f"{SOURCE}; {protocol}; sha256:{data['sha256']}",
    )


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


def audit_source(path, *, protocol):
    """Check source extrema and report each recorded path without filling gaps."""
    data = read_source(path, protocol=protocol)
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "[MAX-MIN]")
    end = next(i for i, line in enumerate(lines) if line.strip() == "[Data]")
    rows = []
    for line in lines[start + 3:end]:
        fields = line.split()
        if fields and fields[0].isdigit():
            rows.append([float(v) for v in fields])
    reference = np.asarray(rows)
    if (reference.ndim != 2 or reference.shape[1] != 7
            or not np.isfinite(reference).all()
            or not np.array_equal(reference[:, 0], data["recorded_cycles"])):
        raise ValueError("MAX-MIN cycle identity/shape does not match recorded paths")
    records = []
    boundaries = np.r_[0, np.flatnonzero(np.diff(data["cycle"])) + 1, len(data["cycle"])]
    errors = []
    for row, a, b in zip(reference, boundaries[:-1], boundaries[1:], strict=True):
        t, s, e, temp = (data[key][a:b] for key in
                         ("time_s", "stress_Pa", "strain", "temperature_K"))
        actual = np.array([s.max()/1e6, s.min()/1e6, e.max()*100, e.min()*100,
                           temp.max()-273.15, temp.min()-273.15])
        errors.append(np.abs(actual-row[1:]))
        records.append({
            "cycle": int(row[0]), "samples": int(b-a),
            "start_s": float(t[0]), "end_s": float(t[-1]),
            "stress_max_Pa": float(s.max()), "stress_min_Pa": float(s.min()),
            "strain_range": float(np.ptp(e)),
            "endpoint_strain_difference": float(e[-1]-e[0]),
            "endpoint_stress_difference_Pa": float(s[-1]-s[0]),
            "sampled_path_work_J_per_m3": float(np.sum((s[1:]+s[:-1])*0.5*np.diff(e))),
        })
    max_error = np.max(errors, axis=0)
    # Source table precision: stresses 0.01 MPa, strains .001 percent, T .1 C.
    if np.any(max_error > np.array([.010001, .010001, .001001, .001001, .100001, .100001])):
        raise ValueError(f"Source extrema disagree with histories: {max_error.tolist()}")
    return {
        "schema": "agentfem.research.bam-source-audit.v1",
        "source": SOURCE, "protocol": protocol, "sha256": data["sha256"],
        "metadata": data["metadata"], "history_rows": len(data["time_s"]),
        "recorded_cycle_count": len(records),
        "max_extrema_discrepancy_MPa_MPa_percent_percent_C_C": max_error.tolist(),
        "records": records, "engineering_qualified": False,
        "life_observation": life_observation(data).as_dict(),
        "scope": "Source consistency and sampled nominal axial path work; no material prediction",
        "work_scope": "No endpoint closure, omitted-cycle integration or dissipation interpretation",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--protocol", required=True, choices=tuple(CHECKSUMS))
    args = parser.parse_args()
    print(json.dumps(audit_source(args.path, protocol=args.protocol), indent=2, allow_nan=False))
