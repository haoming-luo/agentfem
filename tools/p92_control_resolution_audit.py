"""Joint strain/stress endpoint sensitivity; no changed response gate."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_response_screen import POLICY


def endpoint_budget(strain, stress, duration, modulus, *, strain_quantum, stress_quantum):
    e, s = np.asarray(strain, dtype=float), np.asarray(stress, dtype=float)
    if (e.shape != (2,) or s.shape != (2,) or not np.isfinite(e).all() or not np.isfinite(s).all()
            or not np.isfinite([duration, modulus, strain_quantum, stress_quantum]).all()
            or min(duration, modulus) <= 0 or min(strain_quantum, stress_quantum) < 0):
        raise ValueError("Require two finite endpoints and positive time/modulus")
    de, ds = float(e[1]-e[0]), float(s[1]-s[0])
    increment = de-ds/modulus
    bound = 2*strain_quantum+2*stress_quantum/modulus
    return dict(strain_change=de, stress_change_MPa=ds,
        elastic_equivalent_MPa=modulus*de, inferred_nonelastic_increment=increment,
        inferred_nonelastic_rate_per_s=increment/duration,
        endpoint_quantum_sensitivity=bound,
        endpoint_quantum_sensitivity_per_s=bound/duration,
        sign_robust_to_quantum=bool(abs(increment) > bound),
        stress_only_resolution_units=abs(ds)/stress_quantum if stress_quantum else None,
        scope="Independent plus/minus one recorded quantum at each endpoint; not a confidence interval or instrument accuracy")


def audit(source_dir, modulus, expected_sources):
    records, identities = [], {}
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != expected_sources[protocol]:
            raise ValueError("Source identity changed")
        identities[protocol] = source["sha256"]
        for i, segment in enumerate(measured_path(source, cycles=3)["segments"]):
            if not segment["strain_hold"]:
                continue
            tail = segment["tail"]
            duration = float(tail["times"][-1]-tail["times"][0])
            records.append(dict(protocol=protocol, cycle=segment["cycle"], segment=i,
                source_indices=tail["source_indices"], duration_s=duration,
                strain_endpoints=tail["strain"].tolist(), stress_endpoints_MPa=tail["stress"].tolist(),
                budget=endpoint_budget(tail["strain"], tail["stress"], duration, modulus,
                    strain_quantum=POLICY["strain_resolution"], stress_quantum=POLICY["stress_resolution_MPa"])))
    return dict(schema="agentfem.research.control-resolution.v1", records=records,
        modulus_MPa=modulus, source_sha256=identities, policy=dict(POLICY),
        fitted=False, acceptance_thresholds_changed=False, engineering_qualified=False,
        scope="Input-channel sensitivity alongside unchanged nominal-control failures; not a solver replay")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("frozen", type=Path)
    p.add_argument("source_dir", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    raw = args.frozen.read_bytes()
    frozen = json.loads(raw)
    result = audit(args.source_dir, frozen["parameters"]["backbone"]["modulus"], frozen["source_sha256"])
    result.update(frozen_sha256=sha256(raw).hexdigest(), driver_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    write_new(args.output, result)
