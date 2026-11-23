"""Fixed-card switch-bracket sensitivity, not a fitted controller or error bar."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new
from tools.p92_recoverable_frozen import decode_material
from tools.p92_two_rate_controls import replay
from tools.p92_waveform_controls import attach_waveforms


def common_support(paths):
    """A comparison may move switches, never silently change observations."""
    lower, upper = paths
    if len(lower["segments"]) != len(upper["segments"]):
        raise ValueError("Different control paths")
    for a, b in zip(lower["segments"], upper["segments"], strict=True):
        for key in ("source_indices", "measured_stress", "measured_strain"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError("Different observation support")
        for key in ("stress_control", "strain_hold", "target"):
            if a[key] != b[key]:
                raise ValueError("Different control classes")
        if a["tail"]["source_indices"] != b["tail"]["source_indices"]:
            raise ValueError("Different tail support")


def audit(frozen, source_dir):
    if set(frozen["protocols"]) != {"RF", "SLR", "SLC"} or not all(
            item.get("numerical", {}).get("passed", False) for item in frozen["protocols"].values()):
        raise ValueError("Require numerically verified three-protocol frozen card")
    module, material = decode_material(frozen["parameters"], frozen["model_id"])
    records = {}
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != frozen["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        paths = [attach_waveforms(measured_path(source, cycles=1, switch_bound=side), source)
                 for side in ("lower", "upper")]
        common_support(paths)
        records[protocol] = {}
        for side, path in zip(("lower", "upper"), paths, strict=True):
            try:
                _, result = replay(path, material, kernel=module)
            except (ValueError, RuntimeError) as exc:
                records[protocol][side] = dict(completed=False,
                    failure=dict(type=type(exc).__name__, message=str(exc)))
                continue
            records[protocol][side] = dict(completed=True, switches=path["switches"],
                segments=result["segments"], holds=result["hold_diagnostics"],
                mechanical_balance_MPa=result["mechanical_balance_MPa"])
        print(json.dumps(dict(protocol=protocol,
            completed={k: v["completed"] for k, v in records[protocol].items()})), flush=True)
    return dict(schema="agentfem.research.recoverable-switch-audit.v1",
        parameters=frozen["parameters"], model_id=frozen["model_id"],
        source_sha256=frozen["source_sha256"], protocols=records,
        scope="First-cycle all-lower/all-upper switch perturbation; not an exhaustive envelope or confidence interval",
        common_observations=True, fitted=False, acceptance_thresholds_changed=False,
        engineering_qualified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.frozen.read_bytes()
    names = ("p92_recoverable_switch_audit.py", "p92_recoverable_frozen.py",
             "p92_recoverable_mobility_reference.py", "p92_recoverable_spectrum.py",
             "p92_two_rate_reference.py", "p92_two_rate_controls.py",
             "p92_measured_controls.py", "p92_waveform_controls.py", "p92_hold_observations.py",
             "p92_bam_source.py")
    def identities():
        return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    before = identities()
    result = audit(json.loads(raw), args.source_dir)
    if identities() != before or args.frozen.read_bytes() != raw:
        raise RuntimeError("Audit input changed during replay")
    result.update(driver_sha256=before, frozen_sha256=sha256(raw).hexdigest())
    write_new(args.output, result)
