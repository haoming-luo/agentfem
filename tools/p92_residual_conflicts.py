"""Early-only residual accounting for frozen matched cards, without fitting."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_measured_controls import measured_path
from tools.p92_measured_fit import write_new


def split_cost(old, new, times, *, duration, hold, scale):
    old, new, times = (np.asarray(x, dtype=float) for x in (old, new, times))
    if (old.ndim != 1 or not len(old) or old.shape != new.shape or old.shape != times.shape
            or not all(np.isfinite(x).all() for x in (old, new, times))
            or not np.isfinite([duration, scale]).all() or duration <= 0 or scale <= 0
            or np.any(times < 0) or np.any(times > duration) or np.any(np.diff(times) <= 0)):
        raise ValueError("Require aligned finite residual observations and positive scales")
    early = times <= min(10., .1*duration)
    late = times >= .8*duration
    windows = {"early": early, "middle": ~(early | late), "late": late} if hold else {
        "whole": np.ones(len(times), dtype=bool)}
    rows = []
    for name, mask in windows.items():
        if not np.any(mask):
            continue
        a, b = old[mask]/scale, new[mask]/scale
        delta = b-a
        rows.append(dict(window=name, samples=int(mask.sum()),
            baseline_cost=float(a@a/2), candidate_cost=float(b@b/2),
            linear_change=float(a@delta), quadratic_change=float(delta@delta/2),
            baseline_bias=float(old[mask].mean()), candidate_bias=float(new[mask].mean()),
            baseline_rms=float(np.sqrt(np.mean(old[mask]**2))),
            candidate_rms=float(np.sqrt(np.mean(new[mask]**2)))))
        # Diagnostic anchoring only: never change the objective or pass gates.
        # The first recorded sample need not coincide with the control switch.
        for label, residual in (("baseline", old), ("candidate", new)):
            anchor = residual[0]
            shape = residual[mask]-anchor
            rows[-1][f"{label}_anchor_residual"] = float(anchor)
            rows[-1][f"{label}_shape_rms"] = float(np.sqrt(np.mean(shape**2)))
            rows[-1][f"{label}_offset_cost"] = float(mask.sum()*anchor**2/(2*scale**2))
            rows[-1][f"{label}_shape_cost"] = float(shape@shape/(2*scale**2))
            rows[-1][f"{label}_offset_shape_cross"] = float(anchor*shape.sum()/scale**2)
        rows[-1]["anchor_time"] = float(times[0])
    return rows


def audit(baseline, candidate, source_dir):
    if baseline["source_sha256"] != candidate["source_sha256"]:
        raise ValueError("Different source identities")
    records = []
    for protocol in ("RF", "SLR", "SLC"):
        source = read_source(source_dir/f"Data_{protocol}.lis", protocol=protocol)
        if source["sha256"] != baseline["source_sha256"][protocol]:
            raise ValueError("Source identity changed")
        path = measured_path(source, cycles=3)
        previous_target, previous_direction = path["initial_strain"], 0.
        old_segments = baseline["protocols"][protocol]["prediction"]["segments"]
        new_segments = candidate["protocols"][protocol]["prediction"]["segments"]
        for frozen in (baseline, candidate):
            item = frozen["protocols"][protocol]
            prediction = item["prediction"]
            if (not item.get("numerical", {}).get("passed", False)
                    or prediction["source_sha256"] != source["sha256"]
                    or prediction["protocol"] != protocol
                    or prediction["model_id"] != frozen["model_id"]):
                raise ValueError("Require verified, aligned frozen prediction identities")
        if min(len(old_segments), len(new_segments)) < len(path["segments"]):
            raise ValueError("Frozen trajectory does not cover early observations")
        for i, segment in enumerate(path["segments"]):
            hold = segment["stress_control"] or segment["strain_hold"]
            if segment["stress_control"]:
                phase = "stress_hold"
            elif segment["strain_hold"]:
                phase = "negative_hold" if segment["target"] < 0 else (
                    "positive_hold" if segment["target"] > 0 else "zero_hold")
            else:
                direction = np.sign(segment["target"]-previous_target) if previous_target is not None else 1.
                phase = ("post_creep_reload" if i and path["segments"][i-1]["stress_control"] else
                         "initial_ramp" if not i else
                         "reversal_ramp" if direction*previous_direction < 0 else "continuing_ramp")
                previous_direction = direction
            previous_target = segment["target"]
            old, new = old_segments[i], new_segments[i]
            if any(r["cycle"] != segment["cycle"] or r["segment"] != i
                   or r["source_indices"] != segment["source_indices"]
                   for r in (old, new)):
                raise ValueError("Frozen observation identity differs from early path")
            channels = [("stress_MPa", "sampled_stress_MPa", "measured_stress", 100.)]
            if segment["stress_control"]:
                channels.append(("strain", "sampled_strain", "measured_strain", .002))
            for channel, key, observed_key, normalization in channels:
                observed = np.asarray(segment[observed_key])
                if any(np.asarray(r[key]).shape != observed.shape for r in (old, new)):
                    raise ValueError("Frozen sample shape differs from observations")
                scale = normalization*np.sqrt(3*len(path["segments"])*len(channels)*len(observed))
                rows = split_cost(np.asarray(old[key])-observed, np.asarray(new[key])-observed,
                    segment["sample_times"], duration=segment["duration"], hold=hold, scale=scale)
                records.extend(dict(protocol=protocol, cycle=segment["cycle"], segment=i,
                                    phase=phase, channel=channel, **row) for row in rows)
    groups = {}
    for row in records:
        key = f'{row["protocol"]}/{row["phase"]}/{row["channel"]}/{row["window"]}'
        group = groups.setdefault(key, {k: 0. for k in (
            "baseline_cost", "candidate_cost", "linear_change", "quadratic_change")})
        for k in group:
            group[k] += row[k]
    return dict(schema="agentfem.research.early-residual-conflicts.v1", records=records, groups=groups,
        source_sha256=baseline["source_sha256"], training_cycles=[1, 2, 3],
        baseline_cost=sum(r["baseline_cost"] for r in records),
        candidate_cost=sum(r["candidate_cost"] for r in records),
        fitted=False, engineering_qualified=False,
        scope="Exact early objective partition; finite response-change decomposition, not a parameter gradient or causal mechanism proof")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("source_dir", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    raw = [args.baseline.read_bytes(), args.candidate.read_bytes()]
    result = audit(*(json.loads(x) for x in raw), args.source_dir)
    result["frozen_sha256"] = [sha256(x).hexdigest() for x in raw]
    result["driver_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    write_new(args.output, result)
