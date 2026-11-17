"""Frozen-parameter mechanism ablations; diagnostic, never a material fit."""

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source
from tools.p92_bounded_softening import _trial_multiplier, compare_candidate
from tools.p92_measured_controls import measured_path, replay
from tools.p92_published_recovery_reference import Parameters
from tools.p92_two_mechanism import Mechanism


def terms(state, mechanism):
    """Signed MPa/s contributions; not dissipation or integrated strain shares."""
    sigma, x1, x2, R, accumulated = np.asarray(state, dtype=float)
    p = mechanism.parameters
    effective = sigma-x1-x2
    speed = (max(abs(effective)-p.k-R, 0.)/p.Z)**p.n if mechanism.enabled else 0.
    flow = speed*np.sign(effective)
    phi, _ = _trial_multiplier(accumulated, limit=2., rate=.1)
    gamma = p.gamma1+p.gamma2*np.exp(-p.q*accumulated)
    gamma_r = p.gamma_r1+p.gamma_r2*np.exp(-p.r*accumulated)
    backstress = []
    for x, C, a in ((x1, p.C1, p.a1), (x2, p.C2, p.a2)):
        backstress.append(dict(hardening=C*a*flow,
            dynamic_recovery=-C*phi*(abs(x)*phi/a)**mechanism.exponent*x*speed,
            static_recovery=-gamma*abs(x)**p.m*np.sign(x) if mechanism.enabled else 0.))
    return dict(signed_flow_per_s=float(flow), strength_MPa=float(p.k+R),
                effective_stress_MPa=float(effective), backstress_MPa_per_s=backstress,
                isotropic_MPa_per_s=dict(accumulation=p.b*(p.Q-R)*speed+p.H*speed,
                    time_recovery=gamma_r*abs(p.Qr-R)**p.mr*np.sign(p.Qr-R) if mechanism.enabled else 0.))


def variants(mechanisms):
    result = {"frozen_baseline": mechanisms}
    for branch, name in enumerate(("fast", "slow")):
        for kind, change in (
            ("no_static_backstress", dict(gamma1=0., gamma2=0.)),
            ("no_time_isotropic", dict(gamma_r1=0., gamma_r2=0.)),
        ):
            candidate = list(mechanisms)
            candidate[branch] = replace(candidate[branch], parameters=replace(candidate[branch].parameters, **change))
            result[f"{name}_{kind}"] = tuple(candidate)
    # Remove negative accumulation targets, retain -b*R*p_dot restoration.
    result["both_no_negative_accumulation_targets"] = tuple(
        replace(m, parameters=replace(m.parameters, Q=0., H=0.)) for m in mechanisms)
    return result


def screen(frozen, sources):
    cards = frozen["parameters"]
    if sha256(json.dumps(cards, sort_keys=True).encode()).hexdigest() != frozen["parameter_sha256"]:
        raise ValueError("Changed frozen parameters")
    if len(sources) != 3 or {d["protocol"] for d in sources} != {"RF", "SLR", "SLC"}:
        raise ValueError("Require all three protocols")
    mechanisms = tuple(Mechanism(Parameters(**m["parameters"]), m["exponent"], m["enabled"]) for m in cards)
    paths = []
    for data in sources:
        if data["sha256"] != frozen["source_sha256"][data["protocol"]]:
            raise ValueError("Changed source identity")
        paths.append(measured_path(data, cycles=10))
    records = []
    for name, candidate in variants(mechanisms).items():
        for data, path in zip(sources, paths, strict=True):
            _, report = replay(path, candidate)
            _, fine = replay(path, candidate, method="BDF", rtol=1e-10)
            if [x["cycle"] for x in report["reversed_post_hold_ramps"]] != [x["cycle"] for x in fine["reversed_post_hold_ramps"]]:
                raise ValueError("Independent integrators disagree on failed cycle")
            delta = np.abs(np.asarray([r["state"] for r in report["segment_endpoints"]])-
                           np.asarray([r["state"] for r in fine["segment_endpoints"]]))
            holds = []
            old = np.zeros(10)
            old[0], old[9] = path["initial_stress"], path["initial_strain"]
            for endpoint in report["segment_endpoints"]:
                state = np.asarray(endpoint["state"])
                if endpoint["stress_control"]:
                    holds.append(dict(cycle=endpoint["cycle"], strain_increment=float(state[9]-old[9]),
                        entry=[terms(np.r_[old[0], old[o:o+4]], m) for o,m in zip((1,5),candidate)],
                        exit=[terms(np.r_[state[0], state[o:o+4]], m) for o,m in zip((1,5),candidate)],
                        branch_accumulated_strain_increment=(state[[4,8]]-old[[4,8]]).tolist()))
                old = state
            comparison = compare_candidate(data, report) if report["cycles"] else None
            if comparison:
                comparison.pop("groups")
                for row in comparison["records"]:
                    row["role"] = "early_development" if row["cycle"] <= 3 else "inspected_development"
            records.append(dict(variant=name, protocol=data["protocol"], mechanisms=[asdict(m) for m in candidate],
                completed_cycles=len(report["cycles"]), failure=report["reversed_post_hold_ramps"],
                comparison=comparison, stress_holds=holds,
                integrator_max_state_difference=delta.max(axis=0).tolist()))
            print(json.dumps(dict(variant=name, protocol=data["protocol"], complete=len(report["cycles"]),
                                  failure=report["reversed_post_hold_ramps"])), flush=True)
    return dict(schema="agentfem.research.p92-recovery-discriminator.v1", records=records,
        frozen_parameter_sha256=frozen["parameter_sha256"], source_sha256=frozen["source_sha256"],
        scope="Prespecified ablations, no refit; all inspected data are development evidence",
        engineering_qualified=False, life_qualified=False, independent_validation=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path)
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new evidence path")
    result = screen(json.loads(args.frozen.read_text()),
        [read_source(args.sources/f"Data_{p}.lis", protocol=p) for p in ("RF", "SLR", "SLC")])
    result["driver_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
