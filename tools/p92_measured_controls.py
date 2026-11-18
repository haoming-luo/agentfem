"""Source-aligned BAM cycle controls, with recorded switch brackets.

Research driver: peak/zero crossings are interpolated between recorded strain
samples; dwell durations come from the published protocol. These are not
controller logs. Brackets remain evidence, not claimed exact switch times.
SLC hold strain is an observation, never a prescribed input or a state reset.
"""

import numpy as np

from tools.p92_two_mechanism import integrate


def crossing(t, e, ids, target, direction, lo, hi):
    candidates = ids[1:]
    candidates = candidates[(t[candidates] >= lo) & (t[candidates] <= hi)]
    for j in candidates:
        a, b = direction*(e[j-1]-target), direction*(e[j]-target)
        if a < 0 <= b:
            time = t[j-1]+(target-e[j-1])*(t[j]-t[j-1])/(e[j]-e[j-1])
            return dict(time_s=float(time), bracket_s=[float(t[j-1]), float(t[j])],
                        source_indices=[int(j-1), int(j)], method="strain crossing")
    raise ValueError("Missing measured crossing in the declared protocol window")


def departure(t, e, ids, event, duration, direction):
    expected = event["time_s"]+duration
    candidates = ids[2:]
    candidates = candidates[(t[candidates] >= expected-2) & (t[candidates] <= expected+2)]
    for j in candidates:
        rate = direction*(e[j]-e[j-1])/(t[j]-t[j-1])
        if rate > .0004:  # separate nominal .001/s ramps from dwell drift
            lo, hi = float(t[j-2]), float(t[j])
            consistent = lo <= expected <= hi
            return dict(time_s=expected if consistent else (lo+hi)/2, bracket_s=[lo, hi],
                        source_indices=[int(j-2), int(j)],
                        nominal_time_s=expected, nominal_within_bracket=consistent,
                        method="published dwell duration within measured departure bracket" if consistent
                        else "measured departure bracket midpoint; nominal timing inconsistent")
    raise ValueError("Missing measured dwell departure")


def measured_path(data, *, cycles, switch_bound=None):
    if switch_bound not in {None, "lower", "upper"}:
        raise ValueError("Switch bound must be lower or upper")
    if type(cycles) is not int or cycles < 1 or data["protocol"] not in {"RF", "SLR", "SLC"}:
        raise ValueError("Require supported protocol and a positive cycle count")
    t, e, s, c = (np.asarray(data[k]) for k in ("time_s", "strain", "stress_Pa", "cycle"))
    if (any(a.ndim != 1 or a.shape != t.shape or not np.isfinite(a).all() for a in (t, e, s, c))
            or len(t) < 3 or t[0] != 0 or c[0] != 1 or np.any(np.diff(c) < 0)):
        raise ValueError("Require aligned finite records starting at the first cycle")
    selected = np.flatnonzero((c >= 1) & (c <= cycles))
    if (not np.array_equal(np.unique(c[selected]), np.arange(1, cycles+1))
            or np.any(np.diff(t[selected]) <= 0) or np.any(np.diff(t[selected]) > 6.1)):
        raise ValueError("Missing or discontinuous measured cycles")
    segments, events = [], []
    previous = 0.
    for cycle in range(1, cycles+1):
        ids = np.flatnonzero(c == cycle)
        start = t[ids[0]]
        if data["protocol"] == "RF":
            first = crossing(t, e, ids, .002, 1, start, start+5)
            leave = departure(t, e, ids, first, 180., -1)
            second = crossing(t, e, ids, -.002, -1, leave["time_s"], leave["time_s"]+6)
            leave_second = departure(t, e, ids, second, 180., 1)
            sequence = [(first, .002, False), (leave, .002, False),
                        (second, -.002, False), (leave_second, -.002, False)]
        else:
            first = crossing(t, e, ids, -.002, -1, start, start+5)
            leave = departure(t, e, ids, first, 180., 1)
            zero = crossing(t, e, ids, 0., 1, leave["time_s"], leave["time_s"]+5)
            leave_zero = departure(t, e, ids, zero, 1800., 1)
            second = crossing(t, e, ids, .002, 1, leave_zero["time_s"], leave_zero["time_s"]+5)
            leave_second = departure(t, e, ids, second, 180., -1)
            sequence = [(first, -.002, False), (leave, -.002, False), (zero, 0., False),
                        (leave_zero, None if data["protocol"] == "SLC" else 0., data["protocol"] == "SLC"),
                        (second, .002, False), (leave_second, .002, False)]
        sequence.append((dict(time_s=float(t[ids[-1]]), bracket_s=[float(t[ids[-1]])]*2,
                              source_indices=[int(ids[-1])], method="last recorded cycle sample"),
                         float(e[ids[-1]]), False))
        for event, target, stress_control in sequence:
            if switch_bound is not None:
                event = dict(event, time_s=event["bracket_s"][0 if switch_bound == "lower" else -1],
                             sensitivity_bound=switch_bound)
            end = event["time_s"]
            if end <= previous:
                raise ValueError("Unordered reconstructed switches")
            observed = ids[(t[ids] > previous) & (t[ids] <= end)]
            if len(observed) < 2:
                raise ValueError("Insufficient observations between switches")
            desired = previous+(end-previous)*np.r_[0., np.geomspace(.001, 1., 15)]
            chosen = np.unique([observed[np.argmin(abs(t[observed]-x))] for x in desired])
            segments.append(dict(cycle=cycle, duration=end-previous, target=target,
                stress_control=stress_control, sample_times=t[chosen]-previous,
                measured_stress=s[chosen]/1e6, measured_strain=e[chosen],
                source_indices=chosen.tolist()))
            events.append(dict(cycle=cycle, stress_control=stress_control, **event))
            previous = end
    return dict(protocol=data["protocol"], initial_stress=float(s[0]/1e6),
                initial_strain=float(e[0]), segments=segments, switches=events,
                source_sha256=data["sha256"], cycles=cycles)


def timing_sensitivity(data, mechanisms, *, cycles=10):
    """Frozen-parameter deterministic bracket perturbation, not a confidence interval."""
    records = {}
    for side in ("lower", "upper"):
        _, result = replay(measured_path(data, cycles=cycles, switch_bound=side), mechanisms)
        records[side] = dict(cycles=result["cycles"],
                             reversed_post_hold_ramps=result["reversed_post_hold_ramps"])
    return records


def replay(path, mechanisms, *, rtol=1e-8, method="Radau", allow_incompatible=False):
    state = np.zeros(10)
    state[0], state[9] = path["initial_stress"], path["initial_strain"]
    residuals, records, endpoints, reversals = [], [], [], []
    peak = valley = state[0]
    for i, seg in enumerate(path["segments"]):
        if (i and path["segments"][i-1]["stress_control"]
                and seg["target"] < state[9]):
            reversals.append(dict(cycle=seg["cycle"], strain=float(state[9]), target=seg["target"]))
            if not allow_incompatible:
                break  # A partial failed cycle is not a predicted complete cycle.
        rate = 0. if seg["stress_control"] else (seg["target"]-state[9])/seg["duration"]
        sol = integrate(state, seg["duration"], rate, mechanisms,
                        stress_control=seg["stress_control"], rtol=rtol, method=method)
        sample = sol.sol(seg["sample_times"])
        state = sol.y[:, -1].copy()
        peak, valley = max(peak, float(sol.y[0].max())), min(valley, float(sol.y[0].min()))
        channels = [(sample[0]-seg["measured_stress"])/100.]
        if seg["stress_control"]:
            channels.append((sample[9]-seg["measured_strain"])/.002)
        scale = np.sqrt(3*len(path["segments"])*len(channels)*len(sample[0]))
        residuals.extend(channel/scale for channel in channels)
        endpoints.append(dict(cycle=seg["cycle"], stress_control=seg["stress_control"], state=state.tolist()))
        if i+1 == len(path["segments"]) or path["segments"][i+1]["cycle"] != seg["cycle"]:
            records.append(dict(cycle=seg["cycle"], sampled_peak_MPa=peak,
                                sampled_valley_MPa=valley, end_state=state.tolist()))
            peak = valley = state[0]
    return np.concatenate(residuals), dict(protocol=path["protocol"], cycles=records,
        segment_endpoints=endpoints, reversed_post_hold_ramps=reversals,
        incompatible_continuation=allow_incompatible,
        source_sha256=path["source_sha256"],
        scope="Measured-endpoint replay, not autonomous controller prediction",
        engineering_qualified=False, life_qualified=False, independent_validation=False)
