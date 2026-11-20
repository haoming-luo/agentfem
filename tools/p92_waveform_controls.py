"""Research-only endpoint-preserving measured ramp-shape discrimination.

Recorded strain is an input only on strain-controlled ramps. Its deviation
from its own endpoint chord is added to the existing continuous predicted
endpoint ramp. This isolates shape without resetting post-creep strain; it is
not an exact absolute controller replay when measured/predicted endpoints differ.
"""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from tools.p92_two_mechanism import integrate


def attach_waveforms(path, data, *, mode="shape"):
    """Keep switches, hold controls, observations and all fit weights unchanged."""
    if mode not in {"shape", "recorded_prefix"}:
        raise ValueError("Unknown waveform mode")
    if path["source_sha256"] != data["sha256"] or path["protocol"] != data["protocol"]:
        raise ValueError("Waveform source identity changed")
    out = deepcopy(path)
    raw_t, raw_e, cycle = (np.asarray(data[k]) for k in ("time_s", "strain", "cycle"))
    if raw_t.ndim != 1 or raw_e.shape != raw_t.shape or cycle.shape != raw_t.shape:
        raise ValueError("Require aligned waveform source arrays")
    source_ids = np.flatnonzero((cycle >= 1) & (cycle <= path["cycles"]))
    t, e = raw_t[source_ids], raw_e[source_ids]
    duration = path["switches"][-1]["time_s"]
    if (len(t) < 2 or not np.isfinite(t).all() or not np.isfinite(e).all()
            or np.any(np.diff(t) <= 0) or t[0] != 0 or duration > t[-1]+1e-9):
        raise ValueError("Waveform source does not cover the continuous path")
    start = 0.
    prefix = True
    for seg, event in zip(out["segments"], out["switches"], strict=True):
        end = event["time_s"]
        prefix = prefix and not seg["stress_control"]
        # A stress hold has no prescribed strain. A strain dwell stays exactly
        # the declared dwell; instrument drift is not promoted to a command.
        selected = (not seg["stress_control"] and not seg["strain_hold"]
                    if mode == "shape" else prefix)
        if selected:
            ids = np.flatnonzero((t > start) & (t < end))
            times = np.r_[0., t[ids]-start, seg["duration"]]
            values = np.r_[np.interp(start, t, e), e[ids], np.interp(end, t, e)]
            chord = values[0]+(values[-1]-values[0])*times/seg["duration"]
            deviation = values-chord
            deviation[[0, -1]] = 0.
            if mode == "recorded_prefix":
                # Until the first stress hold, continuity from measured e(0)
                # makes the entire imposed strain equal to the recorded PL
                # history. After that switch no measured strain is imposed.
                seg["target"] = float(values[-1])
            slopes = np.diff(values)/np.diff(times)
            seg["waveform"] = dict(times=times, deviation=deviation,
                measured_endpoints=values[[0, -1]].tolist(), source_indices=source_ids[ids].tolist(),
                minimum_recorded_slope_per_s=float(slopes.min()),
                maximum_recorded_slope_per_s=float(slopes.max()),
                maximum_chord_deviation=float(np.max(abs(deviation))))
        start = end
    out["waveform_scope"] = (
        "Measured ramp shape with continuous predicted endpoint chord; no dwell-strain prescription"
        if mode == "shape" else
        "Recorded strain history only before first stress hold; thereafter original controls, no state reset")
    return out


def integrate_waveform(old, duration, target, waveform, mechanisms, *, rtol=1e-8, method="Radau"):
    """Integrate each slope interval exactly at its knot; never interpolate states."""
    if any(np.iscomplexobj(x) for x in (old, waveform["times"], waveform["deviation"], target)):
        raise ValueError("Require real waveform and state")
    state = np.asarray(old, dtype=float).copy()
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError("Require finite ten-component state")
    times = np.asarray(waveform["times"], dtype=float)
    dev = np.asarray(waveform["deviation"], dtype=float)
    if (times.ndim != 1 or len(times) < 2 or dev.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(dev).all()
            or times[0] != 0 or times[-1] != duration or np.any(np.diff(times) <= 0)
            or dev[0] != 0 or dev[-1] != 0 or not np.isfinite(target)):
        raise ValueError("Invalid endpoint-preserving waveform")
    strain = state[9]+(target-state[9])*times/duration+dev
    parts = []
    for a, b, ea, eb in zip(times[:-1], times[1:], strain[:-1], strain[1:], strict=True):
        sol = integrate(state, b-a, (eb-ea)/(b-a), mechanisms, rtol=rtol, method=method)
        parts.append(sol)
        state = sol.y[:, -1].copy()

    def evaluate(query):
        q = np.asarray(query, dtype=float)
        if q.ndim != 1 or not np.isfinite(q).all() or np.any((q < 0) | (q > duration)):
            raise ValueError("Waveform observations must lie in the segment")
        indices = np.minimum(np.searchsorted(times, q, side="right")-1, len(parts)-1)
        result = np.empty((10, len(q)))
        for i in np.unique(indices):
            mask = indices == i
            result[:, mask] = parts[i].sol(q[mask]-times[i])
        return result

    return SimpleNamespace(y=np.concatenate([s.y for s in parts], axis=1), sol=evaluate)
