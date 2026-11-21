"""Independent research replay for the scalar two-rate reference state layout.

Reuses measured controls, not old material states. Returns observations and
the unchanged early-identification residual weights, not a qualification.
"""

import numpy as np

from tools import p92_two_rate_reference
from tools.p92_hold_observations import summarize


def replay(path, material, *, method="Radau", rtol=1e-9, kernel=p92_two_rate_reference):
    # Research-only alternative must preserve the first ten state coordinates.
    if tuple(kernel.STATE_LAYOUT[:10]) != p92_two_rate_reference.STATE_LAYOUT:
        raise ValueError("Incompatible research state layout")
    MODEL_ID, STATE_LAYOUT = kernel.MODEL_ID, kernel.STATE_LAYOUT
    integrate, stored_energy = kernel.integrate, kernel.stored_energy
    state = np.zeros(len(STATE_LAYOUT))
    state[0], state[7] = path["initial_stress"], path["initial_strain"]
    initial_energy = stored_energy(state, material)
    residuals, records = [], []
    holds, cycles = [], []
    peak = valley = state[0]
    for i, segment in enumerate(path["segments"]):
        duration = segment["duration"]
        stress_control = segment["stress_control"]
        if i and path["segments"][i-1]["stress_control"] and segment["target"] < state[7]:
            raise ValueError("Predicted creep leaves the declared subsequent ramp direction")
        if stress_control and "waveform" in segment:
            raise ValueError("Stress hold cannot prescribe a strain waveform")
        waveform = segment.get("waveform")
        times = np.array([0., duration]) if waveform is None else np.asarray(waveform["times"])
        dev = np.zeros(2) if waveform is None else np.asarray(waveform["deviation"])
        if (times.ndim != 1 or len(times) < 2 or dev.shape != times.shape
                or not np.isfinite(times).all() or not np.isfinite(dev).all()
                or times[0] != 0 or times[-1] != duration or np.any(np.diff(times) <= 0)
                or dev[0] != 0 or dev[-1] != 0):
            raise ValueError("Invalid continuous waveform")
        strains = (np.zeros(len(times)) if stress_control else
                   state[7]+(segment["target"]-state[7])*times/duration+dev)
        solutions = []
        for j, (a, b) in enumerate(zip(times[:-1], times[1:], strict=True)):
            solution = integrate(state, b-a, (strains[j+1]-strains[j])/(b-a), material,
                                 stress_control=stress_control, method=method, rtol=rtol)
            solutions.append(solution)
            state = solution.y[:, -1].copy()
        def evaluate(query, times=times, solutions=solutions, duration=duration):
            query = np.asarray(query)
            if query.ndim != 1 or not np.isfinite(query).all() or np.any((query < 0) | (query > duration)):
                raise ValueError("Observation outside control segment")
            indices = np.minimum(np.searchsorted(times, query, side="right")-1, len(solutions)-1)
            sample = np.empty((len(STATE_LAYOUT), len(query)))
            for j in np.unique(indices):
                selected = indices == j
                sample[:, selected] = solutions[j].sol(query[selected]-times[j])
            return sample

        query = np.asarray(segment["sample_times"])
        sample = evaluate(query)
        if stress_control or segment.get("strain_hold", False):
            holds.append(summarize(segment, evaluate, index=i, strain_index=7))
        stress_error = sample[0]-segment["measured_stress"]
        strain_error = sample[7]-segment["measured_strain"]
        channels = [stress_error/100.]
        if stress_control:
            channels.append(strain_error/.002)
        scale = np.sqrt(3*len(path["segments"])*len(channels)*len(query))
        residuals.extend(c/scale for c in channels)
        records.append(dict(segment=i, cycle=segment["cycle"], stress_control=stress_control,
            stress_rms_MPa=float(np.sqrt(np.mean(stress_error**2))),
            strain_rms=float(np.sqrt(np.mean(strain_error**2))),
            sampled_stress_MPa=sample[0].tolist(), sampled_strain=sample[7].tolist(),
            source_indices=segment["source_indices"], end_state=state.tolist()))
        peak = max(peak, *(float(s.y[0].max()) for s in solutions))
        valley = min(valley, *(float(s.y[0].min()) for s in solutions))
        if i+1 == len(path["segments"]) or path["segments"][i+1]["cycle"] != segment["cycle"]:
            cycles.append(dict(cycle=segment["cycle"], sampled_peak_MPa=peak,
                               sampled_valley_MPa=valley, end_state=state.tolist()))
            peak = valley = state[0]
    return np.concatenate(residuals), dict(model_id=MODEL_ID, state_layout=STATE_LAYOUT,
        protocol=path.get("protocol", "synthetic"), cycles=cycles,
        hold_diagnostics=holds, reversed_post_hold_ramps=[], incompatible_continuation=False,
        source_sha256=path["source_sha256"], segments=records,
        mechanical_balance_MPa=float(state[8]-state[9]-stored_energy(state, material)+initial_energy),
        engineering_qualified=False, life_qualified=False)
