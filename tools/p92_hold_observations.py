"""Hold diagnostics with explicit strain layout and actual recorded secants."""

import numpy as np


def summarize(segment, evaluate, *, index, strain_index):
    times = np.asarray(segment["sample_times"])
    sample = evaluate(times)
    early = times <= min(10., .1*segment["duration"])
    tail = segment.get("tail")
    late_mask = times >= .8*segment["duration"]
    late_times = np.asarray(tail["times"]) if tail is not None else times[late_mask]
    late_sample = evaluate(late_times) if len(late_times) >= 2 else None
    fields = {}
    for name, row, key in (("stress_MPa", 0, "stress"), ("strain", strain_index, "strain")):
        predicted = sample[row]
        observed = np.asarray(segment["measured_"+key])
        fields[name] = dict(rms=float(np.sqrt(np.mean((predicted-observed)**2))),
            observed_max_abs=float(np.max(abs(observed))), observed_excursion=float(np.ptp(observed)),
            last_observed=float(observed[-1]), last_predicted=float(predicted[-1]))
        if np.count_nonzero(early) >= 2:
            dt = times[early][-1]-times[early][0]
            fields[name]["early_secant_observed_per_s"] = float((observed[early][-1]-observed[early][0])/dt)
            fields[name]["early_secant_predicted_per_s"] = float((predicted[early][-1]-predicted[early][0])/dt)
        if late_sample is not None:
            late_observed = np.asarray(tail[key]) if tail is not None else observed[late_mask]
            dt = late_times[-1]-late_times[0]
            fields[name]["late_secant_observed_per_s"] = float((late_observed[-1]-late_observed[0])/dt)
            fields[name]["late_secant_predicted_per_s"] = float((late_sample[row][-1]-late_sample[row][0])/dt)
    return dict(cycle=segment["cycle"], segment=index, stress_control=segment["stress_control"],
        duration_s=segment["duration"], sample_times_s=times.tolist(),
        source_indices=segment.get("source_indices", []), fields=fields,
        late_secant_times_s=late_times.tolist(),
        late_secant_source_indices=tail["source_indices"] if tail is not None else [])
