"""Hold diagnostics with explicit strain layout and actual recorded secants."""

import numpy as np


def summarize(segment, evaluate, *, index, strain_index):
    if np.iscomplexobj(segment["sample_times"]):
        raise ValueError("Hold diagnostics require real observation times")
    times = np.asarray(segment["sample_times"], dtype=float)
    duration = segment["duration"]
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or not np.isfinite(duration) or duration <= 0
            or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > duration):
        raise ValueError("Hold diagnostics require ordered observations inside the hold")
    sample = np.asarray(evaluate(times))
    if (sample.ndim != 2 or sample.shape[1] != len(times)
            or not isinstance(strain_index, (int, np.integer))
            or not 0 < strain_index < sample.shape[0]
            or np.iscomplexobj(sample) or not np.isfinite(sample).all()):
        raise ValueError("Hold prediction must provide finite stress and explicit strain rows")
    early = times <= min(10., .1*segment["duration"])
    tail = segment.get("tail")
    late_mask = times >= .8*segment["duration"]
    late_times = np.asarray(tail["times"]) if tail is not None else times[late_mask]
    if (late_times.ndim != 1 or np.iscomplexobj(late_times) or not np.isfinite(late_times).all()
            or np.any(np.diff(late_times) <= 0)
            or np.any(late_times < 0) or np.any(late_times > duration)):
        raise ValueError("Hold tail must contain ordered observations inside the hold")
    late_sample = np.asarray(evaluate(late_times)) if len(late_times) >= 2 else None
    if late_sample is not None and (late_sample.shape != (sample.shape[0], len(late_times))
            or np.iscomplexobj(late_sample) or not np.isfinite(late_sample).all()):
        raise ValueError("Hold tail prediction must match its observations")
    fields = {}
    for name, row, key in (("stress_MPa", 0, "stress"), ("strain", strain_index, "strain")):
        predicted = sample[row]
        observed = np.asarray(segment["measured_"+key])
        if (observed.shape != times.shape or np.iscomplexobj(observed)
                or not np.isfinite(observed).all()):
            raise ValueError("Hold observations must match their sample times")
        residual = predicted-observed
        fields[name] = dict(rms=float(np.sqrt(np.mean((predicted-observed)**2))),
            observed_max_abs=float(np.max(abs(observed))), observed_excursion=float(np.ptp(observed)),
            first_observed=float(observed[0]), first_predicted=float(predicted[0]),
            first_residual=float(residual[0]),
            last_observed=float(observed[-1]), last_predicted=float(predicted[-1]),
            last_residual=float(residual[-1]),
            anchored_shape_rms=float(np.sqrt(np.mean((residual-residual[0])**2))),
            anchored_shape_last_residual=float(residual[-1]-residual[0]))
        if np.count_nonzero(early) >= 2:
            dt = times[early][-1]-times[early][0]
            fields[name]["early_secant_observed_per_s"] = float((observed[early][-1]-observed[early][0])/dt)
            fields[name]["early_secant_predicted_per_s"] = float((predicted[early][-1]-predicted[early][0])/dt)
        if late_sample is not None:
            late_observed = np.asarray(tail[key]) if tail is not None else observed[late_mask]
            if (late_observed.shape != late_times.shape or np.iscomplexobj(late_observed)
                    or not np.isfinite(late_observed).all()):
                raise ValueError("Hold tail observations must match their sample times")
            dt = late_times[-1]-late_times[0]
            fields[name]["late_secant_observed_per_s"] = float((late_observed[-1]-late_observed[0])/dt)
            fields[name]["late_secant_predicted_per_s"] = float((late_sample[row][-1]-late_sample[row][0])/dt)
    return dict(cycle=segment["cycle"], segment=index, stress_control=segment["stress_control"],
        duration_s=segment["duration"], sample_times_s=times.tolist(),
        first_observation_time_s=float(times[0]), last_observation_time_s=float(times[-1]),
        anchor_scope="First recorded sample, not extrapolated control-switch state; shape is diagnostic only",
        source_indices=segment.get("source_indices", []), fields=fields,
        late_secant_times_s=late_times.tolist(),
        late_secant_source_indices=tail["source_indices"] if tail is not None else [])
