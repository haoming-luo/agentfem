# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Raw, unit-explicit cross-solver transient comparisons."""

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.integrate import trapezoid


@dataclass(frozen=True)
class VelocityTrace:
    time: np.ndarray
    velocity: np.ndarray
    time_unit: str
    velocity_unit: str
    metadata: dict

    def __post_init__(self):
        t = np.array(self.time, dtype=float, copy=True)
        v = np.array(self.velocity, dtype=float, copy=True)
        if (
            t.ndim != 1
            or len(t) < 3
            or v.shape != (len(t), 3)
            or not np.all(np.isfinite(t))
            or not np.all(np.isfinite(v))
            or np.any(np.diff(t) <= 0)
        ):
            raise ValueError(
                "AFM-DYNAMIC-TRACE: strictly increasing finite time and (N,3) velocity required."
            )
        if not self.time_unit or not self.velocity_unit:
            raise ValueError("AFM-DYNAMIC-TRACE: explicit units required.")
        t.setflags(write=False)
        v.setflags(write=False)
        object.__setattr__(self, "time", t)
        object.__setattr__(self, "velocity", v)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def speed(self):
        return np.linalg.norm(self.velocity, axis=1)

    @property
    def cumulative_max(self):
        return np.maximum.accumulate(self.speed)


def read_velocity_csv(
    path,
    *,
    time_column="time",
    velocity_columns=("V1", "V2", "V3"),
    time_unit,
    velocity_unit,
    metadata=None,
):
    data = np.genfromtxt(
        path, delimiter=",", names=True, deletechars="", encoding="utf-8"
    )
    if len(velocity_columns) != 3:
        raise ValueError("AFM-DYNAMIC-CSV: three velocity columns required.")
    from hashlib import sha256

    return VelocityTrace(
        np.atleast_1d(data[time_column]),
        np.column_stack([np.atleast_1d(data[n]) for n in velocity_columns]),
        time_unit,
        velocity_unit,
        {
            **(metadata or {}),
            "source": str(path),
            "sha256": sha256(Path(path).read_bytes()).hexdigest(),
        },
    )


def _ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def _features(trace, threshold):
    speed = trace.speed
    t = trace.time
    index = int(np.argmax(speed))
    hits = np.flatnonzero(speed >= threshold)
    duration = t[-1] - t[0]
    rms = np.sqrt(trapezoid(speed**2, t) / duration)
    spectral = None
    if np.allclose(np.diff(t), np.diff(t)[0], rtol=1e-8, atol=0):
        # Component spectrum avoids frequency doubling caused by the speed norm.
        amp = np.linalg.norm(np.fft.rfft(trace.velocity, axis=0), axis=1) * 2 / len(t)
        amp[0] *= 0.5
        if len(t) % 2 == 0:
            amp[-1] *= 0.5
        freq = np.fft.rfftfreq(len(t), d=t[1] - t[0])
        peak = int(np.argmax(amp[1:]) + 1)
        spectral = {
            "frequency": freq.tolist(),
            "vector_amplitude": amp.tolist(),
            "dominant_frequency": float(freq[peak]) if np.any(amp[1:] > 0) else None,
            "definition": "L2 norm of three component FFTs; no window, detrending or filtering; DC excluded from peak",
            "frequency_unit": f"1/{trace.time_unit}",
        }
    return {
        "peak_speed": float(speed[index]),
        "peak_time": float(t[index]),
        "first_arrival": None if not len(hits) else float(t[hits[0]]),
        "arrival_threshold": float(threshold),
        "rms_speed": float(rms),
        "spectrum": spectral,
    }


def compare_dynamic_traces(candidate, reference, *, arrival_threshold, align="exact"):
    """No implicit smoothing, amplitude scaling, time shifting or extrapolation.

    ``align='interpolate_candidate'`` explicitly samples the candidate onto
    reference times within coverage, retaining original data in report output.
    """
    if (
        candidate.time_unit != reference.time_unit
        or candidate.velocity_unit != reference.velocity_unit
    ):
        raise ValueError(
            "AFM-DYNAMIC-UNITS: convert units explicitly before comparison."
        )
    if not np.isfinite(arrival_threshold) or arrival_threshold <= 0:
        raise ValueError("AFM-DYNAMIC-ARRIVAL: positive absolute threshold required.")
    if align == "exact":
        if not np.array_equal(candidate.time, reference.time):
            raise ValueError(
                "AFM-DYNAMIC-TIME: unequal sample times; explicitly select interpolation if intended."
            )
        c = candidate
    elif align == "interpolate_candidate":
        if (
            reference.time[0] < candidate.time[0]
            or reference.time[-1] > candidate.time[-1]
        ):
            raise ValueError("AFM-DYNAMIC-TIME: extrapolation is not permitted.")
        c = VelocityTrace(
            reference.time,
            np.column_stack(
                [
                    np.interp(reference.time, candidate.time, candidate.velocity[:, j])
                    for j in range(3)
                ]
            ),
            candidate.time_unit,
            candidate.velocity_unit,
            candidate.metadata,
        )
    else:
        raise ValueError("AFM-DYNAMIC-TIME: unknown alignment.")
    a = _features(c, arrival_threshold)
    b = _features(reference, arrival_threshold)
    t = reference.time
    numerator = np.sqrt(
        trapezoid(np.sum((c.velocity - reference.velocity) ** 2, axis=1), t)
    )
    denominator = np.sqrt(trapezoid(reference.speed**2, t))
    corr = None
    if np.std(c.speed) > 0 and np.std(reference.speed) > 0:
        corr = float(np.corrcoef(c.speed, reference.speed)[0, 1])
    spectral_metrics = {}
    if a["spectrum"] is not None and b["spectrum"] is not None:
        sa, sb = a["spectrum"], b["spectrum"]
        fa, fb = sa["dominant_frequency"], sb["dominant_frequency"]
        spectral_metrics["dominant_frequency_relative_error"] = (
            None if fa is None or fb is None else _ratio(abs(fa - fb), fb)
        )
        va, vb = np.asarray(sa["vector_amplitude"]), np.asarray(sb["vector_amplitude"])
        spectral_metrics["normalized_amplitude_spectrum_L2"] = _ratio(
            np.linalg.norm(va - vb), np.linalg.norm(vb)
        )
    return {
        "schema": "agentfem.dynamic-comparison",
        "candidate": a,
        "reference": b,
        "metrics": {
            **spectral_metrics,
            "peak_speed_ratio": _ratio(a["peak_speed"], b["peak_speed"]),
            "rms_speed_ratio": _ratio(a["rms_speed"], b["rms_speed"]),
            "normalized_vector_L2": _ratio(numerator, denominator),
            "peak_speed_relative_error": _ratio(
                abs(a["peak_speed"] - b["peak_speed"]), b["peak_speed"]
            ),
            "speed_correlation": corr,
            "arrival_time_error": None
            if a["first_arrival"] is None or b["first_arrival"] is None
            else a["first_arrival"] - b["first_arrival"],
            "peak_time_error": a["peak_time"] - b["peak_time"],
        },
        "alignment": align,
        "filtering": "none",
        "load_or_result_scaling": "none",
        "candidate_metadata": candidate.metadata,
        "reference_metadata": reference.metadata,
    }


def write_dynamic_comparison(
    candidate, reference, directory, *, arrival_threshold, align="exact"
):
    """Write raw traces, numerical JSON, concise Markdown and unsmoothed plots."""
    report = compare_dynamic_traces(
        candidate, reference, arrival_threshold=arrival_threshold, align=align
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "comparison.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, trace in [("candidate", candidate), ("reference", reference)]:
        np.savetxt(
            directory / f"{name}_raw.csv",
            np.column_stack(
                (trace.time, trace.velocity, trace.speed, trace.cumulative_max)
            ),
            delimiter=",",
            header="time,V1,V2,V3,speed,cumulative_max_speed",
            comments="",
        )
    lines = [
        "# Dynamic cross-solver comparison",
        "",
        "Raw signals; no smoothing, amplitude scaling or time shifting.",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [
        f"| {k} | {v if v is not None else 'undefined'} |"
        for k, v in report["metrics"].items()
    ]
    lines += [
        "",
        f"Time unit: {candidate.time_unit}; velocity unit: {candidate.velocity_unit}.",
        "Model, mesh, node, force and energy records are retained in comparison.json metadata.",
        "These metrics do not establish element-formulation equivalence.",
    ]
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), layout="constrained")
    for label, trace in [("AgentFEM", candidate), ("Reference", reference)]:
        axes[0].plot(trace.time, trace.speed, label=label)
        axes[1].plot(trace.time, trace.cumulative_max, label=label)
    for ax, title in zip(axes, ["Velocity magnitude", "Cumulative maximum velocity"]):
        ax.set_title(title)
        ax.set_xlabel(f"Time [{candidate.time_unit}]")
        ax.set_ylabel(candidate.velocity_unit)
        ax.legend()
    for label, features in [
        ("Candidate", report["candidate"]),
        ("Reference", report["reference"]),
    ]:
        spectrum = features["spectrum"]
        if spectrum is not None:
            axes[2].plot(
                spectrum["frequency"], spectrum["vector_amplitude"], label=label
            )
    axes[2].set_title("Raw component-vector amplitude spectrum")
    axes[2].set_xlabel(f"Frequency [1/{candidate.time_unit}]")
    axes[2].set_ylabel(candidate.velocity_unit)
    if axes[2].lines:
        axes[2].legend()
    fig.savefig(directory / "velocity_comparison.png", dpi=180)
    plt.close(fig)
    return path
