# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Separate recorded-loading and material-response errors without refitting."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.p92_bam_source import read_source, replay_window
from tools.p92_bam_cyclic_screen import replay
from tools.p92_nominal_cycles import nominal_cycles
from tools.p92_published_recovery_reference import Parameters


def compare_loading(data, calibration, *, cycles=10, subdivisions=2):
    """Same fixed parameters and starting state; never reset hidden history.

    This comparison uses already inspected observations and is diagnostic,
    not a new independent material validation. Nominal and recorded loading
    use their own clocks. Only cycle extrema, not resampled stresses, are paired.
    """
    if data["protocol"] != "RF" or data["sha256"] != calibration["source_sha256"]:
        raise ValueError("Require the RF source used by this calibration record.")
    if not calibration["fit"]["success"]:
        raise ValueError("Require a converged calibration record.")
    window = replay_window(data, first_cycle=1, last_cycle=cycles, max_interval_s=2.)
    history = window["histories"]
    time, strain = history["strain"].abscissa, history["strain"].values
    observed = history["stress_Pa"].values / 1e6
    selected = (data["cycle"] >= 1) & (data["cycle"] <= cycles)
    cycle_ids = data["cycle"][selected]
    p = Parameters(**calibration["parameters"])
    initial = np.array(calibration["initial_state"], dtype=float)
    recorded = replay(time, strain, p, initial, method="Radau")
    refined = replay(time, strain, p, initial, method="BE", subdivisions=subdivisions)
    nominal = nominal_cycles(p, initial, cycles=cycles, method="Radau")
    if nominal["failure"] is not None:
        raise RuntimeError("Nominal comparison failed; no complete loading discriminator.")
    records = []
    for item in nominal["cycles"]:
        cycle = item["cycle"]
        mask = cycle_ids == cycle
        actual_peak = float(observed[mask].max())
        replay_peak = float(recorded[mask, 0].max())
        nominal_peak = item["sampled_peak_MPa"]
        records.append({
            "cycle": cycle,
            "observed_peak_MPa": actual_peak,
            "recorded_loading_peak_MPa": replay_peak,
            "nominal_loading_peak_MPa": nominal_peak,
            "recorded_response_peak_error_MPa": replay_peak - actual_peak,
            "nominal_response_peak_error_MPa": nominal_peak - actual_peak,
            "loading_choice_peak_difference_MPa": nominal_peak - replay_peak,
            "BE_Radau_max_difference_MPa": float(np.max(np.abs(refined[mask, 0] - recorded[mask, 0]))),
        })
    available = np.unique(data["cycle"])
    missing_count = int(available[-1] - available[0] + 1 - len(available))
    return {
        "schema": "agentfem.research.p92-loading-discriminator.v1",
        "source_sha256": data["sha256"],
        "calibration_sha256": sha256(json.dumps(calibration, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "cycles": records,
        "BE_subdivisions": subdivisions,
        "missing_recorded_cycles_in_source_span": missing_count,
        "scope": "Recorded vs nominal loading on previously inspected RF data; no fit or state reset",
        "extrema_scope": "Sampled extrema, not certified continuous-time extrema",
        "engineering_qualified": False,
        "life_qualified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--subdivisions", type=int, default=2)
    args = parser.parse_args()
    report = compare_loading(read_source(args.source, protocol="RF"),
                             json.loads(args.calibration.read_text()), cycles=args.cycles,
                             subdivisions=args.subdivisions)
    print(json.dumps(report, indent=2, allow_nan=False))
