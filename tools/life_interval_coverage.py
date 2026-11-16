"""Seeded sampling check for censored life intervals, not P92 scatter data."""

import argparse
from bisect import bisect_right
import json
import math

import numpy as np

from agentfem.assessments import LifeObservation, life_survival


def coverage_screen(*, sample_sizes=(5, 20, 100), trials=2000, seed=20261116):
    """Independent exponential failures/withdrawals at a known median horizon.

    Report interval availability separately from coverage conditional on an
    interval being available. Missing intervals never count as certainty.
    """
    sizes = tuple(sample_sizes)
    if (not sizes or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in sizes)
            or isinstance(trials, bool) or not isinstance(trials, int) or trials < 1
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("Require positive integer sizes/trials and nonnegative integer seed")
    rng = np.random.default_rng(seed)
    horizon = math.log(2.)
    reports = []
    for size in sizes:
        finite = covered = outside = undefined = 0
        for _ in range(trials):
            failure = rng.exponential(1., size)
            withdrawal = rng.exponential(2., size)
            rows = [LifeObservation(specimen_id=str(i), life=float(min(f, c)),
                    status="failure" if f <= c else "right_censored", life_unit="h",
                    population="synthetic exponential cohort", batch_id="synthetic",
                    failure_definition="generated event", conditions={"mean_failure_h": 1.},
                    source="Seeded independent exponential failure and withdrawal simulation")
                    for i, (f, c) in enumerate(zip(failure, withdrawal, strict=True))]
            result = life_survival(rows, independent_specimens=True,
                                   noninformative_censoring=True, confidence_level=.95)
            if horizon > result.times[-1]:
                outside += 1
                continue
            index = bisect_right(result.times, horizon)-1
            if index < 0 or result.lower[index] is None or result.upper[index] is None:
                undefined += 1
                continue
            finite += 1
            covered += int(result.lower[index] <= .5 <= result.upper[index])
        reports.append({"sample_size": size, "trials": trials, "finite_intervals": finite,
                        "outside_followup": outside, "undefined_boundary_intervals": undefined,
                        "covered": covered,
                        "conditional_coverage": covered/finite if finite else None,
                        "covered_fraction_of_all_trials": covered/trials})
    return {"schema": "agentfem.research.life-interval-coverage.v1", "seed": seed,
            "generator": "numpy.default_rng, sequential draws across sample sizes",
            "numpy_version": np.__version__, "confidence_level": .95,
            "failure_mean_h": 1., "independent_withdrawal_mean_h": 2.,
            "horizon_h": horizon, "true_survival": .5, "records": reports,
            "scope": "Sampling diagnostic for pointwise log-log Greenwood intervals; not an exact coverage guarantee or material validation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(coverage_screen(trials=args.trials), indent=2, allow_nan=False))
