"""Prospective BAM development gates, separate from fitting and life admission.

Limits are project screens, not ASTM-prescribed tolerances. Recorded resolution
only rejects unresolved secants; it is not an experimental confidence interval.
"""

import numpy as np


POLICY = dict(identity="p92-bam-response-development-20261121",
    stress_absolute_MPa=5., stress_relative=.05,
    strain_absolute=2e-5, strain_excursion_relative=.10,
    secant_relative=.20, minimum_resolution_units=5.,
    stress_resolution_MPa=.01, strain_resolution=1e-5)


def screen(prediction, comparison, *, requested_cycles):
    if type(requested_cycles) is not int or requested_cycles < 1:
        raise ValueError("Require positive requested cycle count")
    checks = []

    def add(name, error, limit, **context):
        finite = np.isfinite([error, limit]).all() and limit >= 0
        checks.append(dict(name=name, error=float(error), limit=float(limit),
            status="passed" if finite and error <= limit else "failed", **context))

    completed = [r["cycle"] for r in prediction["cycles"]]
    checks.append(dict(name="complete_continuous_path", status="passed" if
        completed == list(range(1, requested_cycles+1))
        and not prediction["reversed_post_hold_ramps"]
        and not prediction.get("incompatible_continuation", False) else "failed"))
    expected_holds = {"RF": 2, "SLR": 3, "SLC": 3}[prediction["protocol"]]
    coverage = ([r["cycle"] for r in comparison["records"]] == completed
                and all(sum(h["cycle"] == c for h in prediction["hold_diagnostics"]) == expected_holds
                        for c in completed))
    checks.append(dict(name="observation_coverage", status="passed" if coverage else "unresolved"))
    for row in comparison["records"]:
        for kind in ("peak", "valley"):
            add(f"{kind}_stress", abs(row[f"{kind}_error_MPa"]),
                max(POLICY["stress_absolute_MPa"], POLICY["stress_relative"]*abs(row[f"observed_{kind}_MPa"])),
                cycle=row["cycle"], unit="MPa")
    for hold in prediction["hold_diagnostics"]:
        context = dict(cycle=hold["cycle"], segment=hold["segment"])
        for name in ("stress_MPa", "strain") if hold["stress_control"] else ("stress_MPa",):
            f = hold["fields"][name]
            limit = (max(POLICY["stress_absolute_MPa"], POLICY["stress_relative"]*f["observed_max_abs"])
                     if name == "stress_MPa" else
                     max(POLICY["strain_absolute"], POLICY["strain_excursion_relative"]*f["observed_excursion"]))
            add(f"hold_{name}_rms", f["rms"], limit, **context)
            add(f"hold_{name}_last", abs(f["last_predicted"]-f["last_observed"]), limit, **context)
        # Check only the response channel; a prescribed constant has no
        # identifiable relative slope and must not count as an easy pass.
        name = "strain" if hold["stress_control"] else "stress_MPa"
        f = hold["fields"][name]
        resolution = POLICY["strain_resolution" if name == "strain" else "stress_resolution_MPa"]
        times = np.asarray(hold["sample_times_s"])
        windows = dict(early=times[times <= min(10., .1*hold["duration_s"])],
                       late=np.asarray(hold["late_secant_times_s"]))
        for window, t in windows.items():
            observed = f.get(f"{window}_secant_observed_per_s")
            predicted = f.get(f"{window}_secant_predicted_per_s")
            units = abs(observed*(t[-1]-t[0]))/resolution if observed is not None and len(t) >= 2 else 0.
            if units < POLICY["minimum_resolution_units"] or predicted is None:
                checks.append(dict(name=f"{window}_{name}_secant", status="unresolved",
                    resolution_units=float(units), **context))
            else:
                add(f"{window}_{name}_secant", abs(predicted-observed)/abs(observed),
                    POLICY["secant_relative"], resolution_units=float(units), **context)
    status = ("failed" if any(c["status"] == "failed" for c in checks) else
              "unresolved" if any(c["status"] == "unresolved" for c in checks) else "passed")
    return dict(policy=dict(POLICY), status=status, checks=checks,
                engineering_qualified=False, life_qualified=False,
                scope="Project development screen; not independent validation or standards certification")
