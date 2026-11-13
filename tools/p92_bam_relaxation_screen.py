"""Frozen RF first-hold screen; an effective response model, not a P92 law.

Fit only 5--60 s of cycle 1; predict the remaining tensile hold to 182 s.
The initial 5 s exclude the prescribed 2 s ramp and observed settling.
This is within-path holdout, not independent-specimen validation or life fitting.
"""

import numpy as np
import argparse
import json
from scipy.optimize import least_squares


def relaxation(t, initial_stress, initial_rate, exponent):
    """Positive stress, ds/dt=-rate*(s/s0)**n; stress and rate in MPa, MPa/s."""
    t = np.asarray(t, dtype=float)
    if (not np.isfinite(t).all() or np.any(t < 0)
            or not np.isfinite([initial_stress, initial_rate, exponent]).all()
            or initial_stress <= 0 or initial_rate <= 0 or exponent < 1):
        raise ValueError("Require finite nonnegative time, positive stress/rate and n >= 1")
    x = initial_rate*t/initial_stress
    decay = x if exponent == 1 else np.log1p((exponent-1)*x)/(exponent-1)
    return initial_stress*np.exp(-decay)


def screen(data):
    if data["protocol"] != "RF":
        raise ValueError("This frozen first-hold split is RF only; other paths need explicit alignment")
    t = np.asarray(data["time_s"], dtype=float)
    s = np.asarray(data["stress_Pa"], dtype=float)/1e6
    cycles = np.asarray(data["cycle"], dtype=float)
    if (t.ndim != 1 or s.shape != t.shape or cycles.shape != t.shape
            or not np.isfinite(t).all() or not np.isfinite(s).all()
            or not np.isfinite(cycles).all() or np.any(np.diff(t) < 0)
            or np.any(cycles < 1) or np.any(cycles != np.floor(cycles))):
        raise ValueError("Require aligned finite histories, nondecreasing time and positive integer cycles")
    selected = (cycles == 1) & (t >= 5) & (t <= 182)
    t, s = t[selected], s[selected]
    if len(t) < 10 or not np.isfinite(s).all() or np.any(s <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("Require a resolved, ordered positive tensile hold")
    train = t <= 60
    if np.count_nonzero(train) < 5 or np.count_nonzero(~train) < 5:
        raise ValueError("Insufficient calibration or held-out samples")
    tt, ss = t[train]-t[0], s[train]
    # Time quadrature weighting, not inverse-noise weights or repeated specimens.
    w = np.r_[np.diff(tt)[0]/2, (tt[2:]-tt[:-2])/2, np.diff(tt)[-1]/2]
    w = np.sqrt(w/w.sum())
    fit = least_squares(lambda z: w*(relaxation(tt, ss[0], np.exp(z[0]), z[1])-ss),
                        [np.log(1.), 5.], bounds=([np.log(1e-6), 1.], [np.log(1000.), 30.]),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not fit.success:
        raise RuntimeError(f"Relaxation fit failed: {fit.message}")
    predicted = relaxation(t-t[0], ss[0], np.exp(fit.x[0]), fit.x[1])
    reports = {}
    for name, mask in (("calibration", train), ("within_path_holdout", ~train)):
        error = predicted[mask]-s[mask]
        reports[name] = {"samples": int(mask.sum()), "max_abs_error_MPa": float(abs(error).max()),
                         "rmse_MPa": float(np.sqrt(np.mean(error**2))),
                         "start_s": float(t[mask][0]), "end_s": float(t[mask][-1])}
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    later = []
    for cycle in (10, 100, 1000):
        mask = cycles == cycle
        if not mask.any():
            later.append({"cycle": cycle, "available": False})
            continue
        ct = np.asarray(data["time_s"])[mask]
        ct = ct-ct[0]
        cs = np.asarray(data["stress_Pa"])[mask]/1e6
        hold = (ct >= 5) & (ct <= 182)
        ct, cs = ct[hold], cs[hold]
        # Printed timestamps can repeat. These are evaluation samples, not
        # integration intervals or independent statistical observations.
        if len(ct) < 5 or np.any(cs <= 0) or np.any(np.diff(ct) < 0):
            raise ValueError("Later RF hold is unresolved or nonpositive")
        # Same coefficient B in ds/dt=-B*s**n; only measured initial stress changes.
        rate = np.exp(fit.x[0])*(cs[0]/ss[0])**fit.x[1]
        prediction = relaxation(ct-ct[0], cs[0], rate, fit.x[1])
        later.append({"cycle": cycle, "available": True,
                      "initial_stress_MPa": float(cs[0]),
                      "rmse_MPa": float(np.sqrt(np.mean((prediction-cs)**2))),
                      "max_abs_error_MPa": float(np.max(abs(prediction-cs)))})
    return {"schema": "agentfem.research.bam-rf-relaxation.v1", "source_sha256": data["sha256"],
            "initial_stress_MPa": float(ss[0]), "initial_rate_MPa_per_s": float(np.exp(fit.x[0])),
            "exponent": float(fit.x[1]), "reports": reports,
            "active_parameter_bounds": fit.active_mask.tolist(),
            "jacobian_singular_values": singular.tolist(),
            "later_conditional_holds": later,
            "later_hold_alignment": "5--182 s from first recorded sample in each selected cycle; measured initial stress, no refitting",
            "engineering_qualified": False,
            "scope": "First RF tensile hold only; no cyclic softening, multiaxial, temperature or life prediction",
            "identifiability": "Effective stress-relaxation rate; E and creep coefficient are not separately inferred",
            "uncertainty": "No probability interval inferred from correlated within-specimen samples"}


if __name__ == "__main__":
    from tools.p92_bam_source import read_source
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Checksum-verified Data_RF.lis")
    args = parser.parse_args()
    print(json.dumps(screen(read_source(args.path, protocol="RF")), indent=2, allow_nan=False))
