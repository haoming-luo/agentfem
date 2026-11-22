"""Audit a published 650 C forward-creep relation before using its kinetics.

This is a separate heat/temperature from BAM, not a replacement 620 C card.
No fitting, cyclic extrapolation, or material promotion is performed.
"""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from agentfem.constitutive.creep import PowerLawCreep
from tools.p92_measured_fit import write_new

SOURCE = "https://doi.org/10.1016/j.engfracmech.2017.02.008"
SOURCE_PDF_SHA256 = "2efe69578bcbc5c40acbf79a3b9a6755c1080f14e5dcf69a3a42b6be9bc0ad57"
# Table 4, author manuscript p. 10: As, ns, Ap, np, mp; stress MPa, time h.
REGIMES = ((1.50e-12, 3.33, 8.20e-10, 3.05, .28),
           (2.73e-17, 5.8, 8.20e-15, 5.60, .26),
           (1.00e-52, 22.37, 1.49e-49, 20.94, .35))
# Table 2 minimum rates, not a new calibration dataset or rupture criterion.
OBSERVATIONS = ((40., 3.41e-7, 0), (60., 1.20e-6, 0), (80., 3.48e-6, 0),
                (100., 1.27e-5, 1), (120., 3.73e-5, 1), (170., 8.00e-3, 2))


def components(regime, *, seconds=True):
    if type(regime) is not int or not 0 <= regime < len(REGIMES) or type(seconds) is not bool:
        raise ValueError("Require a declared source regime and time convention")
    a_s, n_s, a_p, n_p, m = REGIMES[regime]
    unit = 3600. if seconds else 1.
    # A = Ap*m/3600 and t_ref = 3600 reproduce Ap*sigma^np*(t/3600)^m.
    return (PowerLawCreep(a_p*m/unit, n_p, m-1., reference_time=unit),
            PowerLawCreep(a_s/unit, n_s))


def primary_rates(regime, stress, time_h):
    """Independent derivative/elimination of Eq. 2 versus literal printed Eq. 6."""
    if (type(regime) is not int or not 0 <= regime < len(REGIMES)
            or not np.isfinite([stress, time_h]).all() or stress <= 0 or time_h <= 0):
        raise ValueError("Require positive stress/time inside a declared monotonic comparison")
    _, _, a, n, m = REGIMES[regime]
    c = a*stress**n
    primary_strain = c*time_h**m
    derivative = m*c*time_h**(m-1.)
    eliminated = m*c**(1/m)*primary_strain**((m-1.)/m)
    literal = c**(1/m)*(m*primary_strain)**((m-1.)/m)
    return dict(primary_strain=primary_strain, differentiated_rate_per_h=derivative,
                eliminated_rate_per_h=eliminated, literal_printed_rate_per_h=literal,
                literal_to_derivative_ratio=literal/derivative,
                analytic_ratio=m**(-1/m))


def audit():
    rows = []
    for stress, observed, regime in OBSERVATIONS:
        primary, secondary = components(regime)
        hourly = components(regime, seconds=False)
        _, _, a_p, n_p, m = REGIMES[regime]
        a_s, n_s, *_ = REGIMES[regime]
        times_h = np.array([0., .001, .1, 1., 10., 100.])
        expected = a_p*stress**n_p*times_h**m+a_s*stress**n_s*times_h
        predicted = np.array([sum(c.constant_stress_strain(stress, t*3600.)
                                  for c in (primary, secondary)) for t in times_h])
        in_hours = np.array([sum(c.constant_stress_strain(stress, t) for c in hourly)
                             for t in times_h])
        split = sum(c.constant_stress_increment(stress, a*3600., b*3600.)
                    for a, b in zip(times_h[:-1], times_h[1:], strict=True)
                    for c in (primary, secondary))
        rate_h = secondary.equivalent_rate(stress, 1.)*3600.
        equation_check = primary_rates(regime, stress, 1.)
        passed = (np.allclose(predicted, expected, rtol=1e-12, atol=1e-15)
                  and np.allclose(predicted, in_hours, rtol=1e-12, atol=1e-15)
                  and abs(split-expected[-1]) <= 1e-12*max(expected[-1], 1e-3)
                  and np.isclose(equation_check["differentiated_rate_per_h"],
                                 equation_check["eliminated_rate_per_h"], rtol=1e-12, atol=0))
        rows.append(dict(stress_MPa=stress, regime=regime, times_h=times_h.tolist(),
            predicted_strain=predicted.tolist(), equation4_strain=expected.tolist(),
            observed_minimum_rate_per_h=observed, equation7_rate_per_h=rate_h,
            minimum_rate_relative_error=(rate_h-observed)/observed,
            primary_equation_audit=equation_check, numerical_passed=bool(passed),
            split_final_strain=float(split), coefficient_time_convention="hours converted explicitly to seconds"))
    boundaries = []
    for i, stress in enumerate((85., 140.)):
        left = REGIMES[i][0]*stress**REGIMES[i][1]
        right = REGIMES[i+1][0]*stress**REGIMES[i+1][1]
        boundaries.append(dict(stress_MPa=stress, left_rate_per_h=left, right_rate_per_h=right,
                               ratio_right_to_left=right/left))
    return dict(schema="agentfem.research.primary-source-audit.v1", source=SOURCE,
        temperature_C=650, records=rows, regime_boundary_limits=boundaries,
        fitted=False, engineering_qualified=False, life_qualified=False,
        bam_validation=False, printed_equation6_admitted=False,
        scope="Source equation/unit audit using existing core integration; no cyclic or changing-stress prediction")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source_digest = sha256(args.source_pdf.read_bytes()).hexdigest()
    if source_digest != SOURCE_PDF_SHA256:
        raise ValueError("Source manuscript identity changed; review before comparing equations")
    result = audit()
    result["source_pdf_sha256"] = source_digest
    result["driver_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    write_new(args.output, result)
