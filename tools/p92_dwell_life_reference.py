"""Isothermal reproduction of Wang et al. 2019, not an engineering material card.

Uses the 600 C Table 2 MCB curve and equations 10/11 with Table 5 parameters.
No temperature polynomial: printed equation 8 has two quadratic terms and
cannot be silently corrected. Same-paper observations are calibration checks,
not independent validation. This empirical law is not a stiffness damage law.
"""

from math import exp, isfinite, log
from numbers import Real

import numpy as np

from agentfem.constitutive import CoffinMansonBasquinCurve

SOURCE = "https://doi.org/10.3390/met9020183"


def hold_factor(minutes, *, alpha, beta, g, h, temperature_C, amplitude):
    """Equations 10/11, with duration in minutes and strain as a fraction."""
    values = (minutes, alpha, beta, g, h, temperature_C, amplitude)
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
            or not isfinite(v) for v in values)
            or minutes < 0 or not 0 < alpha < 1 or not 0 < beta < 1
            or g < 0 or h < 0 or not 20 <= temperature_C < 1530
            or not 0 < amplitude < 1):
        raise ValueError("Invalid dwell-law controls; use minutes and fractional strain")
    if minutes == 0:
        return 1.
    # log of the nonnegative exponent in exp(-exponent). This avoids overflow
    # for long holds without changing the asymptotic lower bound 1-alpha.
    log_exponent = log(-log(beta)) + log(minutes) + g*(temperature_C-20)/1510 - h*amplitude
    decay = 0. if log_exponent > log(745.) else exp(-exp(log_exponent))
    return 1-alpha+alpha*decay


def reference_curve():
    return CoffinMansonBasquinCurve.from_strain_domain(
        young=134509., fatigue_strength_coefficient=397.,
        fatigue_strength_exponent=-.068, fatigue_ductility_coefficient=.341,
        fatigue_ductility_exponent=-.61, calibrated_strain_amplitudes=(.002, .008),
        stress_unit="MPa", source=SOURCE + " Table 2, 600 C",
    )


def predict_600(*, amplitude, tensile_minutes=0., compressive_minutes=0.):
    """Bounded 600 C reproduction; refuse untested combined holds/CF amplitudes."""
    base = reference_curve().cycles_to_failure(amplitude)
    if tensile_minutes and compressive_minutes:
        raise ValueError("Combined holds are not admitted by this P92 reproduction")
    if (tensile_minutes or compressive_minutes) and amplitude != .004:
        raise ValueError("P92 600 C dwell observations cover amplitude 0.004 only")
    if not 0 <= tensile_minutes <= 60 or not 0 <= compressive_minutes <= 10:
        raise ValueError("Hold duration outside the reproduced P92 observation domain")
    common = dict(temperature_C=600., amplitude=amplitude)
    tensile = hold_factor(tensile_minutes, alpha=.6, beta=.68, g=9.2, h=1185., **common)
    compression = hold_factor(compressive_minutes, alpha=.73, beta=.12, g=19.3, h=1893., **common)
    return dict(cycles=base*tensile*compression, base_cycles=base,
                tensile_factor=tensile, compressive_factor=compression)


def screen():
    rows = [("T01", 1., 0., 1303), ("T03", 3., 0., 1190),
            ("T10", 10., 0., 675), ("T60", 60., 0., 748),
            ("C01", 0., 1., 652), ("C03", 0., 3., 416), ("C10", 0., 10., 567)]
    records = []
    for suffix, tensile, compressive, observed in rows:
        result = predict_600(amplitude=.004, tensile_minutes=tensile,
                             compressive_minutes=compressive)
        records.append(dict(specimen="CF600S04"+suffix, observed_cycles=observed,
                            tensile_minutes=tensile, compressive_minutes=compressive,
                            prediction_ratio=result["cycles"]/observed, **result))
    return dict(schema="agentfem.research.p92-isothermal-dwell-life.v1", source=SOURCE,
                temperature_C=600, strain_amplitude=.004, strain_rate_per_s=.001,
                records=records, validation_role="same-paper parameter reproduction",
                independent_validation=False, engineering_qualified=False,
                failure_definition="Not explicitly resolved in source experimental description",
                scope="Table 2 at 600 C plus equations 10/11; not the temperature-polynomial model",
                limitations=["Table 5 also fitted P91 data; not a universal P92 law",
                             "No stress-controlled dwell or evolving structural damage",
                             "No reliability interval inferred from seven different hold conditions"])


if __name__ == "__main__":
    import json
    print(json.dumps(screen(), indent=2, allow_nan=False))
