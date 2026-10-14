# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Optional exact arithmetic implementation, selected once at module import.

No geometry algorithm, tolerance or precision changes. Standard-library
Fraction remains available without the optional gmpy2 distribution. This is
not a runtime setter and never rewrites other modules' functions.
"""
from fractions import Fraction


def _select():
    try:
        import gmpy2
    except ImportError:
        return Fraction, {"library": "fractions.Fraction", "arithmetic": "exact_rational"}
    # Guard binary64 conversion semantics, including subnormal/extreme values.
    for value in (0., .1, -.1, 1e-300, 5e-324, 1e300, float.fromhex('0x1.fffffffffffffp+1023')):
        a, b = Fraction(value), gmpy2.mpq(value)
        if (a.numerator != b.numerator or a.denominator != b.denominator):
            raise RuntimeError("Optional GMP conversion does not preserve binary64 exactly")
    return gmpy2.mpq, {"library": "gmpy2", "version": gmpy2.version(),
                       "gmp_version": gmpy2.mp_version(), "arithmetic": "exact_rational"}


Rational, _INFO = _select()


def exact_geometry_arithmetic():
    """Execution provenance; not a geometric validity or speed certificate."""
    return dict(_INFO)
