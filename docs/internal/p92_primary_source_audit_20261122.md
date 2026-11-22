# Verify the source equation before adopting its kinetics

The [author manuscript](https://strathprints.strath.ac.uk/60381/1/Khayatzadeh_etal_EFM2017_Creep_deformation_and_stress_relaxation_of_a_martensitic_P92.pdf)
for DOI 10.1016/j.engfracmech.2017.02.008 was retrieved and pages 9--10
visually checked. It concerns another P92 heat at 650 C, not BAM's cyclic
620 C tests. Table 4 distinguishes three stress regimes. Its hourly
coefficients are audited against Table 2 minimum rates; the nomenclature's
time label must not be copied into code without unit conversion.

## Algebraic discriminator, not a corrected published model

For a single primary term, write `epsilon_p=C*t^m`, where `C=Ap*sigma^np`.
Direct differentiation and elimination of positive time give

`epsilon_p_dot = m*C^(1/m)*epsilon_p^((m-1)/m)`.

The printed Eq. 6 instead places `m` inside the powered strain bracket.
Even in the primary-only limit its literal rate differs by `m^(-1/m)`:
approximately 94.28, 177.87 and 20.08 for the tabulated exponents.
This inconsistency is present in the page image, not only extracted text.
It does not establish which formula was used in the authors' FE calculations.
Do not silently "correct" their model or claim to reproduce those calculations.
Adding secondary creep also requires distinguishing primary strain from total
creep strain; substitution of one for the other is not algebraic equivalence.

## Existing software integration checked

The audit uses two existing `PowerLawCreep` objects, not a new solver. With
MPa and seconds, primary coefficient `Ap*m/3600`, reference time `3600`,
time exponent `m-1`, and secondary coefficient `As/3600` reproduce the
constant-stress sum in Eq. 4. Numerical differentiation independently checks
the primary rate. Whole-interval and split increments agree without resetting
time; the initial singular rate has a finite integrated strain.

All six stress cases pass this numerical/unit comparison. The tabulated
secondary law differs from the measured minimum rates by roughly -16.1%
to +4.3%; these are reported discrepancies, not a newly relaxed tolerance.
The one-sided rate limits at the two regime boundaries differ by factors
1.061 and 1.334. No smoothing, boundary convention or cyclic extension is
invented to conceal those differences.

This evidence qualifies neither BAM response nor engineering life. Its useful
lesson for the pending mechanism decision is to require a known constant-load
limit, explicit state meaning and unit-consistent kinetics before identification.
The previous irreversible replacement was our declared logarithmic hypothesis,
not this paper's equation; its failed results remain unchanged.

Reproduce with the checksum-verified public manuscript:

```sh
python -m tools.p92_primary_source_audit /path/to/manuscript.pdf /new/path/audit.json
```

The small archived report is `evidence/creep/2026-11-22-primary-source-audit.json`.
The manuscript is not redistributed. No fit, production change, gate change,
merge or release follows from this audit.
