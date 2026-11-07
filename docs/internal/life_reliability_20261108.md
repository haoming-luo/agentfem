# Fixed-horizon reliability and P92 data search

## Bounded implementation

Added `assessments.fixed_horizon_reliability` and its immutable result. Reuses
LifeObservation, grouping/identity validation and SciPy exact binomial intervals.
The scope is one prespecified horizon and fixed specimen count, independent
specimens with a common condition, and fully observed binary outcomes at that
horizon. Any early-censored record is rejected with its ID. No specimen is
discarded, and no fitted lifetime distribution or engineering qualification is
created. This complements KM's undefined Greenwood endpoints, not replaces KM.

For 20 independent synthetic specimens surviving a fixed horizon, the 95%
two-sided lower bound is (0.025)^(1/20), approximately 0.83157, not 1. The test
suite checks both extreme counts and mixed-count binomial tail equations with
independent finite sums, ties, units, metadata, early censoring, fixed-plan
declarations and the executable documentation example.

Local acceptance in `fenicsx-env`: 213 targeted tests passed (fixed-horizon,
survival records, P92 adapters, existing life assessments and documentation),
with RuntimeWarnings treated as errors. All 58 knowledge cards passed import
validation; generated documentation and lint checks passed. No remote CI,
release or P92 engineering-qualification claim is made by this local check.

References:

- [NIST exact binomial confidence intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
- [SciPy exact proportion intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)

## Public-data search outcome

The search did not establish a new admissible P92 same-condition replicate or
run-out cohort. This is not a claim that no such data exist.

- [NIMS CDS 48B announcement](https://www.nims.go.jp/eng/press/2018/04/201804050.html)
  identifies T92/P92 creep data. The [database](https://cds.nims.go.jp/en/) has
  access/use conditions and prohibits scraping and mass acquisition. No restricted
  data were retrieved and no NIMS values were invented or pooled across heats.
- [BAM P92 service-like dataset](https://zenodo.org/records/7198219) provides three
  different RF/SLR/SLC protocols at 620 C, not three interchangeable replicates.
  Recorded cycles are not independent specimens. This remains a potential
  response-history validation source, not newly admitted scatter data.
- [Hectors et al. open fatigue experiment publication](https://biblio.ugent.be/publication/01GWP4BGWQV27742D5J9EG0P9W)
  is a candidate for a separate generic fatigue-statistics benchmark. Its detailed
  data/protocol were not admitted in this increment, and it is not P92 validation.

Retain the already admitted DB P92 adapter and its 18 distinct condition groups.
No change to source values, material coefficients, FE state or life calibration.
Next gate remains a reviewed dataset, explicit specimen identities/conditions and
an appropriate held-out question before introducing parametric scatter models.
