# Specimen life evidence: first-slice acceptance

## Design and scope

Keep raw experimental outcomes separate from FE material state, damage fractions
and design-code allowables. The public `assessments` namespace exposes immutable
`LifeObservation`, `LifeSurvival` and `life_survival`. The estimator delegates to
SciPy rather than adding a statistics engine. Grouping is explicit and exact;
conditions are metadata supplied by the experiment owner, not inferred by AI.

Implemented: failure/right-censor records, provenance, risk/failure/censor
counts, Kaplan–Meier survival, pointwise log-log Greenwood intervals, observed
quantile crossings, JSON-safe unavailable bounds and no extrapolation.

Deferred: left/interval censoring, parametric fitting, accelerated-life and
batch models, prediction intervals, tolerance limits, FE uncertainty propagation
and automatic material qualification. These should follow an actual compatible
dataset and a held-out validation question, not precede them.

## Numerical and data acceptance

`tests/test_life_observations.py` checks product-limit survival and exponential
Greenwood limits against independently coded formulas using Python's standard
normal distribution quantile. It covers ties, censoring before failure,
all-censored and single-specimen inputs, unit invariance, ordering, duplicate
specimens, mixed conditions, immutable metadata, strict JSON, and the executable
public documentation example.

`tools/p92_life_observations.py` uses the previously reviewed Hoey DB source
transcription. Its tests retain all 12 creep and 6 fatigue/dwell rows, the life
units, waveform semantics and original source identity. There are **18 distinct
conditions and no reported right-censored records** in these tables. Each group
has one record, so the result does not claim a calibrated scatter distribution.
Row IDs identify published table entries, not unavailable specimen serials.

The production censoring path is tested with synthetic data, not fictitious
censored P92 observations. P92 provides a source-admission/grouping check only.

## Local evidence

Environment: existing `fenicsx-env`, Python 3.11, single-process targeted tests.
No FE solve, new installation or full simulation was needed.

```sh
PYTHONPATH=src python -m pytest -q -W error::RuntimeWarning \
  tests/test_life_observations.py tests/test_p92_life_observations.py \
  tests/test_assessments.py tests/test_tabulated_rupture_curve.py \
  tests/test_creep_damage_history.py tests/test_p92_db_cohort_screen.py \
  tests/test_documentation.py
PYTHONPATH=src python -m tools.p92_life_observations --output /tmp/p92-life-observations.json
PYTHONPATH=src python build_knowledge.py --check --check-imports
python build_docs.py --check
```

Initial integrated acceptance: **192 tests passed**, 58 scientific cards
validated, generated API documentation current. This is bounded local
acceptance, not an assertion of remote CI, release, Windows testing or P92
engineering-life qualification.

## Sources and next gate

The method and censoring boundary follow
[SciPy ECDF](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ecdf.html).
[NIST's likelihood discussion](https://www.itl.nist.gov/div898/handbook/apr/section4/apr412.htm)
explains why few failures do not justify unqualified parametric estimates.
The P92 source and checked transcription are linked in `docs/p92_validation.md`.

Next admit compatible replicate/run-out records. Fit at most one motivated
conditional-life model, retain all censoring and batch identities, and evaluate
held-out predictions before reporting P92 statistical life bounds.

## Mainline integration, 2026-11-12

Integrated main `156db1f0` without changing estimators, source observations or
acceptance tolerances. Kept the concise agent entry and moved the life-assessment
guidance into its scientific reference. The combined targeted suite, including
fixed-horizon tests and current documentation tests, passed **214 tests** in
`fenicsx-env` with RuntimeWarnings treated as errors. Generated documentation
and focused lint passed. Remote CI remains the merge gate.
