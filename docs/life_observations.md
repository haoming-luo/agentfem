# Specimen life observations and uncertainty

Use `assessments.LifeObservation` for measured failure lives and test run-outs.
Use `assessments.life_survival` to describe one population, batch and test
condition without selecting a parametric life distribution. These tools consume
experimental evidence; they do not change the finite-element solver or material.

## A complete example

```python
from agentfem import assessments

records = [
    assessments.LifeObservation(
        specimen_id=f"coupon-{i}", life=t, status=status,
        life_unit="h", population="synthetic demonstration", batch_id="A",
        failure_definition="rupture", conditions={"stress_MPa": 100., "temperature_K": 923.15},
        source="synthetic fixture, not material calibration",
    )
    for i, (t, status) in enumerate([
        (1, "failure"), (2, "failure"), (3, "failure"),
        (4, "right_censored"), (4, "right_censored"),
    ])
]
result = assessments.life_survival(
    records, independent_specimens=True, noninformative_censoring=True,
    confidence_level=0.95,
)
assert result.quantile(0.5) == 3.0
assert result.quantile(0.9) is None  # not reached within the observed follow-up
evidence = result.as_dict()        # JSON-safe records, risk counts and intervals
```

The survival probabilities after times 1, 2, 3 and 4 are 0.8, 0.6, 0.4 and
0.4. A run-out means the specimen survived to the recorded coordinate, not
that it failed there. At a tied coordinate, censored specimens remain at risk
for failures occurring at that coordinate.

## Data contract

- One record identifies one independent specimen, not one cycle or one frame.
- Positive life coordinates use `cycles`, `reversals`, `s`, `min` or `h`.
  Units are never converted silently.
- `failure` and `right_censored` are supported. Left and interval censoring
  require different estimators and are rejected by this first implementation.
- Population, batch, failure definition, units and the complete `conditions`
  mapping must match exactly within one analysis. Group records explicitly
  using `cohort_key`; the input data can still contain many groups.
- Conditions have explicit unit-bearing keys, such as `temperature_K` and
  `stress_MPa`. Include strain ratio, waveform, dwell, environment and other
  relevant controls. Key names and completeness are supplied by the experiment
  owner; the software does not infer experimental equivalence.
- Source and specimen identity remain in the exported evidence. Condition
  mappings are copied and immutable. Repeated specimen IDs are rejected.

The two required flags declare assumptions, not checks the program can prove.
Non-informative censoring means the censoring mechanism does not preferentially
remove shorter- or longer-lived specimens within the declared group. Stopping
because damage is increasing may violate that assumption. A failed simulation
is not an experimental run-out.

## Meaning of the uncertainty output

The estimator is Kaplan–Meier, implemented by SciPy. Limits use asymptotic
exponential Greenwood (log-log) intervals. They describe **pointwise confidence
in survival probability**, not a prediction interval for a new specimen,
simultaneous confidence band, or a confidence-qualified engineering design life.
`quantile(p)` returns the first observed crossing of failure probability `p`;
it is a point estimate, or `None` when follow-up is insufficient.

At survival zero or one, or an undefined formula limit, bounds are `None`, not
zero-width certainty. All-censored and single-specimen groups remain readable
but do not establish scatter or reliable tail probabilities. Two observed
failures merely set `replicate_failures_available`; that flag is not a statistical
adequacy threshold. Queries beyond observed follow-up are rejected.

This slice does not fit Weibull/lognormal distributions, pool batches, model
stress/temperature covariates, or claim to propagate FE/model uncertainty.
Such models need an explicit calibration and held-out validation design.

## P92 source check

From a source checkout with AgentFEM available:

```sh
python -m tools.p92_life_observations --output /tmp/p92-life-observations.json
```

The admitted Hoey DB tables contain 12 creep and 6 fatigue/dwell observations,
all reported failures, at 18 distinct conditions. Source row IDs are used
because specimen serial numbers are unavailable; the source pipe is identified
without inventing a heat number. Each group has only one record. This verifies
source admission, unit retention and grouping, **not P92 scatter calibration**.
No run-outs or replicate measurements are fabricated. Numerical censoring and
confidence-limit verification instead uses independently computed synthetic
fixtures in `tests/test_life_observations.py`.

## References

- [SciPy ECDF and Kaplan–Meier implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ecdf.html)
- [NIST: censored likelihood and small-failure-sample limitations](https://www.itl.nist.gov/div898/handbook/apr/section4/apr412.htm)
- [Hoey P92 source thesis](https://repository.nottingham.ac.uk/handle/123456789/30506)

For the broader distinction between scatter, confidence, prediction and design
allowables, see [engineering life uncertainty](internal/engineering_life_uncertainty_20261107.md).
