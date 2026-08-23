# Scientific Trust and Verification

## Execution is not acceptance

AgentFEM keeps four claims separate:

| Level | Meaning |
| --- | --- |
| `computed` | A procedure produced finite result data. |
| `converged` | The declared algebraic/nonlinear/time procedure completed its numerical acceptance test. |
| `verified` | Every declared implementation, reference, discretization, or invariant claim passed. |
| `validated` | At least one passed claim compares the model with appropriate physical or experimental evidence. |

A completed solver call does not automatically advance a result beyond
`computed`. An empty verification report can establish `converged`, but it
cannot establish `verified`. Failed and inconclusive claims both prevent that
promotion; they remain distinct so an inapplicable reference theory is not
misreported as a numerical failure.

## Quality presets: the ordinary user path

Most users should not assemble every runtime claim manually. A solved result
can apply one of three stable policies:

| Preset | Minimum evidence | Intended use |
| --- | --- | --- |
| `exploratory` | computed result plus finite payload checks | model development and screening |
| `engineering` | explicit solver/time-procedure convergence plus runtime checks | parameter campaigns and engineering post-processing |
| `release` | engineering evidence plus a passed scientific reference claim | published demos and release contracts |

```python
result = step.solve_result()
result.add_quantity("tip_displacement", tip_u, unit="m")
result.verify(
    "engineering",
    required_quantities=("tip_displacement",),
).require()
```

The automatic checks cover execution status, registered payload, finite live
field coefficients, required quantities/histories/artifacts, materialized
artifact paths, and structured execution-trace completeness when available.
They are recorded as `kind="runtime"`. Passing them can establish an accepted
engineering workflow, but cannot by itself promote a result to `verified`.

For a release Golden:

```python
golden = benchmarks.golden_benchmark("agentfem.benchmark.example")
result.verify(
    "release",
    claims=golden.claims(observables),
    required_artifacts=("fields",),
).require()
```

`GoldenBenchmark.claims(...)` retains the benchmark identifier, reference
version, tolerance, unit, expected value, and validity statement in the result
manifest. A Golden remains regression evidence, not experimental validation.

## Capability evidence audit

The constitutive catalog and benchmark registry are joined by a machine-
readable audit:

```python
from agentfem import benchmarks

for item in benchmarks.audit_capability_evidence():
    print(item.capability, item.maturity, item.gaps)
```

The same records appear under `constitutive_evidence` in
`agentfem capabilities --json`. The audit checks that the declared maturity has
the corresponding interface, material-point, curve, post-processing, or
finite-element evidence. It intentionally does not promote an experimental
capability merely because its present tests pass; external validation,
generality, and stated limitations remain separate scientific claims.

```python
claim = verification.VerificationClaim.compare(
    name="beam_reference",
    observable="tip_displacement",
    actual=tip_u,
    expected=reference_u,
    reference="Euler--Bernoulli closed form",
    relative_tolerance=0.02,
    validity_domain="slender beam with negligible shear deformation",
    applicable=is_slender,
)
result.add_verification(verification.report(claim))
print(result.trust_level)
```

The validity domain is part of the claim. If `is_slender` is false, the claim
is `inconclusive`; AgentFEM does not declare either the finite-element result
or the reference formula correct by construction.

## Discretization evidence

`verification.ConvergenceStudy` consumes coarse-to-fine samples with a
strictly decreasing characteristic size. It reports the last relative change
and, for uniformly refined triples, an observed order. A required order that
cannot be estimated makes the claim inconclusive.

```python
study = verification.convergence_study(
    "hole_peak_stress",
    "maximum circumferential stress",
    (
        verification.ConvergenceSample(h1, s1, label="coarse"),
        verification.ConvergenceSample(h2, s2, label="medium"),
        verification.ConvergenceSample(h3, s3, label="fine"),
    ),
)
claim = study.verify(
    maximum_relative_change=0.05,
    minimum_observed_order=1.5,
)
```

Successive relative change is an initial engineering contract, not a complete
uncertainty estimate. Richardson extrapolation, GCI, singular-field handling,
and goal-oriented error estimators remain separate future consumers.

## Reliability-cliff suite

The first `CAE Reliability Cliff` contract targets silent AI-to-CAE failures:

1. a cantilever, mesh, support, and load are rotated together by 90 degrees;
   the scalar response must be invariant;
2. an ordered refinement study cannot be replaced by one converged solve;
3. a reduced theory outside its declared domain becomes inconclusive;
4. a campaign may require `minimum_trust_level="verified"` before producing
   training data.

The next suite families are deliberately not claimed complete:

- a perforated plate with hole-size and hole-circumference resolution sweeps;
- a T-stiffener compared through beam, shell, and three-dimensional models;
- clean-room and, when artifacts are available, exact CalculiX cross-solver
  reproductions.

## Simulation-to-learning gate

```python
dataset = campaign_report.require_dataset(
    minimum_samples=20,
    quality="engineering",
)
```

This requires every evaluator to return a `SimulationResult` that passed the
named policy. The accepted dataset records the policy decision in its metadata.
Advanced workflows can still use `minimum_trust_level="verified"` directly,
but the preset is safer because it also rejects failed runtime checks rather
than looking only at an ordered trust label.

## What verification does not mean

- A fixed-value Golden is a regression contract, not mesh convergence.
- Matching a second solver is cross-code evidence, not experimental validation.
- A low residual is not evidence that geometry, axes, units, mesh, or theory
  were chosen correctly.
- `validated` is used only when the claim kind is explicitly `validation` and
  all declared claims pass.
