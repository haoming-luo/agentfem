# P92 response milestone: decisions before the next fit

PR #197 remains open. These are bounded project screening gates for the BAM
620 C, 0.2% amplitude protocols, not a universal material accuracy standard or
an engineering life allowable. Declare a new criterion prospectively if a
different intended use requires one; never relabel an old failure by silently
changing its tolerance.

## Sequence and evidence

1. **Control discrimination:** two frozen cards, linear ramps versus measured
   endpoint-preserving shapes, three protocols and ten continuous cycles.
   Publish offsets and source indices. Quantify, rather than assume, the
   contribution of ramp approximation. Remaining controller uncertainties
   stay separate from material mismatch.
2. **Mechanism discrimination:** only after (1), choose the smallest supported
   change. Compare reversal entry, early/late holds and cycle evolution, not
   only a scalar objective. Keep failed hypotheses. No cycle-index correction,
   hidden-state reset, or forced observed response.
3. **Joint identification:** freeze data indices, scales, bounds and evaluation
   budget before optimization. Report active bounds and sensitivity rank;
   optimizer success is not a material acceptance decision. Use cycles 1–3
   only, freeze the card, then replay previously inspected cycles 4–10.
4. **Independent response assessment:** obtain genuinely uninspected compatible
   records, or label a different cohort as transfer. Already inspected cycles
   cannot become blind holdouts by renaming them.
5. **Life assessment:** separate source failure definitions, specimen pedigree,
   censored outcomes and independent prediction errors. Three protocol lives
   are not a replicated population from which to certify reliability. Response
   screening does not authorize a lifetime claim.
6. **Product admission:** only validated reusable behavior enters the installed
   material workflow, with material-point/structure comparison, consistent
   tangent where required, physical time, accepted-state/restart evidence and
   a runnable case. No BAM-specific correction in the production solver.

## Numerical gate (retained)

Independent Radau/BDF and identical-waveform knot subdivision must agree on
completed paths. Stress-like endpoint states and sampled extrema: below
1e-4 MPa; strain-like states: below 1e-8. These limits are far below the
experimental discrepancies and do not substitute for response accuracy.

## Prospective response screening gate

Apply separately to every protocol and complete cycle, including valleys and
all holds. This gate is deliberately independent of the optimization weights.

| Observable | Project screen |
| --- | --- |
| Sampled peak and valley stress | Absolute error <= max(5 MPa, 5% of absolute observed value) |
| Hold stress history | RMS and last-sample error <= max(5 MPa, 5% of maximum absolute observed hold stress) |
| SLC hold strain history | RMS and last-sample error <= max(2e-5, 10% of observed hold strain excursion) |
| Early and late hold secants | Relative error <= 20% when the observed change spans at least five recorded resolution units |
| Path admissibility | All requested cycles complete without imposed hidden-state resets or incompatible post-hold reversal |

The source's recorded strain resolution is 1e-5 and stress resolution 0.01 MPa.
Resolution is not measurement uncertainty. A secant below the signal criterion
is **unresolved**, not passed, and requires a longer observation window or
better data before being used as a material-accuracy assertion. Report signed
errors, source times and resolution ratios. These choices define a development
screen; a final engineering use still needs repeatability/uncertainty and an
explicit applicability review. Do not claim standards compliance from them.

## Victory markers

- **A:** reproducible loading-error budget, even if the material still fails.
- **B:** one frozen candidate passes all three protocol response screens;
  parameter observability limitations remain visible.
- **C:** independent response evidence supports a specified prediction domain.
- **D:** compatible independent lifetime data support a stated uncertainty
  assessment; no unjustified reliability or design-life guarantee.
- **E:** an ordinary installed project reproduces the admitted calculation
  without AI intervention, including restart and standard outputs.

Each coherent batch uses its impact-selected local tests and one PR update.
Research iteration alone does not trigger repeated full CI. No merge or release
is authorized by these markers or by green CI alone.
