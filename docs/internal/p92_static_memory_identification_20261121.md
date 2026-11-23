# Matched identification with substituted static-history directions

Declare before optimization: keep 22 candidate coordinates and ten optimized
SVD coordinates, maximum norm 0.3, 40 optimizer evaluations, central derivative
scales 0.001/0.0005 and all original early residual weights. No later-cycle
observations or widened bounds. Compare baseline with shared-history recovery.

The prior matched-baseline derivative audit identifies A.static_recovery1/2
as the weakest columns: relative norms 3.25e-6 and 1.41e-4 of the largest.
Freeze these two coefficients in the history candidate. Replace their slots
with the shared saturated fraction rho and history rate b. Fitted-coordinate
count does not increase. The two old nonzero constants remain in the law,
so this is not a reduction-neutral count of all constitutive coefficients.
Start at rho=0.5 and b=250/strain, as in the fixed probe.
Use log10 odds rho/(1-rho) for the fraction and log10 multiplier for b; these
dimensionless coordinates preserve positive bounded recovery. Other constants
start from the same original spectrum card as the baseline. E remains fixed.

Recompute both hypotheses' two-scale Jacobians. Retain prior numerical gates:
relative Jacobian difference <=0.002, top-ten subspace distance <=0.02,
smallest retained singular value >=1e-5 and retained condition number <=1e4.
For the history candidate, also report how its two columns project outside
the leading eight directions of the other twenty columns and how strongly
the top-ten retained subspace contains the two new coordinates. These are
local numerical diagnostics, NOT confidence intervals or global/statistical
identifiability. Require finite diagnostics, two-coordinate retained Gram
minimum eigenvalue >=1e-4 and residualized two-column singular ratio >=1e-3
before fitting; otherwise record unresolved and do not force optimization.

Each fit uses its own verified top-ten basis, the same norm budget and original
hold/reversal/extrema objective. Report raw residual-call counts too: equal
optimizer limits do not imply equal wall time. Reject invalid trial paths and
freshly evaluate the final card. Freeze converged parameters before ten-cycle
checks with independent integration and refinement. Preserve all response
gates; lower aggregate cost cannot compensate for an unacceptable hold.

An unsuccessful audit or fit is retained as evidence. Do not relax its bounds,
change weights or add parameters after inspecting the result. PR #197 remains
open; this is research identification, not material or engineering-life admission.

## Interpretation and literature

The recovery hypothesis follows the functional motivation in
[Zhang et al. (2020), equations 9–11](https://doi.org/10.3390/met10030307),
not its material constants. The two-flow shared history remains our declared
extension. Identifiability must be assessed on the actual observations:
[Morand et al. (2024)](https://doi.org/10.1007/s11340-024-01057-x)
demonstrate ill-posed Chaboche-type parameter identification for copper-alloy
relaxation and discuss sensitivity to starting points. Their material and
surrogate performance do not transfer to P92. This motivates reporting
observable combinations, not interpreting a converged optimizer as unique
physical constants or using a faster optimizer to conceal missing information.

The residualized history diagnostic conditions on the eight leading nuisance
directions, not all twenty original parameters. Passing it permits the bounded
ten-coordinate experiment; it does not establish two individually identifiable
physical parameters when every nuisance coordinate is free. The coefficient
count, retained dimension, scaling and limits must accompany its interpretation.

## Completed matched experiment (2026-11-22)

Both two-scale audits passed. Baseline/history relative Jacobian differences
are 0.0007064/0.0009087 and subspace distances 0.004623/0.002645. The history
residualized singular ratio is 0.26887, but the retained-coordinate Gram
eigenvalues are only 0.00015899 and 0.031075. The weakest exceeds the declared
0.0001 gate narrowly. These are INITIAL local diagnostics, not a post-fit
confidence analysis or evidence of strong individual identifiability.

Both optimizers terminate by the unchanged gradient criterion in eight
function evaluations. Actual response calls differ: 238 baseline, 258 history
(the same evaluation cap is not the same realized cost). Two versus three
retained coordinates finish on bounds; do not widen bounds after this result.
The candidate freezes rho=0.50974606 and b=242.18519/strain. The two original
A static coefficients remain fixed. All other coordinates follow the same
declared backbone mapping; no additional state or fitted coordinate is added.

| Quantity | Matched baseline | Shared history |
| --- | ---: | ---: |
| Early joint objective | 0.00236078003 | 0.00231955771 |
| SLR cycle-3 peak error, MPa | 12.6174 | 11.9509 |
| SLR cycle-10 peak error | +8.9314% | +8.6212% |
| RF cycle-10 peak error | +7.7252% | +7.5616% |
| SLC cycle-10 peak error | -1.0349% | -0.6844% |
| SLC first inherited hold stress, MPa (observed 121.70) | 123.0871 | 124.0903 |
| SLC first post-creep hold end, MPa (observed 174.88) | 160.7281 | 160.6938 |
| Same SLC hold stress RMS, MPa | 10.4489 | 10.7783 |
| Failed response checks across RF/SLR/SLC | 135 | 135 |

The 1.7461% objective reduction does not solve the competing holds. The
candidate's SLR failures increase from 40 to 44, while RF goes 21 to 20 and
SLC 74 to 71. Both retain ten unresolved SLC secants; those are not passes.
In particular, an improved later peak coexists with a worse inherited SLC
stress and post-creep RMS. Reject promotion as an accepted material upgrade.

All six continuous ten-cycle paths complete. Every independent-BDF and
waveform-refinement numerical comparison passes for all twelve state entries;
the maximum independent stress-state difference is below 6.3e-7 MPa. This
supports a model/identification discrepancy rather than integration error,
not engineering accuracy. Cycles 4–10 remain previously inspected development
data. The experimental response screens fail in all six paths.

Validation: 387 affected P92/CI-scope tests passed in 10.27 s. Ruff, generated
documentation and diff checks pass. New tests enforce
fixed fitting dimension, unchanged frozen coefficients, power accounting,
rejection of unobservable derivatives and refusal to let observable nuisance
parameters conceal unobservable history directions. No public material or
solver default changes.

Records: `evidence/creep/2026-11-21-static-memory-matched-{baseline,shared}.json`
and their `-frozen.json` companions. Full local trajectories are retained under
`/private/tmp/p92-static-memory-matched-*-frozen.json`; compact repository
records retain all response gates, numerical checks, selected holds and the
original-file hashes. The hashes distinguish original and normalized archives.

Next gate: do not extend this candidate's parameter bounds or re-fit later
cycles. Diagnose the EARLY residual projections by protocol and segment to
separate the descent directions that improve SLR peaks from those worsening
SLC inherited stress and hold slopes. Use that conflict, with a physical
mechanism and an unchanged parameter budget, to choose the next substitution.
A statistical or final-card identifiability claim requires a separate audit;
this experiment does not provide one. PR #197 remains unmerged and unreleased.
