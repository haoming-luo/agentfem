# P92 recovery batch: negative hypotheses, stronger discrimination

Scope: PR #197 only; no merge, release or production material promotion.
The BAM source hashes, measured switching paths and experimental response
thresholds remain unchanged. No per-protocol correction or late-cycle fitting
was introduced. Cycles 4–10 have already been inspected and are development
checks, not a blind holdout or engineering life qualification.

## Outcomes

| Experiment | Outcome | Decision |
| --- | --- | --- |
| Matched linear/nonlinear static recovery | Costs 0.002361 / 0.002396; both numerical checks pass, all three protocol response screens fail | Reject nonlinear recovery as a demonstrated upgrade |
| Positive-definite energetic mechanism coupling | Fixed-card aggregate cost increases | Do not identify or promote this candidate |
| Bounded history-dependent kinetic resistance | All four fixed-card probes complete; nonzero interaction worsens aggregate cost | Do not expand bounds to chase a fit |
| Equal-stress positive holds | SLC after creep has slower observed flow than SLR; current card predicts similar rates | Retain as a mechanism discriminator, not proof of a microscopic law |

The main result is a sharper test of history interaction rather than a new
accepted P92 card. The next hypothesis must explain slow post-creep SLC
relaxation without worsening inherited hold stress, reversal or SLR peaks.
Use published equations with verified definitions where available; do not
transfer unverified table transcriptions or different-heat constants.

## Software verification

The research references check scalar mechanical power, dissipation,
zero-interaction limits, sign reversal, invariant domains, independent
Radau/BDF integration, split continuation and rejected-input preservation.
Frozen comparisons include every internal-state coordinate and cycle extrema.
The source-table helper is explicitly named `unverified_table2_transcription`.

Validation on 2026-11-21: the complete `_P92_CONTROL_TESTS` tuple plus
`tests/test_ci_validation_scope.py` passed: **350 tests in 8.83 seconds** in
fenicsx-env, with numeric threads limited to one. Ruff passed for all changed
Python files; `git diff --check` passed. The initial restricted-environment
test process ended with exit 143, without an assertion failure; it is retained
as an interrupted attempt, not counted as a pass. A subsequent complete run
produced the test count above. No MPI or full composite run was started.

Research JSON artifacts retain source and implementation hashes and explicit
qualification flags. Compact frozen archives omit trajectory arrays but keep
response screens and independent/refinement comparisons. Full local run files
are not replaced or deleted. CI is requested only for the consolidated batch.
