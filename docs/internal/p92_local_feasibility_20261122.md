# Local joint-feasibility discriminator

Lower sampled least-squares cost has not delivered simultaneous hold-entry,
creep and relaxation accuracy. Before inventing another mechanism, distinguish
a local tradeoff from the scalar objective's compromise. This is not another
parameter search or a claim of global infeasibility.

For the matched original and recoverable-memory mobility fits, retain their
exact seed, fitted ten coordinates, orthonormal basis and radius 0.3 box.
Replay only cycles 1--3. Collect each resolved response error divided by its
unchanged limit. Also report the first sampled stress error at each strain
hold against the existing 5 MPa / 5% stress scale as an additional diagnostic,
not a retroactive change to the acceptance policy. Unresolved observations
remain unresolved and do not become zero residuals.

Compute the local derivatives at two bounded difference sizes, 0.001 and
0.0005. Require relative Jacobian discrepancy <=0.002 and identical metric
identities. A linear program minimizes the maximum predicted normalized
error within the original coordinate box. Do not claim a nonlinear lower
bound: this linearization can be inaccurate. Replay the proposed change at
the three predeclared fractions 1, 1/2 and 1/4, without repeating the linear
program or shifting its center. Record numerical/path failures unchanged.

This is the same fixed diagnostic budget in both arms: one baseline, forty
perturbed three-protocol replays and at most three candidate replays. The
accepted model and all late-cycle evidence remain unchanged. Only a candidate
that reduces the maximum violation and each protocol's maximum, without
losing a previously passing resolved check, merits a separately frozen
continuation; even then it is not an engineering acceptance. Otherwise stop.

## Outcome: neither arm advances

The original card's derivative discrepancy is 0.00065852, below the
predeclared 0.002 diagnostic limit. Its initial maximum normalized violation
is 8.75297. The three prescribed nonlinear replays reduce it to 7.47729,
8.09955 and 8.42424, but none satisfies the joint admission rule. Reducing
the worst error alone is insufficient. This arm uses 44 response evaluations.

The mobility card's derivative discrepancy is 0.0576612, exceeding the same
limit. Its initial maximum is 8.12162. The audit therefore stops after 41
evaluations without using an unreliable local derivative to propose a fit.
This is a local diagnostic failure, not evidence of physical impossibility
or a reason to relax experimental limits. Neither card receives another
frozen continuation from this audit. All prior frozen failures remain.

The two `2026-11-22-*-local-feasibility.json` records preserve the source,
seed and driver identities, normalized metrics and candidate outcomes.

## Derivative follow-up declared before rerunning

The retained per-row perturbations locate the large mobility discrepancy in
four already nearly satisfied absolute-error metrics (normalized errors
0.0034--0.0179), dominated by coordinate zero crossing their residual signs.
Differentiating absolute values across a zero creates a kink. This is not
evidence that the material differential equations are inaccurate.

Repeat both fixed-card diagnostics with signed point/secant errors and both
inequalities `r + J*du <= t` and `-r - J*du <= t`. RMS remains a norm; its
negative inequality is redundant. This is the same minimax constraint, not
a changed experimental limit or a new material parameter. Keep the same
two difference sizes, derivative check, directions, box, 44-call ceiling
and three candidate fractions. Check signs against the original gate
magnitudes; unknown metric types must fail. Preserve the original absolute
diagnostic and its failure. Do not infer feasibility from the linear program.

The two-sided epigraph is the standard infinity-norm approximation formulation
in [Boyd and Vandenberghe, chapter 6](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
The implementation uses the explicit inequality/bound interface documented
by [SciPy HiGHS](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html).
Convexity concerns this local affine subproblem, not the original material
identification or its experimental validity.

Both signed audits complete the identical 44-response budget. The original
card reproduces its previous result. The mobility derivative discrepancy
falls from 0.0576612 to 0.000421533 without changing the material equations
or perturbation sizes. Its three actual maximum violations are 7.51616,
7.83822 and 7.98607, down from 8.12162, but every candidate fails the
predeclared joint admission rule. No later-cycle continuation is admitted.
The absolute-error kink was a diagnostic artifact; resolving it does not
resolve the experimental conflict. This distinction is retained in both
signed evidence records, not concealed by replacing the earlier failed audit.
