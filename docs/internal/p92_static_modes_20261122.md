# Static recovery modes, without new time scales

Prospective research discriminator after the critical-state comparison and
conditional incoming-state audit. Keep the original smooth dynamic recovery,
flow laws, Kelvin strains and quadratic energy. Do not combine this with the
critical-state law or prior mobility/strength hypotheses.

## Motivation and derivation

The current card has very small branch-A static rates. A weak logarithmic
sensitivity near a rate of zero does not establish that static recovery of
that component is physically irrelevant. Nevertheless, widening those rate
ranges to chase the hold data is not authorized. Instead ask a different
structural question: are the existing static time scales assigned to the
appropriate combinations of memory variables?

[Wolff, Bohm and Taleb (2011), section 6, equations 6.2--6.4](https://journals.ub.ovgu.de/index.php/techmech/article/download/766/743)
derives cross-mechanism recovery terms and their positive-semidefinite
dissipation condition. Its displayed terms are flow-dependent dynamic
recovery, not our proposed static law. The following constant static mobility
construction is our own explicitly checked extension, not a published P92
parameterization or an implementation of that entire paper.

For corresponding backstress components define normalized forces
q=(X_A/sqrt(H_A), X_B/sqrt(H_B)), with energy q.q/2. Originally the static
part is qdot=-diag(r_A,r_B)q. Replace this by qdot=-R q with
R=U diag(r_A,r_B) U.T, where U is an orthogonal shared/difference-mode
transformation. Dissipation is q.T R q>=0, and both eigenvalues (the existing
inverse time scales) remain exactly unchanged. This changes dissipative
coupling, not the energetic coupling already tested in the preceding batch.

Predeclare only three arms: original diagonal mobility, and the two possible
shared/difference assignments for the **slow backstress pair**. For the new
arms U has entries +/-1/sqrt(2). No angle, coefficient, state or time scale is
fitted or added. The fast backstress pair and both recoverable strain branches
are unchanged. These are discrete model hypotheses; calling them parameter
neutral does not establish their physical validity.

Coupled backstresses need not stay in the original rectangular invariant
domain. Do not clip them or claim that invariant. The retained nonlinear
dynamic dissipation, total energy, nonnegative dissipation and finite state
checks remain mandatory. The state identity is distinct from old restarts.

## Gates

Check zero-arm equivalence, matrix eigenvalues, inactive-flow matrix-exponential
solution, reflection, mechanical power and integrated energy, restart, and
independent integration. Then evaluate all three early protocols with the
same frozen coefficients. Reject isolated improvements that worsen creep
accumulation or post-creep hold kinetics. Only an overall useful fixed-arm
result can justify same-budget joint identification. No range expansion,
new production API, PR merge, or release.

## First result and separate recoverable-memory arm

The three fixed arms complete with costs 0.0023607800272608744,
0.0025517872891463534 and 0.0034247482921220254. Neither cross-backstress
assignment warrants a fit. This rejects these discrete assignments, not all
coupled recovery models. The original arm reproduces the archived control.

A separate final two-arm test will couple the slow B backstress and slow
Kelvin strain **dissipatively**, retaining the original backstress-to-backstress
mobility. This is distinct from the previously rejected coupling of their
stored energies. In normalized coordinates z=(X_B2/sqrt(H_B2),sqrt(K_s)*a_s),
the dissipative forces are g=(-z_1,sigma/sqrt(K_s)-z_2). Their original
mobility is diag(r_B2,1/tau_s). Apply the same two fixed orthogonal
shared/difference transformations to that matrix, keeping its eigenvalues.
The inelastic hardening drive of X_B2 and every other term remain unchanged.
The Kelvin rate must also enter the total strain/stress update and external
work; dissipation changes from g.T D g to g.T R g. Keeping only the backstress
correction would violate the power identity and is not allowed.

This is an independently derived positive-semidefinite mobility hypothesis,
not a claim that a cited paper supplied this particular P92 coupling. It adds
no coefficient, state or fitted angle. Verify power, sign symmetry, inactive
linear matrix-exponential response, independence of integration and restart
before replay. Do not combine it with the rejected cross-backstress arms.
Same fixed-card three-protocol screen, no extra fit unless the joint evidence
warrants it. Preserve every failure.

Both recoverable-memory assignments also worsen the joint objective: costs
0.003062809804631075 and 0.004316671383240117, versus the reproduced
control 0.0023607800272608744. They are not selected for identification.

## Separate assignment discriminator

The conditional incoming-state audit leaves excessive late relaxation-rate
error even after improving the early SLC segment. Both 45-degree assignments
above mix the two rates, so neither isolates their allocation to A versus B.
One final, discrete assignment comparison is therefore declared: exchange
only r_A2 and r_B2. Leave the fast rates and all Kelvin terms unchanged.
This is a 90-degree permutation, not another fitted rotation or a wider rate
range. The quadratic energy, positive dissipation and the unordered pair of
time scales are unchanged. Each component again has an ordinary diagonal
static decay. Unlike cross coupling, this also retains the original rectangular
backstress invariant domain.

Physical question: does restoring slow static recovery in the low-threshold
A branch while preserving B's slow memory reduce the early/late conflict?
This is a hypothesis about parameter assignment, not evidence that A and B
have identified microscopic meanings. Compare the unchanged original card
and this single exchange on early RF/SLR/SLC only; no scans, new rates or
conditional-state injection. If the joint creep/reload/relaxation evidence
does not improve, reject it without fitting.

The exchange completes but raises cost to 0.00520411597. The first SLC
creep endpoint rises from 0.000991685 to 0.001132964 (0.00106 observed),
while the subsequent hold endpoint falls from 160.728 to 153.117 MPa
(174.88 observed). Reject this assignment; do not fit it or scan more
rotations. This result does not exclude all possible static-recovery laws.
The three separate probe archives retain every declared arm and the exact
control: `2026-11-22-static-modes-probe.json`,
`2026-11-22-recoverable-modes-probe.json`, and
`2026-11-22-static-exchange-probe.json` under `evidence/creep/`.

The remaining distinct assignment question is the slow Kelvin/B pair, not
another A/B rotation. Declare exactly one diagonal exchange of their existing
inverse times: r_B2 becomes 1/tau_s, and tau_s becomes 1/r_B2. All moduli,
amplitudes and other rates stay fixed. In the current card this assigns the
roughly 630 s time to recoverable strain and 84 s to B's slow backstress.
No time scale is introduced or enlarged; the existing pair is reassigned.
Unlike the previous 45-degree test there is no cross term. This can alter
late relaxation capacity, but may spoil the observed creep accumulation.
Check exact equivalence with the original equations under that permutation,
analytic constant-stress response, both controls and energy before the same
three-protocol screen. Do not use the conditional reconstructed state as input.
Reject a worse joint result; this does not authorize a scan of time scales.

The Kelvin/B exchange completes but raises the three-protocol cost to
0.00496868796, versus 0.00236078003. Reject it without identification.
The unchanged control and candidate, including all response screens, are
retained in `evidence/creep/2026-11-22-kelvin-time-exchange-probe.json`.
