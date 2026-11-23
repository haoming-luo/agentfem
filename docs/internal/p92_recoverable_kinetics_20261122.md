# Recoverable kinetics discriminator

## Decision before identification

The irreversible replacement worsened the first SLC post-creep hold entry,
despite improving its anchored relaxation shape. Preserve both recoverable
strains and their energies. Test force-dependent recoverable mobility alone;
do not combine it with directional recall, primary flow or history-dependent
static recovery. This is a research discriminator, not a production P92 law.

For each recoverable strain a, modulus H and time tau, define z=sigma-H*a,
s0=120 MPa, and

    adot = z/(H*tau) * abs(z/s0)^(m-1), m >= 1.
    stored energy = H*a*a/2.
    dissipation = z*adot >= 0.
    dual potential = s0*s0/(H*tau*(m+1))*abs(z/s0)^(m+1).

The original Kelvin law is exactly m=1. Constant-stress force decay for m>1
is z=z0/[1+(m-1)*abs(z0/s0)^(m-1)*t/tau]^(1/(m-1)); m=1 is exponential.
At zero stress the strain tends to zero: recoverability is retained. The
normalization is the existing backbone's fixed stress scale, not another
parameter. State layout and energy are unchanged; serialized material cards
explicitly record both exponents. The primary-creep discriminator rejects
nonlinear recoverable cards rather than silently combining hypotheses.

## Literature and limits

Benaarbia, Rouse and Sun (2018), DOI 10.1016/j.ijplas.2018.03.015,
https://nottingham-repository.worktribe.com/output/960836, establishes a
thermodynamic viscoelastic/viscoplastic decomposition for P91 at 600 C,
including recoverable contributions across different time scales. It supports
retaining recovery, but does not establish our nonlinear dashpot or a P92 card.

Kazemi et al. (2023), DOI 10.1016/j.euromechsol.2022.104821,
https://www.sciencedirect.com/science/article/pii/S0997753822002510,
uses two Kelvin-Voigt elements and a viscoplastic part, and compares prior-load
creep, relaxation and recovery in 10%Cr steel. Its accessible abstract supports
testing these responses together. We have not verified its full equations and
do not attribute the power law above to that publication.

Our force-power extension is an explicitly declared convex-dissipation
hypothesis. A thermodynamic construction alone cannot establish experimental
accuracy, uniqueness of mechanism or engineering lifetime qualification.

## Gates and budget

First run fixed arms (mfast, mslow)=(1,1),(2,1),(1,2),(2,2) from the same
archived seed, with actual RF/SLR/SLC control paths and only cycles 1--3.
Inspect creep strain, hold entry and actual hold evolution together; anchored
shape alone is not acceptance. Do not use later cycles to select the arm.

Before any fit, check constant-stress analytic creep and recovery, signed
symmetry, energy balance, split continuation and independent integration.
If a matched fit is warranted, replace the two weak branch-A static-recovery
coordinates with the two exponents, holding those static coefficients at seed
values. Keep 22 candidate directions, 10 retained directions, norm bound .3,
40 optimizer evaluations, unchanged objective and derivative/observability
gates. No extra fitted coordinates, wider bounds or protocol-specific coefficients.
Freeze before continuous ten-cycle replay. Previously inspected cycles 4--10
are development validation, not blind experimental data.

All work remains on PR #197, unmerged and unreleased.

## Fixed-arm checkpoint

The original arm reproduces cost 0.002576080096967705. Costs for (2,1),
(1,2), (2,2) are 0.0026827704966810003, 0.003108627839914325,
0.0030406792265425366. None improves the aggregate objective. Slow-only
nonlinearity reduces the third SLR peak error from 15.915 to 13.908 MPa,
but the first SLC post-creep hold's early stress secant becomes -5.026
instead of -4.815 MPa/s (observed -3.794). Its creep endpoint strain also
falls from 0.00103459 to 0.00100096 (observed 0.00106). Thus this is not
an accepted improvement. A bounded joint identification tests whether the
tradeoff survives adjustment of the existing coefficients; it cannot justify
an expanded parameter count or bounds. The declared center is (2,2), with
m=1+10^z; original branch-A static coefficients are held fixed.

## Interpretation checkpoint

The BAM original paper, Sonntag et al., DOI 10.1016/j.ijfatigue.2022.107381,
https://d-nb.info/1275082467/34, sections 3.1--3.3, distinguishes the shorter
post-creep reload and competing hardening/recovery during stress- versus
strain-controlled holds. Its mechanism-map discussion is a qualitative
interpretation, not a unique constitutive equation or fitted parameter card.
Do not encode an SLC/SLR label as a material mechanism.

An added diagnostic integrates signed flows at the same stress endpoints and
checks their sum against total nonelastic strain. For the matched baseline,
the first SLC hold near 210 MPa has average contributions A=3.277e-6,
B=7.555e-6, fast recovery=-1.713e-6, slow recovery=1.103e-6 /s. Total predicted
rate is 1.022e-5 /s versus observed 4.645e-6 /s. The two irreversible
contributions alone already exceed the observed total. This does not prove
their microscopic identities; it warns against attributing the whole error
to the slow recoverable dashpot. Preserve signed recovery, rather than
reporting absolute contributions as positive mechanism fractions.

## Matched and frozen result: not accepted

The two derivative steps and the local observability check pass. Identification
converges in six optimizer evaluations (216 total residual calls), with powers
1.94802426 and 1.82046490. Cost .0026024071014057763 is 10.23% higher than
the matched linear baseline .0023607800272608744. The largest cost increase
is SLC reversal loading, not its long creep dwell.

All ten continuous cycles complete for RF, SLR and SLC. Independent BDF and
waveform-refinement checks pass for all three, but all experimental response
screens fail. Failed-check counts are 19, 53 and 65, respectively; these
counts are not an accuracy ranking. SLR cycle-ten peak error is 8.58%, while
the first SLC post-creep hold anchor error grows from -0.302 to +2.901 MPa.
Its late stress error worsens from -14.152 to -15.785 MPa, and anchored
shape error from -13.850 to -18.686 MPa. Retaining recoverability was
necessary for this comparison but nonlinear mobility alone is not sufficient.

The matched card, frozen screens, residual decomposition and all fixed arms
are archived under `evidence/creep/2026-11-22-recoverable-kinetics-*`.
Numerical success is explicitly separate from failed experimental acceptance.
No response tolerance, fitting dimension or search radius was changed.

Parameter-accounting correction from final review: freezing the two A static
rates at seed values is not the same as removing them from the material law.
The two nontrivial recoverable exponents remain additional constitutive
constants even though they replace two optimization coordinates. This is an
equal-fitting-dimension diagnostic, not a strictly coefficient-neutral
replacement. Its failed evidence is retained, but it cannot satisfy the
user's no-additional-parameters promotion requirement. The separate mobility
law uses no new fitted or independent material coefficient.
