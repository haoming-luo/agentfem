# Finite-J2 analytical-plane contact candidate — 2026-10-16

Status: bounded experimental composition, not a released industrial forming
capability. This note is an evidence snapshot for the 0.4.3 decision gate.

## Scientific and architecture boundary

- Ordinary three-dimensional strong-boundary finite-strain logarithmic J2.
- One analytical rigid plane with frictionless one-sided penalty contact; fixed
  or following prescribed translation/rotation. The existing contact potential
  contributes residual and semismooth tangent to the J2 Newton form.
- The J2 quadrature transaction remains the State owner. Accepted energy
  records material stored energy, plastic dissipation, contact penalty
  potential, natural/strong-boundary work, and generalized tool work separately.
- Failed finalization restores displacement, material history, energy ledger
  and tool pose. Serial checkpoints include the contact identity and accepted
  work history; a changed penalty is rejected on restore.
- No triangular tool geometry, edge/vertex transition, friction, periodic MPC,
  mixed pressure, cross-rank restart or MPI composition claim.

## Evidence collected on the local FEniCSx 0.11 environment

The source checkout passed 91 related tests across finite-J2, existing contact,
public workflows and validation after the rotating-tool additions. Seven
new/contact tests also passed from an isolated installed pure-Python wheel,
with `agentfem.__file__` resolved under its install target. The relevant
platform CI and independent structure-level reference remain open gates.

The one-cell uniaxial elastic limit is independently solved from
`E log(1+u)/(1+u) + k u = F` for a fixed plane and from
`E log(1+u)/(1+u) + k(u+d) = 0` for a moving plane. Numerical fields match
these scalar roots within the declared test tolerance. A separate active
plastic-contact patch has positive PEEQ, balances strong reaction + contact
resultant + natural force to `1e-6` absolute, and retains the same accepted
state after interrupted serial restart.

| Active plastic-contact path | 3 increments | 6 increments | 12 increments |
| --- | ---: | ---: | ---: |
| Right displacement | 0.012399086 | 0.012403449 | 0.012405173 |
| Maximum PEEQ | 0.013502257 | 0.013518648 | 0.013528577 |
| Relative accepted-path ledger residual | 0.0063534 | 0.0026861 | 0.0009795 |

For six increments, prescribed translation produced tool work `0.016685405`
with relative ledger residual `1.54e-5`; translation plus a `0.02` rad small
rotation produced tool work `0.024131152` with residual `3.94e-5` and a
nonzero contact moment. These are internal small-model checks, not an external
forming validation or a claim of exact energy conservation at finite step.

## Next gates before release promotion

1. Complete final source and installed-wheel regressions; review the diff and
   old-provider dispatch; run relevant platform CI without changing the
   established explicit production case.
2. Test a nonuniform contact patch under mesh/step/penalty changes and include
   a public independent structure-level reference. A homogeneous one-cell
   spring pair is necessary but insufficient for broad contact validation.
3. For discrete rigid tools, implement reviewed current-geometry projection,
   facet/edge/vertex transitions and consistent residual/tangent in a separate
   candidate; compare a triangulated flat tool to the analytical plane.
4. Only then compose periodic MPC, ensuring both contact residual and tangent
   participate in the same reduction, trial/rollback and work ledger.

Performance is a separate evidence lane: this composition has no measured
full-run acceleration claim. Optimize body/contact/data flow against a frozen
installed candidate and accepted physical stations, per
`skills/agentfem-performance`; do not change penalty, inertia or time grid to
manufacture a timing win.
