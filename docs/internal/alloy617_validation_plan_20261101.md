# Public-material validation: source gate and execution plan

Status: source audited; **no calibrated Alloy 617 simulation or held-out
prediction has passed**. Do not confuse the existing synthetic/analytical
verification suite with experimental material validation.

## Selected public source

[ORNL/TM-2022/2517 (July 2022)](https://info.ornl.gov/sites/publications/Files/Pub179816.pdf)
is a useful candidate: Alloy 617 at 950 °C, tensile holds, ordinary strain
control and separately identified elastic follow-up. PDF pages 15 and 19
(printed pages 5 and 9) were visually checked, not only text-extracted.
Table 2 gives A-6/F-5: strain range 1%, hold 180 s, lives 350/450 cycles;
A-13: 1%, 600 s, 300 cycles. These are ranges, not amplitudes. Keep repeats
and test-family labels separate. Table 4 and Equation 8 include negative
softening parameters and damage-dependent evolution. They cannot be copied
into AgentFEM's current nonnegative-saturation Chaboche model as an equivalent
law. The report also separates cavity-growth and time-fraction assumptions.

[ASTM E2714-26's public catalogue](https://store.astm.org/standards/e2714)
identifies the creep-fatigue test-method family. The complete normative text
has not been reviewed; neither a dataset audit nor a passing calculation
establishes ASTM or component design-code compliance.

## Admission decisions

1. Retain the source version, specimen/test-family identifiers, temperature,
   strain range versus amplitude, rate, hold direction/duration, failure
   definition and runouts. Reconcile duplicated records against their original
   reports before joining datasets; never silently overwrite a discrepancy.
2. Extract stress–strain loops and hold-relaxation histories with explicit
   digitization uncertainty, or obtain numerical author data. A life table
   alone cannot identify a unique deformation model.
3. Freeze a compatible local law and its validity domain before fitting. If
   cyclic softening is needed, use the existing external material boundary
   and verify its tangent/state/work first. Do not remove hardening-model
   safeguards to make coefficients fit or add a material-specific core solver.
4. Split by complete loading conditions/specimens, **not time samples from
   the same curve**. Reserve at least one strain/hold condition for prediction.
   Retain replicate scatter; do not tune against the held-out result.
5. Run material point → homogeneous structure → hold/reversal → restart.
   Compare loop stresses, relaxation shape and strain partition before life.
   Use the new co-located history channel, not unrelated field extrema.
6. Only then assess a declared rupture/interaction model within its calibration
   domain. Separate numerical error, fit error, prediction error and design
   acceptance. Publish failures and remaining parameter/data uncertainty.

## Present executable evidence and next composition

Existing Norton relaxation, elastic follow-up and manufactured nonuniform
creep/plastic/thermal fields remain the numerical oracles. The NAFEMS R0027
thick-cylinder gate is a separate public structural reference. None supplies
an Alloy 617 fit.

The first composition in this slice is nonuniform accepted temperature plus
co-located histories, with existing regional **pure creep** and homogeneous
joint creep/plasticity tested separately. The serial thermal v1 restart identity
bug was found by this acceptance work and corrected. New regional joint
plasticity, temperature-dependent generic finite-strain user materials, contact
and MPI admission remain separate tasks. Priority after source reconciliation:
the smallest real local law and its temperature/state contract, not all four
combinations at once.
