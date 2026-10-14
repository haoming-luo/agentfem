# Bounded composite delivery sprint — 2026-10-14

Window: 05:27–09:27 UTC. No scheduled continuation and no new full industrial
run required for this packaging sprint. The retained complete result is the
reference, not a result produced by the new candidate.

## Release-candidate gates

1. Keep initial numerical-mass generation in core; report the controlling
   operator/node and distinguish physical mass, reference estimate and evolving
   nonlinear validity. Never invent a material tangent/envelope.
2. Exercise the existing elastic bond plus unilateral contact in tension,
   compression, shear and closure. Preserve the explicit independent-contribution
   choice; do not silently change the model to match a third-party interpretation.
3. Admit pure accepted-station history requests with dyadic retries, preserving
   subincrement work accounting, macro rollback and restart schema. Expose
   detached accepted contact observations instead of case-owned residual access.
4. Transport named material-state components through the common cell/output
   contract. No case-written connectivity, smoothing or interpretation of an
   unknown material column. Keep private material semantics outside core.
5. Deliver one project with readable modeling truth, separate operational
   controls, source-build preparation, standard CLI run/restart, verification
   notes and a short AI handoff. Do not ship old nodal-inertia arrays or assume
   a particular CPU count as a scientific compatibility rule.
6. Test source and installed candidate, checkpoint continuity and independent
   ParaView volume/warp. Run representative checks within the remaining window;
   preserve failed sensitivity evidence and label unfinished full-run acceptance.

## First implementation gate

123 focused real-FEniCSx checks pass: response material/substeps/observations,
deformable contact, initial mass planning, source-node and boundary mapping,
bond/compression composition, transient journals and checkpoint recovery.
Ruff and generated API checks pass. Original full-mesh source identities and
oriented Q1 faces match the retained private extraction; nodal integration
measures agree within the predeclared floating-point tolerance. This does not
constitute a new full dynamics run or speed measurement.

## Scientific boundary

InitialMassScaling is not Abaqus variable mass scaling and not an all-state
stability certificate. The private material supplies response, not a reviewed
all-state tangent bound. The current contact route is declared one-sided Q1
penalty, not unrestricted ALL EXTERIOR/self-contact. Independent elastic bonding
and compression contact are explicit operators, not an asserted vendor-equivalent
combined cohesive-contact implementation. Internal work is not stored energy.

The existing early time/mass screens remain failed/inconclusive evidence; their
thresholds will not be relaxed after inspection. Later comparisons require
nontrivial common load stations, response-specific absolute and relative bounds,
and the completed trajectory. Package readiness, implementation parity, numerical
sensitivity and independent Abaqus agreement are separate gates.

## References reviewed

- [Abaqus cohesive contact](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEITNRefMap/simaitn-c-cohesivebehavior.htm):
  node-to-face Explicit formulation; initially eligible bonds; the distinction
  between permanent-bond and regular-contact behavior must be retained.
- [Abaqus mass scaling](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-massscaling.htm):
  fixed/variable scaling are numerical controls whose effect on inertia requires
  assessment, not material-density edits.
